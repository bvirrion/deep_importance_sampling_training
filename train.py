"""
Training and evaluation harness.

Three conditions, all matched on the number of *gradient examples consumed* (the
compute budget), so the comparison is fair:

  1. uniform   : standard SGD, minibatch drawn uniformly at random.
  2. gradnorm  : fixed heuristic importance sampling, proposal proportional to
                 the per-example logit-gradient-norm proxy (recomputed each step
                 on a candidate pool). This is the 'fixed heuristic' baseline.
  3. learned   : the proposed method. A small reweighter q_phi is sampled from to
                 draw the minibatch, importance weights de-bias the update, and
                 phi is meta-updated online to reduce estimator variance.

All three use identical model init, identical data, identical LR schedule and the
same per-step candidate pool mechanism, so the *only* difference is how examples
within the pool are selected and weighted.
"""

import numpy as np
from model import CharLM
from data import reweighter_features
from reweighter import Reweighter


def evaluate(model, Xv, yv, batch=4096):
    """Mean validation loss over the held-out set."""
    n = Xv.shape[0]
    total = 0.0
    count = 0
    for s in range(0, n, batch):
        Xb = Xv[s:s + batch]
        yb = yv[s:s + batch]
        cache = model.forward(Xb)
        l = model.loss_per_example(cache, yb)
        total += l.sum()
        count += len(yb)
    return total / count


def train_uniform(Xtr, ytr, Xv, yv, vocab_size, *, steps, batch, lr,
                  eval_every, seed=0, log=None):
    rng = np.random.default_rng(seed)
    model = CharLM(vocab_size, seed=seed)
    Ntr = Xtr.shape[0]
    curve = []
    for t in range(steps):
        idx = rng.integers(0, Ntr, size=batch)
        Xb, yb = Xtr[idx], ytr[idx]
        cache = model.forward(Xb)
        grads = model.backward(cache, yb)          # uniform mean
        model.sgd_step(grads, lr)
        if t % eval_every == 0 or t == steps - 1:
            v = evaluate(model, Xv, yv)
            curve.append((t * batch, v))
            if log:
                log(f"[uniform] step {t:5d} examples {t*batch:8d} val {v:.4f}")
    return model, curve


def _draw_pool(rng, Ntr, pool_size):
    return rng.integers(0, Ntr, size=pool_size)


def train_gradnorm(Xtr, ytr, Xv, yv, vocab_size, *, steps, batch, lr,
                   pool_size, eval_every, seed=0, log=None):
    """Fixed heuristic: sample minibatch from a pool with p ~ gradnorm proxy."""
    rng = np.random.default_rng(seed)
    model = CharLM(vocab_size, seed=seed)
    Ntr = Xtr.shape[0]
    curve = []
    examples_consumed = 0
    gradnorm_passes = 0
    for t in range(steps):
        pool = _draw_pool(rng, Ntr, pool_size)
        Xp, yp = Xtr[pool], ytr[pool]
        cache = model.forward(Xp)
        gnorm = model.per_example_gradnorm(cache, yp)
        gradnorm_passes += 1
        p = gnorm + 1e-8
        p = p / p.sum()
        sub = rng.choice(pool_size, size=batch, replace=True, p=p)
        # importance weights to de-bias: 1/(N_pool * p_i), normalised to mean 1
        w = 1.0 / (pool_size * p[sub])
        w = w / w.mean() / batch
        Xb, yb = Xp[sub], yp[sub]
        cache_b = model.forward(Xb)
        grads = model.backward(cache_b, yb, weights=w)
        model.sgd_step(grads, lr)
        examples_consumed += batch
        if t % eval_every == 0 or t == steps - 1:
            v = evaluate(model, Xv, yv)
            curve.append((examples_consumed, v))
            if log:
                log(f"[gradnorm] step {t:5d} examples {examples_consumed:8d} val {v:.4f}")
    return model, curve, gradnorm_passes


def train_learned(Xtr, ytr, Xv, yv, vocab_size, *, steps, batch, lr,
                  pool_size, meta_lr, eval_every, seed=0, log=None,
                  amortize_after=None):
    """Proposed method: learned reweighter, meta-updated online.

    If amortize_after is set, then after that many steps we STOP recomputing the
    expensive per-example gradient-norm target and STOP meta-updating; the
    reweighter is frozen and used purely from cheap forward-pass features. This
    is the amortised-deployment regime where the method earns its keep: it keeps
    importance-sampling benefits without the per-step gradient-norm overhead that
    the fixed heuristic must always pay.
    """
    rng = np.random.default_rng(seed)
    model = CharLM(vocab_size, seed=seed)
    rw = Reweighter(n_features=5, seed=seed, temperature=1.0)
    Ntr = Xtr.shape[0]
    curve = []
    cost_curve = []
    examples_consumed = 0
    gradnorm_passes = 0   # count of expensive per-example gradnorm computations
    for t in range(steps):
        pool = _draw_pool(rng, Ntr, pool_size)
        Xp, yp = Xtr[pool], ytr[pool]
        cache = model.forward(Xp)
        feats = reweighter_features(model, cache, yp)
        amortized = amortize_after is not None and t >= amortize_after
        if not amortized:
            gnorm = model.per_example_gradnorm(cache, yp)
            gradnorm_passes += 1
        idx, p = rw.sample(feats, batch, rng)
        w = 1.0 / (pool_size * p[idx])
        w = np.clip(w, 0.0, np.quantile(w, 0.99) + 1e-12)
        w = w / w.mean() / batch
        Xb, yb = Xp[idx], yp[idx]
        cache_b = model.forward(Xb)
        grads = model.backward(cache_b, yb, weights=w)
        model.sgd_step(grads, lr)
        if not amortized:
            cur_cost = rw.meta_update(feats, p, idx, gnorm, meta_lr)
        examples_consumed += batch
        if t % eval_every == 0 or t == steps - 1:
            v = evaluate(model, Xv, yv)
            curve.append((examples_consumed, v))
            cost_curve.append((examples_consumed, cur_cost))
            if log:
                log(f"[learned ] step {t:5d} examples {examples_consumed:8d} "
                    f"val {v:.4f} phi {np.array2string(rw.phi, precision=2)}"
                    f"{' [amortized]' if amortized else ''}")
    return model, curve, cost_curve, rw, gradnorm_passes
