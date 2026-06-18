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
from code.model import CharLM
from code.data import reweighter_features, reweighter_features_content
from code.reweighter import Reweighter


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
                   pool_size, eval_every, seed=0, log=None, debias=True):
    """Fixed heuristic: sample minibatch from a pool with p ~ gradnorm proxy.

    debias=True  -> importance-weighted, *unbiased* update (the estimator only
                    changes variance, not the expected optimisation trajectory).
    debias=False -> plain-mean update on the proposal-sampled batch. This is a
                    *biased* variant: it still samples more where the gradient is
                    large, but trains on the ordinary gradient, deliberately
                    steering the trajectory toward high-gradient examples (soft
                    hard-example mining) to reduce their error faster.
    """
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
        Xb, yb = Xp[sub], yp[sub]
        cache_b = model.forward(Xb)
        if debias:
            # importance weights to de-bias: 1/(N_pool * p_i), normalised to mean 1
            w = 1.0 / (pool_size * p[sub])
            w = w / w.mean() / batch
            grads = model.backward(cache_b, yb, weights=w)
        else:
            grads = model.backward(cache_b, yb)   # plain mean -> biased
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
                  hidden=32, meta_burst=100, refresh_period=1000, debias=True):
    """Proposed method: neural reweighter with a periodic-refresh duty-cycle.

    The reweighter is meta-trained in short bursts and otherwise used frozen on
    cheap forward-pass features alone. Specifically, on steps where
    (t % refresh_period) < meta_burst we recompute the expensive per-example
    gradient-norm target p* and meta-update the network (warm-started from its
    last state, since it is never reset); on all other steps we sample from the
    frozen network with NO gradient-norm pass and NO meta-update.

    This is the amortised regime where the method earns its keep: it keeps the
    importance-sampling benefit while paying the per-example gradient-norm cost on
    only a small fraction of steps, and the periodic refresh lets the proposal
    track the model's drifting notion of which examples are informative -- unlike
    the fixed heuristic, which must recompute its proxy every step.

    debias=True keeps the unbiased importance-weighted update; debias=False uses a
    plain-mean update on the proposal-sampled batch (a biased variant that steers
    training toward high-gradient examples to reduce their error faster).
    """
    rng = np.random.default_rng(seed)
    model = CharLM(vocab_size, seed=seed)
    rw = Reweighter(n_features=5, hidden=hidden, seed=seed, temperature=1.0)
    Ntr = Xtr.shape[0]
    curve = []
    cost_curve = []
    examples_consumed = 0
    gradnorm_passes = 0   # count of expensive per-example gradnorm computations
    cur_cost = float("nan")
    for t in range(steps):
        pool = _draw_pool(rng, Ntr, pool_size)
        Xp, yp = Xtr[pool], ytr[pool]
        cache = model.forward(Xp)
        feats = reweighter_features(model, cache, yp)
        meta_step = (t % refresh_period) < meta_burst
        if meta_step:
            gnorm = model.per_example_gradnorm(cache, yp)
            gradnorm_passes += 1
        idx, p = rw.sample(feats, batch, rng)
        Xb, yb = Xp[idx], yp[idx]
        cache_b = model.forward(Xb)
        if debias:
            w = 1.0 / (pool_size * p[idx])
            w = np.clip(w, 0.0, np.quantile(w, 0.99) + 1e-12)
            w = w / w.mean() / batch
            grads = model.backward(cache_b, yb, weights=w)
        else:
            grads = model.backward(cache_b, yb)   # plain mean -> biased
        model.sgd_step(grads, lr)
        if meta_step:
            cur_cost = rw.meta_update(feats, p, idx, gnorm, meta_lr)
        examples_consumed += batch
        if t % eval_every == 0 or t == steps - 1:
            v = evaluate(model, Xv, yv)
            curve.append((examples_consumed, v))
            cost_curve.append((examples_consumed, cur_cost))
            if log:
                log(f"[learned ] step {t:5d} examples {examples_consumed:8d} "
                    f"val {v:.4f}{' [meta]' if meta_step else ' [frozen]'}")
    return model, curve, cost_curve, rw, gradnorm_passes


def _kmeans_assign(Z, G, rng, iters=5):
    """Tiny k-means; returns a cluster id per row of Z."""
    n = Z.shape[0]
    G = min(G, n)
    centers = Z[rng.choice(n, G, replace=False)].copy()
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = ((Z[:, None, :] - centers[None]) ** 2).sum(-1)   # (n, G)
        assign = d.argmin(1)
        for g in range(G):
            sel = assign == g
            if sel.any():
                centers[g] = Z[sel].mean(0)
    return assign


def _grouped_improvement(model, clone, Xp, yp, content, Xref, yref, alpha,
                         ref_before, n_groups, rng):
    """Cluster-level realized improvement (route 2, denoised).

    Per-example one-step improvement on a held-out set is far too noisy to use as a
    label (a single step's effect is dominated by variance). So we cluster the pool
    by *content* (the mean-embedding features, which separate input regions), take
    one gradient step on each cluster, and measure the reduction in held-out
    reference loss it produces:

        impr_g = L_ref(theta) - L_ref(theta - alpha * grad_on_cluster_g).

    Every example in cluster g is labelled with impr_g. A learnable cluster reduces
    the reference loss (it transfers to held-out learnable data); an
    unlearnable-noise cluster does not. This separates learnable from noise -- which
    the gradient norm (large for both) cannot -- and costs only n_groups reference
    evals per burst step.
    """
    n = Xp.shape[0]
    assign = _kmeans_assign(content, n_groups, rng)
    target = np.zeros(n)
    for g in np.unique(assign):
        sel = assign == g
        clone.copy_params_from(model)
        gg = clone.backward(clone.forward(Xp[sel]), yp[sel])   # cluster mean gradient
        clone.sgd_step(gg, alpha)
        target[sel] = ref_before - evaluate(clone, Xref, yref)
    return target


def train_learned_reducible(Xtr, ytr, Xv, yv, vocab_size, *, steps, batch, lr,
                            pool_size, meta_lr, eval_every, Xref, yref,
                            seed=0, log=None, hidden=32, meta_burst=50,
                            refresh_period=500, probe_lr=0.1, n_groups=8):
    """Sampler trained on measured *reducible improvement* instead of gradient norm.

    During short periodic bursts we cluster the pool by content and measure, per
    cluster, how much a gradient step on it improves loss on a held-out reference
    set (_grouped_improvement); the scorer is meta-trained to match a proposal
    proportional to that improvement (clamped at 0, so unlearnable noise -> ~0
    weight). The scorer sees content-aware features (mean context embedding) so it
    can tell learnable regions from noise regions, and between bursts it is frozen
    and used on cheap features alone.

    The model update is the BIASED plain-mean gradient over the proposal-sampled
    batch (no importance weights), per the deliberately-biased variant: we sample
    where improvement is high and train on the ordinary gradient.
    """
    rng = np.random.default_rng(seed)
    model = CharLM(vocab_size, seed=seed)
    clone = CharLM(vocab_size, context=model.C, emb_dim=model.D, hidden=model.H)
    rw = Reweighter(n_features=5 + model.D, hidden=hidden, seed=seed, temperature=1.0)
    Ntr = Xtr.shape[0]
    curve, cost_curve = [], []
    examples_consumed = 0
    measure_passes = 0           # number of bursts paying the measurement cost
    for t in range(steps):
        pool = _draw_pool(rng, Ntr, pool_size)
        Xp, yp = Xtr[pool], ytr[pool]
        cache = model.forward(Xp)
        feats = reweighter_features_content(model, cache, yp)
        meta_step = (t % refresh_period) < meta_burst
        if meta_step:
            ref_before = evaluate(model, Xref, yref)
            impr = _grouped_improvement(model, clone, Xp, yp, feats[:, 5:],
                                        Xref, yref, probe_lr, ref_before,
                                        n_groups, rng)
            target = np.maximum(impr, 0.0)
            measure_passes += 1
        idx, p = rw.sample(feats, batch, rng)
        Xb, yb = Xp[idx], yp[idx]
        cache_b = model.forward(Xb)
        grads = model.backward(cache_b, yb)        # biased plain-mean update
        model.sgd_step(grads, lr)
        if meta_step:
            rw.meta_update(feats, p, idx, target, meta_lr)
        examples_consumed += batch
        if t % eval_every == 0 or t == steps - 1:
            v = evaluate(model, Xv, yv)
            curve.append((examples_consumed, v))
            if log:
                log(f"[reducible] step {t:5d} examples {examples_consumed:8d} "
                    f"val {v:.4f}{' [meta]' if meta_step else ' [frozen]'}")
    return model, curve, rw, measure_passes
