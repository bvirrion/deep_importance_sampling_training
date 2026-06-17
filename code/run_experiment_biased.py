"""
Biased-sampling experiment (concentrated-difficulty regime).

We keep the change of measure for *sampling* (draw the minibatch from a pool with
probability proportional to the gradient-norm proxy, or from the learned proposal)
but drop the importance weights and train on the PLAIN-MEAN gradient. The update is
therefore biased: it deliberately steers the optimisation toward high-gradient
(here, rare hard) examples -- soft hard-example mining -- to reduce their error
faster, at the cost of unbiasedness.

Two phases:
  1. Main comparison (5 seeds) on the concentrated corpus: uniform, the unbiased
     importance samplers (grad-norm / learned), and the biased samplers
     (grad-norm / learned). Headline metric: how many more examples uniform needs
     to reach a given HARD-subset loss than the biased samplers.
  2. Learning-rate stability sweep: the biased update inflates the effective
     gradient magnitude, so we sweep lr and report final hard-subset loss and how
     often each method diverges.

Artifacts: results_biased.json, figs/curves_biased.{pdf,png}, train_log_biased.txt.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from code.data import (make_corpus_concentrated, build_examples, split,
                       hard_example_mask)
from code.train import train_uniform, train_gradnorm, train_learned, evaluate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS_DIR = os.path.join(REPO_ROOT, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

CONFIG = dict(
    n_chars=120_000,
    context=8,
    easy_frac=0.99,
    n_hard_patterns=6,
    steps=6000,
    batch=128,
    pool_size=512,
    lr=0.5,
    meta_lr=0.1,
    hidden=32,
    meta_burst=100,
    refresh_period=1000,
    eval_every=100,
    seeds=[0, 1, 2, 3, 4],
    taus=[1.1, 1.0, 0.9, 0.8, 0.7],
    sweep_lrs=[0.5, 1.0, 2.0, 3.0],
    sweep_seeds=[0, 1, 2],
    sweep_steps=4000,
)

# The conditions we compare. (label -> (trainer, debias))
CONDS = ["uniform", "gradnorm_is", "learned_is", "gradnorm_biased", "learned_biased"]
PRETTY = dict(uniform="Uniform SGD",
              gradnorm_is="Grad-norm IS (unbiased)",
              learned_is="Learned IS (unbiased)",
              gradnorm_biased="Grad-norm sampling, plain update (biased)",
              learned_biased="Learned sampling, plain update (biased)")


def _concentrated_split(seed, cfg):
    data, vocab, stoi, cih = make_corpus_concentrated(
        n_chars=cfg["n_chars"], seed=seed, easy_frac=cfg["easy_frac"],
        n_hard_patterns=cfg["n_hard_patterns"])
    X, y = build_examples(data, context=cfg["context"])
    (Xtr, ytr), (Xv, yv) = split(X, y, frac_train=0.9)
    mask_v = hard_example_mask(cih, context=cfg["context"])[ytr.shape[0]:]
    return (Xtr, ytr), (Xv, yv), (Xv[mask_v], yv[mask_v]), len(vocab)


def _train(cond, Xtr, ytr, Xvh, yvh, V, *, steps, batch, lr, pool_size,
           meta_lr, eval_every, seed, hidden, meta_burst, refresh_period):
    """Train one condition; return (model, curve-on-hard-subset)."""
    if cond == "uniform":
        m, c = train_uniform(Xtr, ytr, Xvh, yvh, V, steps=steps, batch=batch,
                             lr=lr, eval_every=eval_every, seed=seed)
        return m, c
    if cond.startswith("gradnorm"):
        m, c, _ = train_gradnorm(
            Xtr, ytr, Xvh, yvh, V, steps=steps, batch=batch, lr=lr,
            pool_size=pool_size, eval_every=eval_every, seed=seed,
            debias=cond.endswith("_is"))
        return m, c
    # learned_*
    m, c, _cc, _rw, _ = train_learned(
        Xtr, ytr, Xvh, yvh, V, steps=steps, batch=batch, lr=lr,
        pool_size=pool_size, meta_lr=meta_lr, eval_every=eval_every, seed=seed,
        hidden=hidden, meta_burst=meta_burst, refresh_period=refresh_period,
        debias=cond.endswith("_is"))
    return m, c


def examples_to_target(xs, row, tau):
    hit = np.where(row <= tau)[0]
    if len(hit) == 0:
        return np.nan
    j = hit[0]
    if j == 0:
        return float(xs[0])
    x0, x1, y0, y1 = xs[j - 1], xs[j], row[j - 1], row[j]
    return float(x1) if y0 == y1 else float(x0 + (tau - y0) * (x1 - x0) / (y1 - y0))


def tstat(d):
    d = np.asarray(d, float); n = len(d)
    s = d.std(ddof=1)
    return (float("inf") if d.mean() > 0 else 0.0) if s == 0 else float(d.mean() / (s / np.sqrt(n)))


def main():
    cfg = CONFIG
    logs = []
    def log(m):
        logs.append(m); print(m, flush=True)

    # ---- Phase 1: main comparison on the concentrated task -----------------
    runs = {c: [] for c in CONDS}          # per cond: list of (xs, curve) per seed
    overall = {c: [] for c in CONDS}
    for s in cfg["seeds"]:
        log(f"==== seed {s} ====")
        (Xtr, ytr), (Xv, yv), (Xvh, yvh), V = _concentrated_split(s, cfg)
        for cond in CONDS:
            m, c = _train(cond, Xtr, ytr, Xvh, yvh, V,
                          steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"],
                          pool_size=cfg["pool_size"], meta_lr=cfg["meta_lr"],
                          eval_every=cfg["eval_every"], seed=s, hidden=cfg["hidden"],
                          meta_burst=cfg["meta_burst"], refresh_period=cfg["refresh_period"])
            xs = np.array([x for x, _ in c]); ys = np.array([v for _, v in c])
            runs[cond].append((xs, ys))
            overall[cond].append(float(evaluate(m, Xv, yv)))
            log(f"  {cond:16s} hard-final {ys[-1]:.4f}  overall {overall[cond][-1]:.4f}")

    summary = {}
    for cond in CONDS:
        M = np.stack([ys for _, ys in runs[cond]], axis=0)
        xs = runs[cond][0][0]
        summary[cond] = dict(
            hard_final_mean=float(M[:, -1].mean()), hard_final_std=float(M[:, -1].std()),
            overall_final_mean=float(np.mean(overall[cond])),
            curve_x=xs.tolist(), curve_mean=M.mean(0).tolist(), curve_std=M.std(0).tolist())

    # efficiency: examples to reach each tau, multiplier vs uniform
    eff = []
    perseed = {c: {} for c in CONDS}
    for tau in cfg["taus"]:
        rec = dict(tau=float(tau))
        for cond in CONDS:
            ex = np.array([examples_to_target(xs, ys, tau) for xs, ys in runs[cond]])
            perseed[cond][tau] = ex
            rec[cond] = dict(mean_examples=float(np.nanmean(ex)),
                             frac_reached=float(np.mean(~np.isnan(ex))))
        for cond in CONDS:
            if cond != "uniform":
                rec[f"uniform_over_{cond}"] = float(np.nanmean(perseed["uniform"][tau] / perseed[cond][tau]))
        eff.append(rec)
    summary["efficiency_hard"] = eff

    # paired t-stats on hard-subset final loss vs uniform (positive = cond better)
    Mu = np.stack([ys for _, ys in runs["uniform"]], 0)[:, -1]
    summary["paired_vs_uniform"] = {}
    for cond in CONDS:
        if cond == "uniform":
            continue
        Mc = np.stack([ys for _, ys in runs[cond]], 0)[:, -1]
        summary["paired_vs_uniform"][cond] = dict(mean=float((Mu - Mc).mean()), t=tstat(Mu - Mc))

    # ---- Phase 2: learning-rate stability sweep ----------------------------
    log("\n==== lr stability sweep ====")
    sweep = []
    sweep_conds = ["uniform", "gradnorm_biased", "learned_biased"]
    for lr in cfg["sweep_lrs"]:
        rec = dict(lr=float(lr))
        for cond in sweep_conds:
            finals, ndiv = [], 0
            for s in cfg["sweep_seeds"]:
                (Xtr, ytr), (Xv, yv), (Xvh, yvh), V = _concentrated_split(s, cfg)
                _, c = _train(cond, Xtr, ytr, Xvh, yvh, V,
                              steps=cfg["sweep_steps"], batch=cfg["batch"], lr=lr,
                              pool_size=cfg["pool_size"], meta_lr=cfg["meta_lr"],
                              eval_every=cfg["sweep_steps"] // 4, seed=s,
                              hidden=cfg["hidden"], meta_burst=cfg["meta_burst"],
                              refresh_period=cfg["refresh_period"])
                fv = c[-1][1]
                diverged = (not np.isfinite(fv)) or fv > 5.0
                if diverged:
                    ndiv += 1
                else:
                    finals.append(fv)
            rec[cond] = dict(hard_final_mean=(float(np.mean(finals)) if finals else float("nan")),
                             diverged=ndiv, n=len(cfg["sweep_seeds"]))
        sweep.append(rec)
        log(f"  lr={lr}: " + " | ".join(
            f"{c} {rec[c]['hard_final_mean']:.3f} (div {rec[c]['diverged']}/{rec[c]['n']})"
            for c in sweep_conds))
    summary["lr_sweep"] = sweep
    summary["config"] = cfg

    with open(os.path.join(REPO_ROOT, "results_biased.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Figure: hard-subset curves ---------------------------------------
    plt.figure(figsize=(7.5, 4.8))
    colors = dict(uniform="#888888", gradnorm_is="#1f77b4", learned_is="#2ca02c",
                  gradnorm_biased="#d62728", learned_biased="#ff7f0e")
    styles = dict(uniform="-", gradnorm_is="--", learned_is="--",
                  gradnorm_biased="-", learned_biased="-")
    for cond in CONDS:
        xs = np.array(summary[cond]["curve_x"]); mu = np.array(summary[cond]["curve_mean"])
        sd = np.array(summary[cond]["curve_std"])
        plt.plot(xs, mu, styles[cond], color=colors[cond], label=PRETTY[cond], linewidth=2)
        plt.fill_between(xs, mu - sd, mu + sd, color=colors[cond], alpha=0.12)
    plt.xlabel("training examples consumed")
    plt.ylabel("hard-subset validation loss (nats/char)")
    plt.title("Biased sampling vs unbiased IS vs uniform (concentrated 1% hard)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "curves_biased.pdf"))
    plt.savefig(os.path.join(FIGS_DIR, "curves_biased.png"), dpi=130)
    plt.close()

    with open(os.path.join(REPO_ROOT, "train_log_biased.txt"), "w") as f:
        f.write("\n".join(logs))

    # ---- printed summary ---------------------------------------------------
    print("\n==== SUMMARY (biased sampling, concentrated, 5 seeds) ====")
    for cond in CONDS:
        s = summary[cond]
        print(f"{cond:16s} hard-final {s['hard_final_mean']:.4f} +/- {s['hard_final_std']:.4f}"
              f"   overall {s['overall_final_mean']:.4f}")
    print("\nexamples to reach hard-subset target (mean), multiplier vs uniform:")
    hdr = f"  {'tau':>5} | " + " ".join(f"{c[:9]:>9}" for c in CONDS)
    print(hdr)
    for rec in summary["efficiency_hard"]:
        row = f"  {rec['tau']:>5.2f} | " + " ".join(f"{rec[c]['mean_examples']:>9.0f}" for c in CONDS)
        mult = "  ".join(f"uni/{c.split('_')[0][:2]}{'b' if 'biased' in c else 'i'}={rec[f'uniform_over_{c}']:.2f}x"
                         for c in CONDS if c != "uniform")
        print(row + "   " + mult)
    print("\npaired vs uniform (hard final, positive = better):")
    for cond, v in summary["paired_vs_uniform"].items():
        print(f"  {cond:16s} d={v['mean']:+.4f}  t={v['t']:.2f}")
    print("\nlr stability sweep (hard-final, divergences):")
    for rec in summary["lr_sweep"]:
        print(f"  lr={rec['lr']}: " + " | ".join(
            f"{c}={rec[c]['hard_final_mean']:.3f}(div{rec[c]['diverged']}/{rec[c]['n']})"
            for c in ["uniform", "gradnorm_biased", "learned_biased"]))


if __name__ == "__main__":
    main()
