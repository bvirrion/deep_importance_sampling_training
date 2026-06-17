"""
Concentrated-difficulty experiment: ~99% trivially-easy / ~1% rare-but-learnable
hard examples. This is the regime where the informative "pain" is a tiny fraction
of the data, so importance sampling -- which reduces the variance of the gradient
estimate on the rare high-gradient examples -- reaches a given hard-tier loss with
substantially fewer training examples than uniform SGD.

We reuse the training conditions (uniform / gradnorm IS / learned) and the model
unchanged. The key reporting twist: because the easy 99% dominates the overall
validation loss (and dilutes any difference ~100x), the decisive metric is the
loss on the HARD ~1% subset. We obtain the hard-subset learning curve for free by
passing the hard-subset val arrays into the existing train_* functions (which only
use Xv,yv for evaluation), and report, for several target losses, HOW MANY MORE
training examples uniform needs than the importance samplers to reach them.

Artifacts are written to *_concentrated filenames so the main-experiment results
are not clobbered.
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
    n_hard_patterns=6,    # small enough to be learnable, large enough to be data-hungry
    steps=9000,           # long enough that uniform also reaches the target losses
    batch=128,
    pool_size=512,
    lr=0.5,
    meta_lr=0.1,
    hidden=32,
    meta_burst=100,
    refresh_period=1000,
    eval_every=100,
    seeds=[0, 1, 2, 3, 4],
    # Target hard-subset losses at which we measure "examples to reach"; the
    # headline metric is how many MORE examples uniform needs than the IS methods.
    taus=[1.1, 1.0, 0.9, 0.8, 0.7],
)


def run_once(seed, cfg, log):
    data, vocab, stoi, char_is_hard = make_corpus_concentrated(
        n_chars=cfg["n_chars"], seed=seed, easy_frac=cfg["easy_frac"],
        n_hard_patterns=cfg["n_hard_patterns"])
    X, y = build_examples(data, context=cfg["context"])
    (Xtr, ytr), (Xv, yv) = split(X, y, frac_train=0.9)
    V = len(vocab)

    # Example-level hard mask, aligned with targets, restricted to the val split.
    mask = hard_example_mask(char_is_hard, context=cfg["context"])
    mask_v = mask[ytr.shape[0]:]
    Xvh, yvh = Xv[mask_v], yv[mask_v]   # hard-subset validation arrays

    common = dict(steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"],
                  eval_every=cfg["eval_every"], seed=seed, log=log)

    # Pass the HARD-subset val arrays so the recorded curve is the hard-subset loss.
    m_uni, c_uni = train_uniform(Xtr, ytr, Xvh, yvh, V, **common)
    m_gn, c_gn, gn_passes = train_gradnorm(
        Xtr, ytr, Xvh, yvh, V, pool_size=cfg["pool_size"], **common)
    m_lr, c_lr, _cost, _rw, lr_passes = train_learned(
        Xtr, ytr, Xvh, yvh, V, pool_size=cfg["pool_size"], meta_lr=cfg["meta_lr"],
        hidden=cfg["hidden"], meta_burst=cfg["meta_burst"],
        refresh_period=cfg["refresh_period"], **common)

    # Final OVERALL loss (full val set) of each trained model, for context.
    overall = dict(uniform=float(evaluate(m_uni, Xv, yv)),
                   gradnorm=float(evaluate(m_gn, Xv, yv)),
                   learned=float(evaluate(m_lr, Xv, yv)))
    return dict(uniform=c_uni, gradnorm=c_gn, learned=c_lr,
                overall=overall, gn_passes=gn_passes, lr_passes=lr_passes)


def aggregate(all_runs, key):
    curves = [np.array([v for _, v in r[key]]) for r in all_runs]
    xs = np.array([x for x, _ in all_runs[0][key]])
    return xs, np.stack(curves, axis=0)   # xs, (seeds, points)


def tstat(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    if d.std(ddof=1) == 0:
        return float("inf") if d.mean() > 0 else 0.0
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))


def examples_to_target(xs, row, tau):
    """Examples needed for a single curve to first reach loss <= tau.

    Linearly interpolates between the two eval points that bracket the crossing,
    so the estimate is not quantised to eval_every. Returns nan if never reached.
    """
    hit = np.where(row <= tau)[0]
    if len(hit) == 0:
        return np.nan
    j = hit[0]
    if j == 0:
        return float(xs[0])
    x0, x1 = xs[j - 1], xs[j]
    y0, y1 = row[j - 1], row[j]
    if y0 == y1:
        return float(x1)
    return float(x0 + (tau - y0) * (x1 - x0) / (y1 - y0))


def main():
    cfg = CONFIG
    logs = []
    def log(msg):
        logs.append(msg)
        print(msg, flush=True)

    all_runs = []
    for s in cfg["seeds"]:
        log(f"==== seed {s} ====")
        all_runs.append(run_once(s, cfg, log))

    summary = {}
    # Hard-subset learning curves + final hard-subset loss.
    for key in ["uniform", "gradnorm", "learned"]:
        xs, M = aggregate(all_runs, key)
        final = M[:, -1]
        summary[key] = dict(
            hard_final_mean=float(final.mean()),
            hard_final_std=float(final.std()),
            overall_final_mean=float(np.mean([r["overall"][key] for r in all_runs])),
            overall_final_std=float(np.std([r["overall"][key] for r in all_runs])),
            curve_x=xs.tolist(),
            curve_mean=M.mean(axis=0).tolist(),
            curve_std=M.std(axis=0).tolist(),
        )

    # HEADLINE: sample efficiency on the HARD subset -- how many examples each
    # method needs to reach a given hard-subset loss, and how many MORE times
    # uniform needs than the importance samplers (the speed multiplier).
    curves = {k: aggregate(all_runs, k) for k in ["uniform", "gradnorm", "learned"]}
    eff_by_tau = []
    for tau in cfg["taus"]:
        rec = dict(tau=float(tau))
        per = {}
        for key in ["uniform", "gradnorm", "learned"]:
            xs, M = curves[key]
            ex = np.array([examples_to_target(xs, row, tau) for row in M])
            per[key] = ex
            rec[key] = dict(mean_examples=float(np.nanmean(ex)),
                            frac_reached=float(np.mean(~np.isnan(ex))))
        # multiplier = uniform examples / IS examples, averaged per-seed where both reached
        for is_key in ["gradnorm", "learned"]:
            ratio = per["uniform"] / per[is_key]
            rec[f"uniform_over_{is_key}"] = float(np.nanmean(ratio))
        eff_by_tau.append(rec)
    summary["efficiency_hard"] = eff_by_tau

    # Paired t-stats on hard-subset final loss (positive = first arg better).
    _, Mu = aggregate(all_runs, "uniform")
    _, Mg = aggregate(all_runs, "gradnorm")
    _, Ml = aggregate(all_runs, "learned")
    summary["paired_hard"] = dict(
        gradnorm_vs_uniform_mean=float((Mu[:, -1] - Mg[:, -1]).mean()),
        gradnorm_vs_uniform_t=tstat(Mu[:, -1] - Mg[:, -1]),
        learned_vs_uniform_mean=float((Mu[:, -1] - Ml[:, -1]).mean()),
        learned_vs_uniform_t=tstat(Mu[:, -1] - Ml[:, -1]),
        learned_vs_gradnorm_mean=float((Mg[:, -1] - Ml[:, -1]).mean()),
        learned_vs_gradnorm_t=tstat(Mg[:, -1] - Ml[:, -1]),
        n_seeds=len(cfg["seeds"]),
    )
    gn_passes = float(np.mean([r["gn_passes"] for r in all_runs]))
    lr_passes = float(np.mean([r["lr_passes"] for r in all_runs]))
    summary["overhead"] = dict(
        gradnorm_passes_heuristic=gn_passes,
        gradnorm_passes_learned=lr_passes,
        saved_fraction=float(1.0 - lr_passes / gn_passes) if gn_passes else 0.0,
    )
    summary["config"] = cfg

    with open(os.path.join(REPO_ROOT, "results_concentrated.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Figure: hard-subset validation loss vs compute budget.
    plt.figure(figsize=(7, 4.5))
    colors = dict(uniform="#888888", gradnorm="#1f77b4", learned="#d62728")
    labels = dict(uniform="Uniform SGD",
                  gradnorm="Fixed grad-norm IS",
                  learned="Learned reweighter (ours)")
    for key in ["uniform", "gradnorm", "learned"]:
        xs = np.array(summary[key]["curve_x"])
        mu = np.array(summary[key]["curve_mean"])
        sd = np.array(summary[key]["curve_std"])
        plt.plot(xs, mu, color=colors[key], label=labels[key], linewidth=2)
        plt.fill_between(xs, mu - sd, mu + sd, color=colors[key], alpha=0.15)
    plt.xlabel("training examples consumed")
    plt.ylabel("hard-subset validation loss (nats/char)")
    plt.title("Concentrated difficulty (1% hard): loss on the hard subset")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "curves_concentrated.pdf"))
    plt.savefig(os.path.join(FIGS_DIR, "curves_concentrated.png"), dpi=130)
    plt.close()

    with open(os.path.join(REPO_ROOT, "train_log_concentrated.txt"), "w") as f:
        f.write("\n".join(logs))

    print("\n==== SUMMARY (concentrated, 5 seeds) ====")
    for key in ["uniform", "gradnorm", "learned"]:
        s = summary[key]
        print(f"{key:9s} hard-final {s['hard_final_mean']:.4f} +/- {s['hard_final_std']:.4f}"
              f"   overall-final {s['overall_final_mean']:.4f} +/- {s['overall_final_std']:.4f}")
    print("\nexamples to reach a hard-subset loss target (mean over seeds), and "
          "how many MORE times uniform needs than the IS methods:")
    print(f"  {'tau':>5} | {'uniform':>9} {'gradnorm':>9} {'learned':>9} | "
          f"{'uni/gn':>7} {'uni/lr':>7}")
    for rec in summary["efficiency_hard"]:
        print(f"  {rec['tau']:>5.2f} | {rec['uniform']['mean_examples']:>9.0f} "
              f"{rec['gradnorm']['mean_examples']:>9.0f} {rec['learned']['mean_examples']:>9.0f} | "
              f"{rec['uniform_over_gradnorm']:>6.2f}x {rec['uniform_over_learned']:>6.2f}x")
    p = summary["paired_hard"]
    print("\npaired (hard final) gradnorm vs uniform: d={:.4f} t={:.2f}".format(
        p["gradnorm_vs_uniform_mean"], p["gradnorm_vs_uniform_t"]))
    print("paired (hard final) learned  vs uniform: d={:.4f} t={:.2f}".format(
        p["learned_vs_uniform_mean"], p["learned_vs_uniform_t"]))
    print("paired (hard final) learned  vs gradnorm: d={:.4f} t={:.2f}".format(
        p["learned_vs_gradnorm_mean"], p["learned_vs_gradnorm_t"]))


if __name__ == "__main__":
    main()
