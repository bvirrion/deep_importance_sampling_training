"""
Run the full A/B/C experiment across multiple seeds, save curves, summary
statistics, and figures.

Fairness: every method is charged the same number of gradient examples
(steps * batch). The pool-based methods additionally do forward passes on the
candidate pool; we report that overhead explicitly so the comparison is honest.
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import make_corpus, build_examples, split
from train import train_uniform, train_gradnorm, train_learned, evaluate

CONFIG = dict(
    n_chars=120_000,
    context=8,
    steps=600,
    batch=128,
    pool_size=512,
    lr=0.5,
    meta_lr=0.5,
    amortize_after=150,
    eval_every=20,
    seeds=[0, 1, 2, 3, 4],
)


def run_once(seed, cfg, log):
    data, vocab, stoi = make_corpus(n_chars=cfg["n_chars"], seed=seed)
    X, y = build_examples(data, context=cfg["context"])
    (Xtr, ytr), (Xv, yv) = split(X, y, frac_train=0.9)
    V = len(vocab)

    _, c_uni = train_uniform(
        Xtr, ytr, Xv, yv, V,
        steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"],
        eval_every=cfg["eval_every"], seed=seed, log=log)

    _, c_gn, gn_passes = train_gradnorm(
        Xtr, ytr, Xv, yv, V,
        steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"],
        pool_size=cfg["pool_size"], eval_every=cfg["eval_every"], seed=seed, log=log)

    _, c_lr, cost_curve, rw, lr_passes = train_learned(
        Xtr, ytr, Xv, yv, V,
        steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"],
        pool_size=cfg["pool_size"], meta_lr=cfg["meta_lr"],
        eval_every=cfg["eval_every"], seed=seed, log=log,
        amortize_after=cfg["amortize_after"])

    return dict(uniform=c_uni, gradnorm=c_gn, learned=c_lr,
                cost=cost_curve, phi=rw.phi.tolist(),
                gn_passes=gn_passes, lr_passes=lr_passes)


def aggregate(all_runs, key):
    """Stack final-value and full-curve arrays across seeds for a condition."""
    curves = [np.array([v for _, v in r[key]]) for r in all_runs]
    xs = np.array([x for x, _ in all_runs[0][key]])
    M = np.stack(curves, axis=0)   # (seeds, points)
    return xs, M


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

    # Aggregate final validation losses
    summary = {}
    for key in ["uniform", "gradnorm", "learned"]:
        xs, M = aggregate(all_runs, key)
        final = M[:, -1]
        summary[key] = dict(
            final_mean=float(final.mean()),
            final_std=float(final.std()),
            curve_x=xs.tolist(),
            curve_mean=M.mean(axis=0).tolist(),
            curve_std=M.std(axis=0).tolist(),
        )

    # Budget to reach a target val loss (efficiency): examples to first hit tau
    def examples_to_target(xs, M, tau):
        out = []
        for row in M:
            hit = np.where(row <= tau)[0]
            out.append(xs[hit[0]] if len(hit) else np.nan)
        return np.array(out)

    xs_u, M_u = aggregate(all_runs, "uniform")
    # target = best loss uniform reliably reaches (its mean final + small margin)
    tau = summary["uniform"]["final_mean"] + 0.02
    eff = {}
    for key in ["uniform", "gradnorm", "learned"]:
        xs, M = aggregate(all_runs, key)
        e = examples_to_target(xs, M, tau)
        eff[key] = dict(tau=float(tau),
                        mean_examples=float(np.nanmean(e)),
                        frac_reached=float(np.mean(~np.isnan(e))))
    summary["efficiency"] = eff

    # Paired improvement learned vs uniform and vs gradnorm (final loss)
    _, Mu = aggregate(all_runs, "uniform")
    _, Mg = aggregate(all_runs, "gradnorm")
    _, Ml = aggregate(all_runs, "learned")
    d_lu = Mu[:, -1] - Ml[:, -1]   # positive = learned better
    d_lg = Mg[:, -1] - Ml[:, -1]
    def tstat(d):
        d = np.asarray(d, dtype=float)
        n = len(d)
        if d.std(ddof=1) == 0:
            return float("inf") if d.mean() > 0 else 0.0
        return float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    summary["paired"] = dict(
        learned_vs_uniform_mean=float(d_lu.mean()),
        learned_vs_uniform_t=tstat(d_lu),
        learned_vs_gradnorm_mean=float(d_lg.mean()),
        learned_vs_gradnorm_t=tstat(d_lg),
        n_seeds=len(cfg["seeds"]),
    )
    summary["mean_phi"] = np.mean([r["phi"] for r in all_runs], axis=0).tolist()
    # overhead: how many expensive per-example gradient-norm passes each method needs
    gn_passes = float(np.mean([r["gn_passes"] for r in all_runs]))
    lr_passes = float(np.mean([r["lr_passes"] for r in all_runs]))
    summary["overhead"] = dict(
        gradnorm_passes_heuristic=gn_passes,
        gradnorm_passes_learned=lr_passes,
        saved_fraction=float(1.0 - lr_passes / gn_passes) if gn_passes else 0.0,
    )
    summary["config"] = cfg

    with open("/home/claude/lrw/results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Figure 1: learning curves (val loss vs examples) ------------------
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
    plt.ylabel("validation loss (nats/char)")
    plt.title("Validation loss vs compute budget (mean $\\pm$ s.d., 5 seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/claude/lrw/figs/curves.pdf")
    plt.savefig("/home/claude/lrw/figs/curves.png", dpi=130)
    plt.close()

    # ---- Figure 2: variance proxy (reweighter cost) over training ----------
    plt.figure(figsize=(7, 4.0))
    for r in all_runs:
        cc = np.array([c for _, c in r["cost"]])
        xs = np.array([x for x, _ in r["cost"]])
        plt.plot(xs, cc, color="#d62728", alpha=0.35)
    plt.xlabel("training examples consumed")
    plt.ylabel("estimator-variance proxy  $E_p[C]$")
    plt.title("Reweighter meta-objective (per-seed) decreasing over training")
    plt.grid(alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("/home/claude/lrw/figs/variance.pdf")
    plt.savefig("/home/claude/lrw/figs/variance.png", dpi=130)
    plt.close()

    # ---- Figure 3: learned phi weights -------------------------------------
    feat_names = ["loss", "grad-norm", "entropy", "max-prob", "bias"]
    phi = np.array(summary["mean_phi"])
    plt.figure(figsize=(6, 3.8))
    plt.bar(feat_names, phi, color="#d62728", alpha=0.8)
    plt.ylabel("mean learned weight $\\phi$")
    plt.title("What the reweighter learned to weight (mean over seeds)")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("/home/claude/lrw/figs/phi.pdf")
    plt.savefig("/home/claude/lrw/figs/phi.png", dpi=130)
    plt.close()

    with open("/home/claude/lrw/train_log.txt", "w") as f:
        f.write("\n".join(logs))

    # Print compact summary
    print("\n==== SUMMARY ====")
    for key in ["uniform", "gradnorm", "learned"]:
        print(f"{key:9s} final val {summary[key]['final_mean']:.4f} "
              f"+/- {summary[key]['final_std']:.4f}")
    print("efficiency (examples to reach tau = {:.3f}):".format(tau))
    for key in ["uniform", "gradnorm", "learned"]:
        e = summary["efficiency"][key]
        print(f"  {key:9s} {e['mean_examples']:.0f} examples "
              f"(reached in {100*e['frac_reached']:.0f}% of seeds)")
    print("paired learned vs uniform : d={:.4f} t={:.2f}".format(
        summary["paired"]["learned_vs_uniform_mean"],
        summary["paired"]["learned_vs_uniform_t"]))
    print("paired learned vs gradnorm: d={:.4f} t={:.2f}".format(
        summary["paired"]["learned_vs_gradnorm_mean"],
        summary["paired"]["learned_vs_gradnorm_t"]))
    print("mean phi:", np.array2string(np.array(summary["mean_phi"]), precision=3))
    ov = summary["overhead"]
    print("overhead: heuristic needs {:.0f} gradnorm passes, learned needs {:.0f} "
          "({:.0f}% saved via amortization)".format(
              ov["gradnorm_passes_heuristic"], ov["gradnorm_passes_learned"],
              100 * ov["saved_fraction"]))


if __name__ == "__main__":
    main()
