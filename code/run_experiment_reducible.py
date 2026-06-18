"""
Reducible-improvement experiment.

Regime (make_corpus_reducible): ~90% trivially-easy text, plus two HIGH-GRADIENT
tiers in disjoint content regions -- a learnable-hard tier (fixed codebook, set A
letters) and an unlearnable-noise tier (random, set B letters). Gradient norm is
large for BOTH hard tiers, so a gradient-norm sampler cannot tell them apart and
wastes budget on noise. The reducible-improvement scorer is meta-trained on the
measured reduction in held-out (learnable) loss produced by a step on each content
cluster, so it learns to concentrate on the learnable tier and ignore the noise.

All conditions use the BIASED plain-mean update (sample with the change of measure,
train on the ordinary gradient). We report (a) the sampling weight each method puts
on each tier -- the headline: reducible << gradnorm on noise -- and (b) the
learnable-subset loss vs budget.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from code.data import (make_corpus_reducible, build_examples, split,
                       tier_example_mask, reweighter_features_content)
from code.model import CharLM
from code.train import (train_uniform, train_gradnorm, train_learned_reducible,
                        evaluate)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS_DIR = os.path.join(REPO_ROOT, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

CONFIG = dict(
    n_chars=120_000, context=8, steps=4000, batch=128, pool_size=256,
    lr=0.5, meta_lr=0.2, hidden=32, eval_every=100,
    meta_burst=50, refresh_period=500, probe_lr=0.1, n_groups=8,
    ref_size=256, seeds=[0, 1, 2, 3, 4], taus=[1.0, 0.8, 0.6],
)
TIER = {0: "easy", 1: "learnable", 2: "noise"}


def _splits(seed, cfg):
    data, vocab, stoi, tier = make_corpus_reducible(n_chars=cfg["n_chars"], seed=seed)
    X, y = build_examples(data, context=cfg["context"])
    (Xtr, ytr), (Xv, yv) = split(X, y, frac_train=0.9)
    m = tier_example_mask(tier, context=cfg["context"])
    mtr, mv = m[:ytr.shape[0]], m[ytr.shape[0]:]
    # held-out learnable: first half = measurement reference, second half = eval curve
    li = np.where(mv == 1)[0]
    half = len(li) // 2
    ref = (Xv[li[:half]], yv[li[:half]])
    learn_eval = (Xv[li[half:]], yv[li[half:]])
    noise_eval = (Xv[mv == 2], yv[mv == 2])
    return (Xtr, ytr), (Xv, yv), mtr, ref, learn_eval, noise_eval, len(vocab)


def tier_weight_share(model, rw, Xtr, ytr, mtr, cfg, seed):
    """Mean sampling weight (x uniform) each method puts on each tier, on a balanced pool."""
    rng = np.random.default_rng(900 + seed)
    idx = np.concatenate([rng.choice(np.where(mtr == t)[0], size=128, replace=False)
                          for t in [0, 1, 2]])
    pt = mtr[idx]
    cache = model.forward(Xtr[idx])
    out = {}
    # gradnorm proposal
    gn = model.per_example_gradnorm(cache, ytr[idx]); pgn = gn / gn.sum()
    out["gradnorm"] = {TIER[t]: float(pgn[pt == t].mean() * len(pgn)) for t in [0, 1, 2]}
    # reducible scorer proposal (if provided)
    if rw is not None:
        feats = reweighter_features_content(model, cache, ytr[idx])
        pr = rw.proposal(feats)
        out["reducible"] = {TIER[t]: float(pr[pt == t].mean() * len(pr)) for t in [0, 1, 2]}
    return out


def ex_to_target(curve, tau):
    xs = [x for x, _ in curve]; ys = [v for _, v in curve]
    for j in range(len(ys)):
        if ys[j] <= tau:
            if j == 0:
                return float(xs[0])
            x0, x1, y0, y1 = xs[j-1], xs[j], ys[j-1], ys[j]
            return float(x1) if y0 == y1 else float(x0 + (tau-y0)*(x1-x0)/(y1-y0))
    return np.nan


def main():
    cfg = CONFIG
    logs = []
    def log(m): logs.append(m); print(m, flush=True)

    conds = ["uniform", "gradnorm_biased", "reducible"]
    curves = {c: [] for c in conds}          # learnable-subset loss curve per seed
    noise_final = {c: [] for c in conds}
    overall_final = {c: [] for c in conds}
    shares = {"gradnorm": [], "reducible": []}

    for s in cfg["seeds"]:
        log(f"==== seed {s} ====")
        (Xtr, ytr), (Xv, yv), mtr, ref, leval, neval, V = _splits(s, cfg)
        Xle, yle = leval; Xne, yne = neval; Xref, yref = ref
        com = dict(steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"],
                   eval_every=cfg["eval_every"], seed=s)
        # curve is evaluated on the held-out LEARNABLE subset (the capability we care about)
        m_u, c_u = train_uniform(Xtr, ytr, Xle, yle, V, **com)
        m_g, c_g, _ = train_gradnorm(Xtr, ytr, Xle, yle, V, pool_size=cfg["pool_size"],
                                     debias=False, **com)
        m_r, c_r, rw, mp = train_learned_reducible(
            Xtr, ytr, Xle, yle, V, pool_size=cfg["pool_size"], meta_lr=cfg["meta_lr"],
            Xref=Xref, yref=yref, hidden=cfg["hidden"], meta_burst=cfg["meta_burst"],
            refresh_period=cfg["refresh_period"], probe_lr=cfg["probe_lr"],
            n_groups=cfg["n_groups"], **com)
        for c, cv, mdl in [("uniform", c_u, m_u), ("gradnorm_biased", c_g, m_g),
                           ("reducible", c_r, m_r)]:
            curves[c].append(cv)
            noise_final[c].append(float(evaluate(mdl, Xne, yne)))
            overall_final[c].append(float(evaluate(mdl, Xv, yv)))
        sh = tier_weight_share(m_r, rw, Xtr, ytr, mtr, cfg, s)
        shares["gradnorm"].append(sh["gradnorm"]); shares["reducible"].append(sh["reducible"])
        log(f"  learnable-final: uniform {c_u[-1][1]:.3f}  gradnorm_b {c_g[-1][1]:.3f}  reducible {c_r[-1][1]:.3f}")

    summary = {}
    for c in conds:
        M = np.stack([np.array([v for _, v in cv]) for cv in curves[c]], 0)
        xs = np.array([x for x, _ in curves[c][0]])
        summary[c] = dict(
            learn_final_mean=float(M[:, -1].mean()), learn_final_std=float(M[:, -1].std()),
            noise_final_mean=float(np.mean(noise_final[c])),
            overall_final_mean=float(np.mean(overall_final[c])),
            curve_x=xs.tolist(), curve_mean=M.mean(0).tolist(), curve_std=M.std(0).tolist())

    # tier weight shares (mean over seeds)
    summary["weight_share"] = {
        meth: {t: float(np.mean([sh[t] for sh in shares[meth]])) for t in ["easy", "learnable", "noise"]}
        for meth in ["gradnorm", "reducible"]}

    # efficiency on the learnable subset: examples to reach target, multiplier vs uniform
    eff = []
    for tau in cfg["taus"]:
        rec = {"tau": tau}
        ex = {c: np.array([ex_to_target(cv, tau) for cv in curves[c]]) for c in conds}
        for c in conds:
            rec[c] = float(np.nanmean(ex[c]))
        for c in ["gradnorm_biased", "reducible"]:
            rec[f"uniform_over_{c}"] = float(np.nanmean(ex["uniform"] / ex[c]))
        eff.append(rec)
    summary["efficiency_learnable"] = eff
    summary["config"] = cfg

    with open(os.path.join(REPO_ROOT, "results_reducible.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # figure: learnable-subset loss vs budget
    plt.figure(figsize=(7.2, 4.6))
    colors = dict(uniform="#888888", gradnorm_biased="#1f77b4", reducible="#d62728")
    labels = dict(uniform="Uniform SGD", gradnorm_biased="Grad-norm biased (chases noise)",
                  reducible="Reducible-improvement (ours)")
    for c in conds:
        xs = np.array(summary[c]["curve_x"]); mu = np.array(summary[c]["curve_mean"])
        sd = np.array(summary[c]["curve_std"])
        plt.plot(xs, mu, color=colors[c], label=labels[c], linewidth=2)
        plt.fill_between(xs, mu - sd, mu + sd, color=colors[c], alpha=0.13)
    plt.xlabel("training examples consumed")
    plt.ylabel("learnable-subset validation loss (nats/char)")
    plt.title("Reducible-improvement sampling vs gradnorm-biased vs uniform")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "curves_reducible.pdf"))
    plt.savefig(os.path.join(FIGS_DIR, "curves_reducible.png"), dpi=130)
    plt.close()

    # figure: tier weight shares (grad-norm chases noise; reducible avoids it)
    tiers = ["easy", "learnable", "noise"]
    gn = [summary["weight_share"]["gradnorm"][t] for t in tiers]
    rd = [summary["weight_share"]["reducible"][t] for t in tiers]
    x = np.arange(len(tiers)); w = 0.38
    plt.figure(figsize=(6.2, 4.0))
    plt.bar(x - w/2, gn, w, label="Grad-norm biased", color="#1f77b4")
    plt.bar(x + w/2, rd, w, label="Reducible (ours)", color="#d62728")
    plt.axhline(1.0, color="#888888", ls="--", lw=1, label="uniform")
    plt.xticks(x, tiers); plt.ylabel(r"mean sampling weight ($\times$ uniform)")
    plt.title("What each sampler targets (per tier)")
    plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "weights_reducible.pdf"))
    plt.savefig(os.path.join(FIGS_DIR, "weights_reducible.png"), dpi=130)
    plt.close()

    with open(os.path.join(REPO_ROOT, "train_log_reducible.txt"), "w") as f:
        f.write("\n".join(logs))

    print(f"\n==== SUMMARY (reducible, {len(cfg['seeds'])} seeds) ====")
    for c in conds:
        s = summary[c]
        print(f"{c:16s} learnable-final {s['learn_final_mean']:.4f}+/-{s['learn_final_std']:.4f}"
              f"  noise-final {s['noise_final_mean']:.3f}  overall {s['overall_final_mean']:.4f}")
    print("\nsampling weight (x uniform) per tier:")
    for meth in ["gradnorm", "reducible"]:
        w = summary["weight_share"][meth]
        print(f"  {meth:10s} easy {w['easy']:.2f}  learnable {w['learnable']:.2f}  noise {w['noise']:.2f}")
    print("\nexamples to reach learnable-subset target (mean), multiplier vs uniform:")
    for rec in summary["efficiency_learnable"]:
        print(f"  tau={rec['tau']}: uniform {rec['uniform']:.0f}  gradnorm_b {rec['gradnorm_biased']:.0f} "
              f"({rec['uniform_over_gradnorm_biased']:.2f}x)  reducible {rec['reducible']:.0f} "
              f"({rec['uniform_over_reducible']:.2f}x)")


if __name__ == "__main__":
    main()
