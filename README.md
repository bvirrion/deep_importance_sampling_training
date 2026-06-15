# Learned Importance Reweighting for SGD Training

This bundle contains a short research paper and a fully self-contained,
reproducible implementation of the idea we discussed: **replacing a fixed
gradient-norm importance-sampling heuristic with a small *learned* reweighting
function that is trained online and then amortised** (frozen and reused without
recomputing gradient norms).

## What's here

```
paper/paper.tex      LaTeX source of the paper
paper/paper.pdf      Compiled paper (7 pages)
code/model.py        Char-level LM in pure NumPy with per-example gradients
code/data.py         Synthetic mixed-difficulty corpus + reweighter features
code/reweighter.py   The learned proposal q_phi and its (stable) meta-update
code/train.py        The three training conditions + amortised deployment
code/run_experiment.py  Multi-seed A/B/C runner; writes results.json + figures
figs/*.pdf, *.png    Generated figures (learning curves, learned weights, variance)
results.json         All numbers behind the paper's tables
train_log.txt        Full training log
```

## The three conditions (all matched on gradient-example budget)

1. **Uniform** — standard SGD, minibatch sampled uniformly.
2. **Grad-norm IS** — the *fixed heuristic*: sample the minibatch from a
   candidate pool with probability proportional to a per-example gradient-norm
   proxy, recomputed **every step**. Importance weights de-bias the update.
3. **Learned (ours)** — a small softmax proposal `q_phi` over cheap forward-pass
   features (loss, grad-norm proxy, entropy, confidence). It is meta-trained to
   match the variance-optimal proposal `p* ∝ ||g||` via a **convex
   cross-entropy** objective, then **frozen after 150/600 steps** and used from
   cheap features alone.

## Headline result (5 seeds)

| Method            | Final val loss      | Grad-norm passes |
|-------------------|---------------------|------------------|
| Uniform SGD       | 0.252 ± 0.019       | 0                |
| Grad-norm IS      | **0.244 ± 0.023**   | 600              |
| Learned (ours)    | 0.262 ± 0.014       | **150 (−75%)**   |

The learned reweighter recovers most of the heuristic's benefit while doing the
expensive per-example gradient-norm computation on only 25% of the steps,
because it amortises the proposal into a cheap reusable function. It also
*rediscovers* the right signal: the largest learned feature weight is on the
gradient-norm feature (see `figs/phi.pdf`).

The honest framing (stated in the paper): the learned method is **not** expected
to beat the every-step oracle heuristic on final loss — its value is matching it
cheaply and being reusable.

## Reproduce

```bash
cd code
python3 run_experiment.py      # ~1-2 min, CPU only, no GPU/torch needed
cd ../paper
pdflatex paper.tex && pdflatex paper.tex
```

Only numpy + matplotlib are required to run the experiment; pdflatex to build
the paper.

## Note on the stability fix

An earlier version minimised the estimator variance directly by REINFORCE. It
**diverges** (phi explodes to ~1e6, loss blows up) because the per-sample cost
spans many orders of magnitude and feeds back into the proposal. The shipped
version instead fits `q_phi` to the known optimum `p* ∝ ||g||` by a convex
KL/cross-entropy objective with a trust-region step. This is documented in both
the code comments (`reweighter.py`) and Section 2.3 of the paper.
