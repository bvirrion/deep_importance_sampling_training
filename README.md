# Learned Importance Reweighting for SGD Training

This bundle contains a short research paper and a fully self-contained,
reproducible implementation of the idea we discussed: **replacing a fixed
gradient-norm importance-sampling heuristic with a small *neural network* that
maps the batch to a change of measure, is trained online to sample more where the
gradient is larger, and is amortised by *periodic refresh*** (meta-trained in
short bursts, otherwise used frozen without recomputing gradient norms).

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
3. **Learned (ours)** — a small **neural network** `q_phi` (one-hidden-layer MLP
   + softmax over the pool) over cheap forward-pass features (loss, grad-norm
   proxy, entropy, confidence). It is meta-trained to match the variance-optimal
   proposal `p* ∝ ||g||` via a **convex cross-entropy** objective, on a
   **periodic-refresh schedule** (100-step bursts every 1000 steps over 4000
   steps), and used from cheap features alone between bursts.

## Headline result (5 seeds)

| Method            | Final val loss   | Examples to val 0.256 | Grad-norm passes |
|-------------------|------------------|-----------------------|------------------|
| Uniform SGD       | 0.236 ± 0.018    | 84k                   | 0                |
| Grad-norm IS      | 0.238 ± 0.020    | **63k**               | 4000             |
| Learned (ours)    | 0.240 ± 0.021    | 67k                   | **400 (−90%)**   |

By 4000 steps the task saturates, so all three reach a statistically
indistinguishable final loss. The payoff of importance sampling is in **sample
efficiency**: the learned reweighter (like the heuristic) reaches the target loss
with ~20% fewer gradient examples than uniform, while doing the expensive
per-example gradient-norm computation on only 10% of the steps — because it
amortises the proposal into a cheap, periodically-refreshed neural network. It
also *rediscovers* the right signal: the gradient-norm feature has by far the
largest saliency in the trained network (see `figs/phi.pdf`).

The honest framing (stated in the paper): the learned method is **not** expected
to beat the every-step oracle heuristic — its value is matching its efficiency
cheaply (90% fewer selection passes) and being reusable.

## Reproduce

```bash
python3 -m code.run_experiment   # run from the repo root; CPU only, no GPU/torch
cd paper
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
