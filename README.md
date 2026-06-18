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
code/data.py         Synthetic graded multi-tier difficulty corpus + reweighter features
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
3. **Learned (ours)** — a small **neural network** `q_phi` (three-hidden-layer MLP
   + softmax over the pool) over cheap forward-pass features (loss, grad-norm
   proxy, entropy, confidence). It is meta-trained to match the variance-optimal
   proposal `p* ∝ ||g||` via a **convex cross-entropy** objective, on a
   **periodic-refresh schedule** (100-step bursts every 1000 steps over 4000
   steps), and used from cheap features alone between bursts.

## Headline result (5 seeds)

| Method            | Final val loss   | Examples to val 0.553 | Grad-norm passes |
|-------------------|------------------|-----------------------|------------------|
| Uniform SGD       | 0.533 ± 0.036    | 243k                  | 0                |
| Grad-norm IS      | 0.526 ± 0.031    | **194k**              | 4000             |
| Learned (ours)    | **0.521 ± 0.035**| 199k                  | **400 (−90%)**   |

On this harder, graded-difficulty corpus the task does **not** saturate within
the budget, so the importance-sampling benefit persists to the final loss. The
learned reweighter reaches the **lowest** final loss — significantly below uniform
(paired `t=3.66`) and within noise of the every-step heuristic (`t=2.42`) — and is
also more **sample-efficient**, reaching the target loss with ~18–20% fewer
gradient examples than uniform. It does all this while computing the expensive
per-example gradient-norm target on only 10% of the steps, because it amortises
the proposal into a cheap, periodically-refreshed neural network. It also
*rediscovers* the right signal: the gradient-norm feature has the largest saliency
in the trained network (with predictive entropy close behind; see `figs/phi.pdf`).

The honest framing (stated in the paper): the learned method **matches** the
every-step oracle heuristic (the gap is within noise) and beats uniform — its
value is reaching the oracle's accuracy and efficiency cheaply (90% fewer
selection passes) and being reusable.

## Second experiment: concentrated difficulty (when the pain is rare)

A second corpus stresses the regime the method is built for: **99% trivially-easy**
deterministic text and a **rare (~1%) but learnable hard tier** (a fixed codebook
of deterministic phrases in rare letters absent from the easy text). Because the
easy 99% dominates the aggregate loss (~0.092 for all methods — the benefit is
invisible there), we measure the loss on the **hard 1% subset** and report how
many more examples uniform needs to reach it.

| Hard-subset target | Uniform | Grad-norm IS | Learned | **Uniform / IS** |
|--------------------|---------|--------------|---------|------------------|
| 1.10               | 571k    | 373k         | 399k    | 1.51× / 1.42×    |
| 0.90               | 689k    | 427k         | 452k    | 1.59× / 1.51×    |
| 0.70               | 866k    | 510k         | 545k    | **1.68× / 1.57×**|

**Uniform SGD needs ~1.5–1.7× more training examples than the importance samplers
to reach the same hard-subset loss** (the multiplier grows as the target tightens).
Both IS methods significantly beat uniform (paired `t≈3.2`) and the learned
reweighter matches the every-step oracle (`t=0.2`). Since the update is unbiased,
this is a pure variance-reduction *speed-up* — it shows up as sample efficiency on
the rare tier, not as a different converged loss. Run with
`python3 -m code.run_experiment_concentrated` (writes `results_concentrated.json`
and `figs/curves_concentrated.*`).

## Third experiment: trading unbiasedness for speed (biased sampling)

What if we keep the change of measure for **sampling** but **drop the importance
weights** and train on the plain-mean gradient? The update is then *biased* — it
steers training toward high-gradient examples (soft hard-example mining). On the
concentrated task this is much faster (5 seeds, hard-subset loss):

| Method                       | Final hard loss | Examples to 0.9 | Speed-up vs uniform |
|------------------------------|-----------------|-----------------|---------------------|
| Uniform SGD                  | 0.835 ± 0.182   | 582k            | 1.0×                |
| Grad-norm IS (unbiased)      | 0.470 ± 0.059   | 427k            | 1.5×                |
| **Grad-norm sampling, biased** | **0.368 ± 0.045** | **121k**      | **5.4×**            |
| Learned sampling, biased     | 0.375 ± 0.050   | 164k            | 3.9×                |

**~5× faster than uniform** (and ~3.6× faster than the unbiased sampler) to reach
the rare-tier target, with the lowest final hard loss — but at two honest costs:
**(i)** it's biased, so on the spread-out *graded* task it instead **hurts** (0.61
vs 0.55); **(ii)** it's less stable — it **diverges for any learning rate ≥ 1.0**,
while uniform is stable up to 3.0. A targeted tool for the rare-but-learnable
regime. Run with `python3 -m code.run_experiment_biased` (writes `results_biased.json`
and `figs/curves_biased.*`).

## Fourth experiment: learning what remains to be learned (reducible improvement)

Gradient norm can't tell a **hard-but-learnable** example from **hard-but-unlearnable
noise** — both have big gradients, so a gradient-norm sampler chases the junk. Here
the sampler is instead trained to estimate each input's **reducible improvement**:
periodically we cluster the pool by content, take a step on each cluster, and
measure how much it reduces loss on a held-out **learnable** reference set (RHO-LOSS
/ realized-influence, amortized into the learned scorer). Corpus: 90% easy + 5%
learnable-hard + 5% random noise (disjoint letter sets). 5 seeds:

| Sampler (biased)             | Targets noise | Targets learnable | Learnable loss | Speed-up vs uniform |
|------------------------------|---------------|-------------------|----------------|---------------------|
| Grad-norm                    | **2.5× (!)**  | 0.3×              | 0.305 *(≈uniform)* | 2.2×            |
| **Reducible (ours)**         | 0.9×          | **1.6×**          | **0.260**      | **4.2×**            |

The gradient-norm sampler's **biggest target is the unlearnable noise**, so it ends
up no better than uniform on the learnable capability. The reducible scorer learns
to **ignore the noise** and focus on the learnable tier, reaching that capability
~4.2× faster than uniform (and ~1.9× faster than gradnorm). Cost: a few reference
evals per burst + a held-out set defining the target capability.

**Biased vs. unbiased.** The speed-up uses the biased plain-mean update. If we
instead *debias* (use the same learned change of measure as an importance-sampling
proposal with weights `1/(N·p_i)`, so the gradient estimator is unbiased), the
speed-up **vanishes** — the unbiased reducible sampler falls back to ~uniform
(0.292 final, 0.9× examples-to-target). That's expected: an unbiased estimator
shares uniform's expected trajectory for any proposal, so the gain was intrinsically
from the *biased* objective (concentrating learning on the learnable tier). What
debiasing buys instead is **robustness** — it stays stable at lr=2.0 where the
biased version starts to diverge. (Unbiased grad-norm IS doesn't even match uniform
here: its ∝‖g‖ proposal dumps weight on the noise.) Run with
`python3 -m code.run_experiment_reducible` (writes `results_reducible.json`,
`figs/curves_reducible.*`, `figs/weights_reducible.*`).

## Reproduce

```bash
python3 -m code.run_experiment                # main graded-difficulty experiment
python3 -m code.run_experiment_concentrated   # concentrated 99%/1% experiment
python3 -m code.run_experiment_biased         # biased-sampling variant + lr sweep
python3 -m code.run_experiment_reducible      # reducible-improvement sampler
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
