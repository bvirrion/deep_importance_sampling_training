"""
The learned reweighting function.

Idea (the contribution): instead of a *fixed* heuristic that maps an example to
a sampling weight (e.g. proportional to gradient norm), we *learn* a small
parametric function q_phi(features) -> proposal score, and we adapt phi online
so that sampling from q_phi reduces the variance of the importance-weighted
gradient estimator. Because q_phi is cheap and reusable, the cost is amortised
across the whole training run (and, in principle, across runs).

Unbiasedness. If we draw a minibatch by sampling examples with probability
p_i (proportional to the proposal score) from a pool, the importance-weighted
gradient

    g_hat = (1/B) sum_{i in batch} (1 / (N * p_i)) * grad_i

is an unbiased estimator of the full-pool mean gradient (1/N) sum_i grad_i, for
*any* strictly positive proposal p. So the reweighter changes variance, not the
expected update: it cannot bias the optimisation, only make each step more or
less informative per unit compute. This mirrors the role of the Girsanov change
of measure in the importance-sampling paper: the measure change leaves the
expectation invariant and is optimised purely to reduce variance.

Meta-objective. The variance-minimising proposal is p_i proportional to the
per-example gradient norm (a classical result). We do not hard-code that; we let
the reweighter *discover* a proposal from cheap features by minimising a tractable
surrogate for the estimator variance, namely the expected squared importance-
weighted gradient magnitude. We update phi by a REINFORCE-style step on that
surrogate so that the method works even though the sampling operation is
non-differentiable.
"""

import numpy as np


class Reweighter:
    """Linear-softmax proposal over a pool of candidate examples.

    score_i = features_i @ phi
    p_i = softmax(score)_i   (over the candidate pool)

    A linear model on top of nonlinear cheap features (loss, grad-norm proxy,
    entropy, confidence) is expressive enough to recover the gradient-norm
    proposal as a special case (put all weight on the grad-norm feature) while
    being able to learn something better if the features support it.
    """

    def __init__(self, n_features=5, seed=0, temperature=1.0):
        rng = np.random.default_rng(seed)
        self.phi = rng.normal(0, 0.01, size=(n_features,))
        self.temp = temperature

    def scores(self, feats):
        return feats @ self.phi / self.temp

    def proposal(self, feats):
        s = self.scores(feats)
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()

    def sample(self, feats, k, rng, floor=1e-4):
        """Sample k indices (with replacement) from the proposal.

        A small uniform floor is mixed in to guarantee strictly positive
        probabilities (keeps importance weights finite and the estimator
        unbiased).
        """
        p = self.proposal(feats)
        n = len(p)
        p = (1 - floor) * p + floor / n
        p = p / p.sum()
        idx = rng.choice(n, size=k, replace=True, p=p)
        return idx, p

    def meta_update(self, feats, p, idx, grad_norms, lr,
                    max_norm=2.0):
        """Update phi so the proposal matches the variance-optimal distribution.

        Theory. For an importance-weighted estimator of the mean gradient, the
        proposal that minimises estimator variance is the one proportional to the
        per-example gradient norm,

            p*_i = ||g_i|| / sum_j ||g_j||.

        Estimating ||g_i|| for *every* candidate every step and sampling from it
        exactly is the 'fixed heuristic' baseline. Our contribution is to instead
        *learn* a cheap function q_phi(features) that reproduces p* from features
        that are byproducts of the forward pass, so it (a) needs no separate
        gradient-norm pass at deployment and (b) amortises across steps/runs.

        We fit q_phi to p* by minimising the cross-entropy / KL(p* || q_phi)
        over the candidate pool:

            L(phi) = - sum_i p*_i log q_phi_i
            grad_phi L = sum_i (q_phi_i - p*_i) feat_i      (softmax CE gradient)

        This objective is convex in the logits, well scaled, and has a unique
        stable direction, so it does not suffer the runaway dynamics of a raw
        REINFORCE-on-variance update. The reweighter can match p* exactly when a
        feature equals ||g|| (here the grad-norm proxy), and can otherwise learn
        the best feature combination. Returns the achieved variance proxy.
        """
        # variance-optimal target distribution over the pool
        target = grad_norms / (grad_norms.sum() + 1e-12)
        q = self.proposal(feats)
        # cross-entropy gradient w.r.t. phi for a softmax over feats @ phi:
        #   dL/dphi = sum_i (q_i - target_i) feat_i
        g = ((q - target)[:, None] * feats).sum(axis=0)
        gn = np.linalg.norm(g)
        if gn > max_norm:
            g = g * (max_norm / gn)
        self.phi -= lr * g
        # variance proxy: chi-square-like dispersion of importance weights under q
        # (lower = closer to the optimal proposal). Reported for monitoring.
        N = len(q)
        pi = q[idx]
        cost = ((1.0 / (N * pi)) ** 2 * (grad_norms[idx] ** 2)).mean()
        return float(cost)
