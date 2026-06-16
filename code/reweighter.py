"""
The learned reweighting function: a small neural network that maps a batch to a
change of measure.

Idea (the contribution): instead of a *fixed* heuristic that maps an example to
a sampling weight (e.g. proportional to gradient norm), we *learn* a small
neural network q_phi(features) -> proposal over the candidate pool, and we adapt
it online so that sampling from q_phi reduces the variance of the importance-
weighted gradient estimator. Because q_phi is cheap and reusable, the cost is
amortised across the training run: we refresh it in short bursts and otherwise
sample from it frozen, on cheap forward-pass features alone.

This is the discrete-SGD analogue of the construction in Deep Importance
Sampling (arXiv:2007.02692): there a neural network maps the past of a
trajectory to a Girsanov change of measure on path space; here a neural network
maps the per-example features of a candidate pool ("the batch space") to a
discrete probability measure over the pool, used for the importance-sampling
change of measure.

Unbiasedness. If we draw a minibatch by sampling examples with probability
p_i (proportional to the proposal score) from a pool, the importance-weighted
gradient

    g_hat = (1/B) sum_{i in batch} (1 / (N * p_i)) * grad_i

is an unbiased estimator of the full-pool mean gradient (1/N) sum_i grad_i, for
*any* strictly positive proposal p. So the network changes variance, not the
expected update: it cannot bias the optimisation, only make each step more or
less informative per unit compute. This mirrors the role of the Girsanov change
of measure: the measure change leaves the expectation invariant and is optimised
purely to reduce variance.

Meta-objective. The variance-minimising proposal is p_i proportional to the
per-example gradient norm (a classical result), i.e. "sample more where the
gradient is larger". We train the network to reproduce that target p* from cheap
features by minimising the cross-entropy / KL(p* || q_phi). This convex (in the
output logits) surrogate has the same optimum as direct variance minimisation
but well-behaved dynamics, unlike a raw REINFORCE-on-variance update which
diverges (the per-sample cost spans many orders of magnitude and feeds back into
the proposal).
"""

import numpy as np


class Reweighter:
    """Neural-network proposal over a pool of candidate examples.

    Architecture (a one-hidden-layer MLP shared across pool examples):

        h_i     = tanh(feats_i @ W1 + b1)        (H,)
        score_i = h_i @ W2 + b2                   scalar
        q_i     = softmax(score / temperature)_i  (over the candidate pool)

    The network maps the batch (its per-example feature vectors) to a probability
    measure q over the pool -- the change of measure for importance sampling. It
    can recover the gradient-norm proposal as a special case while being able to
    learn richer difficulty structure that a linear model or fixed proxy cannot.
    """

    def __init__(self, n_features=5, hidden=16, seed=0, temperature=1.0):
        rng = np.random.default_rng(seed)
        # Small-random init; He-ish scaling for the tanh hidden layer.
        self.W1 = rng.normal(0, 1.0 / np.sqrt(n_features), size=(n_features, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, 1))
        self.b2 = np.zeros(1)
        self.temp = temperature

    # ---- forward -----------------------------------------------------------
    def _forward(self, feats):
        """Return (scores, cache) for a pool feature matrix feats (n, F)."""
        pre_h = feats @ self.W1 + self.b1        # (n, H)
        h = np.tanh(pre_h)                        # (n, H)
        scores = (h @ self.W2 + self.b2)[:, 0]    # (n,)
        cache = dict(feats=feats, h=h)
        return scores, cache

    def scores(self, feats):
        s, _ = self._forward(feats)
        return s / self.temp

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

    # ---- meta-update -------------------------------------------------------
    def meta_update(self, feats, p, idx, grad_norms, lr, max_norm=2.0):
        """Update the network so the proposal matches the variance-optimal target.

        Theory. For an importance-weighted estimator of the mean gradient, the
        proposal that minimises estimator variance is the one proportional to the
        per-example gradient norm,

            p*_i = ||g_i|| / sum_j ||g_j||,

        i.e. "sample more where the gradient descent is larger". We fit q_phi to
        p* by minimising the cross-entropy / KL(p* || q_phi) over the pool:

            L = - sum_i p*_i log q_phi_i.

        For a softmax over per-example scores, the gradient w.r.t. the scores is
        the standard clean form

            dL/dscore_i = q_i - p*_i,

        which we backpropagate through the MLP. The objective is convex in the
        output logits, well scaled (q and p* are both probability vectors), and
        trust-regioned (we cap the global gradient norm), so it avoids the runaway
        dynamics of a raw REINFORCE-on-variance update. Returns the achieved
        variance proxy for monitoring.
        """
        # variance-optimal target distribution over the pool
        target = grad_norms / (grad_norms.sum() + 1e-12)

        scores, cache = self._forward(feats)
        s = scores / self.temp
        s = s - s.max()
        e = np.exp(s)
        q = e / e.sum()                            # (n,)

        h = cache["h"]                             # (n, H)
        n = len(q)

        # dL/dscore_i = q_i - p*_i  (softmax cross-entropy); temperature scales it.
        dscore = (q - target) / self.temp         # (n,)

        # Backprop through score_i = h_i @ W2 + b2 and h = tanh(feats @ W1 + b1).
        dW2 = h.T @ dscore[:, None]                # (H, 1)
        db2 = np.array([dscore.sum()])             # (1,)
        dh = dscore[:, None] * self.W2[:, 0][None, :]   # (n, H)
        dpre_h = dh * (1.0 - h ** 2)               # tanh'
        dW1 = feats.T @ dpre_h                      # (F, H)
        db1 = dpre_h.sum(axis=0)                    # (H,)

        grads = [dW1, db1, dW2, db2]
        # global-norm trust region
        gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        scale = (max_norm / gn) if gn > max_norm else 1.0
        self.W1 -= lr * scale * dW1
        self.b1 -= lr * scale * db1
        self.W2 -= lr * scale * dW2
        self.b2 -= lr * scale * db2

        # variance proxy: dispersion of importance weights under q (lower = closer
        # to the optimal proposal). Reported for monitoring only.
        pi = q[idx]
        cost = ((1.0 / (n * pi)) ** 2 * (grad_norms[idx] ** 2)).mean()
        return float(cost)

    # ---- feature attribution (for the "what it learned" figure) ------------
    def feature_saliency(self, feats):
        """Mean over the pool of |d score_i / d feature_k|.

        With a nonlinear network there is no single weight vector to plot, so we
        report input saliency as the analogue: which input features most move the
        proposal score. The grad-norm feature should dominate, showing the network
        rediscovered p* proportional to ||g||.
        """
        _, cache = self._forward(feats)
        h = cache["h"]                             # (n, H)
        # d score / d feats = (1 - h^2) * W2  propagated through W1
        dpre_h = (1.0 - h ** 2) * self.W2[:, 0][None, :]   # (n, H)
        dfeats = dpre_h @ self.W1.T                # (n, F)
        return np.abs(dfeats).mean(axis=0)         # (F,)
