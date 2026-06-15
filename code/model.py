"""
A small char-level language model implemented in pure NumPy.

The model is intentionally simple (an embedding + one hidden layer MLP over a
fixed context window, predicting the next character) so that the whole training
pipeline is self-contained, CPU-only, deterministic and fast. It is, however, a
genuine next-token language model trained by stochastic gradient descent, which
is exactly the setting where minibatch importance sampling is defined.

The key capability we need for importance sampling research is *per-example*
loss and gradient information, so the code is written to expose those.
"""

import numpy as np


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


class CharLM:
    """Fixed-context char-level MLP language model.

    Architecture:
        context of C characters -> embedding (D each) -> concat (C*D)
        -> hidden (H) with tanh -> output logits (V)
    """

    def __init__(self, vocab_size, context=8, emb_dim=24, hidden=128, seed=0):
        self.V = vocab_size
        self.C = context
        self.D = emb_dim
        self.H = hidden
        rng = np.random.default_rng(seed)
        # Parameter initialisation (small random).
        self.E = rng.normal(0, 0.1, size=(self.V, self.D))
        self.W1 = rng.normal(0, 1.0 / np.sqrt(self.C * self.D), size=(self.C * self.D, self.H))
        self.b1 = np.zeros(self.H)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(self.H), size=(self.H, self.V))
        self.b2 = np.zeros(self.V)

    # ---- parameter helpers -------------------------------------------------
    def params(self):
        return [self.E, self.W1, self.b1, self.W2, self.b2]

    def copy_params_from(self, other):
        self.E = other.E.copy()
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

    # ---- forward / backward ------------------------------------------------
    def forward(self, X):
        """X: (B, C) int array of context characters. Returns cache."""
        B = X.shape[0]
        emb = self.E[X]                       # (B, C, D)
        flat = emb.reshape(B, self.C * self.D)  # (B, C*D)
        pre_h = flat @ self.W1 + self.b1      # (B, H)
        h = np.tanh(pre_h)                    # (B, H)
        logits = h @ self.W2 + self.b2        # (B, V)
        probs = softmax(logits, axis=1)
        cache = dict(X=X, emb=emb, flat=flat, pre_h=pre_h, h=h, probs=probs)
        return cache

    def loss_per_example(self, cache, y):
        probs = cache["probs"]
        B = y.shape[0]
        p = probs[np.arange(B), y]
        return -np.log(p + 1e-12)            # (B,)

    def backward(self, cache, y, weights=None):
        """Return gradients of the (optionally weighted) mean loss.

        weights: (B,) nonnegative per-example weights that multiply each
        example's contribution. If None, uniform mean is used.
        """
        X = cache["X"]
        h = cache["h"]
        flat = cache["flat"]
        pre_h = cache["pre_h"]
        probs = cache["probs"]
        B = y.shape[0]

        if weights is None:
            w = np.ones(B) / B
        else:
            w = weights.astype(np.float64)

        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0       # (B, V) = dCE/dlogits per example
        dlogits *= w[:, None]                 # apply per-example weight

        dW2 = h.T @ dlogits                    # (H, V)
        db2 = dlogits.sum(axis=0)              # (V,)
        dh = dlogits @ self.W2.T               # (B, H)
        dpre_h = dh * (1.0 - h ** 2)           # tanh'
        dW1 = flat.T @ dpre_h                  # (C*D, H)
        db1 = dpre_h.sum(axis=0)               # (H,)
        dflat = dpre_h @ self.W1.T             # (B, C*D)
        demb = dflat.reshape(B, self.C, self.D)

        dE = np.zeros_like(self.E)
        np.add.at(dE, X, demb)                 # scatter-add into embedding rows

        return [dE, dW1, db1, dW2, db2]

    def sgd_step(self, grads, lr):
        for p, g in zip(self.params(), grads):
            p -= lr * g

    # ---- per-example gradient-norm (for the heuristic baseline) ------------
    def per_example_gradnorm(self, cache, y):
        """Cheap approximation to the per-example gradient norm.

        We use the norm of the gradient w.r.t. the logits, which is the standard
        cheap proxy used in importance-sampling-for-SGD work (it is the dominant
        and cheapest-to-compute term, requiring no parameter-sized scatter).
        """
        probs = cache["probs"]
        B = y.shape[0]
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        return np.linalg.norm(dlogits, axis=1)  # (B,)
