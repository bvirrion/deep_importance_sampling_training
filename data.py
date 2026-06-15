"""
Data utilities: build a char-level dataset of (context, target) pairs and
provide cheap per-example features for the learned reweighter.

We synthesise a corpus with deliberately *heterogeneous difficulty* so that
importance sampling has something to exploit: most of the text is highly
predictable (easy, low-loss) while a minority of positions are genuinely hard
(high entropy). A uniform sampler wastes most of its gradient budget on the
easy majority; a good reweighter should concentrate on the informative minority.
"""

import numpy as np


def make_corpus(n_chars=200_000, seed=0):
    """Synthesise a corpus with mixed easy/hard structure.

    Vocabulary: lowercase a-z plus space and a few digits.
    - 'Easy' regions: a small set of repeating template words (highly
      predictable given context).
    - 'Hard' regions: short bursts of near-random digit strings (high entropy,
      large gradients), interleaved at low frequency.
    """
    rng = np.random.default_rng(seed)
    vocab = list("abcdefghijklmnopqrstuvwxyz 0123456789")
    stoi = {c: i for i, c in enumerate(vocab)}

    easy_words = [
        "the cat sat on the mat ",
        "a dog ran in the park ",
        "she sells sea shells ",
        "we go to the shop now ",
        "rain falls on the town ",
    ]

    out = []
    while len(out) < n_chars:
        if rng.random() < 0.85:
            # easy, predictable template
            w = easy_words[rng.integers(len(easy_words))]
            out.extend(w)
        else:
            # hard, high-entropy burst of digits
            k = rng.integers(4, 9)
            for _ in range(k):
                out.append("0123456789"[rng.integers(10)])
            out.append(" ")
    text = "".join(out)[:n_chars]
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return data, vocab, stoi


def build_examples(data, context=8):
    """Turn a 1-D stream into (X, y) of contexts and next-char targets."""
    N = len(data) - context
    X = np.empty((N, context), dtype=np.int64)
    y = np.empty((N,), dtype=np.int64)
    for i in range(N):
        X[i] = data[i:i + context]
        y[i] = data[i + context]
    return X, y


def split(X, y, frac_train=0.9):
    n = X.shape[0]
    n_tr = int(n * frac_train)
    return (X[:n_tr], y[:n_tr]), (X[n_tr:], y[n_tr:])


def reweighter_features(model, cache, y):
    """Cheap, label-aware features for each example in a batch.

    All of these are byproducts of the forward pass (no extra parameter-sized
    computation), so the reweighter is cheap to evaluate:
      f0: current per-example loss (how 'painful' the example is now)
      f1: logit-gradient norm (cheap gradient-magnitude proxy)
      f2: predictive entropy (how uncertain the model is)
      f3: max predicted prob (confidence)
      f4: bias term (1.0)
    Returns array (B, 5).
    """
    probs = cache["probs"]
    B = y.shape[0]
    loss = model.loss_per_example(cache, y)
    gnorm = model.per_example_gradnorm(cache, y)
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    maxp = np.max(probs, axis=1)
    f = np.stack([loss, gnorm, ent, maxp, np.ones(B)], axis=1)
    # standardise the first four columns to keep the reweighter conditioning sane
    mu = f[:, :4].mean(axis=0, keepdims=True)
    sd = f[:, :4].std(axis=0, keepdims=True) + 1e-6
    f[:, :4] = (f[:, :4] - mu) / sd
    return f
