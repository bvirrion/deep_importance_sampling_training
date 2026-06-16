"""
Data utilities: build a char-level dataset of (context, target) pairs and
provide cheap per-example features for the learned reweighter.

We synthesise a corpus with deliberately *heterogeneous difficulty* so that
importance sampling has something to exploit. Rather than a simple easy/hard
binary split, we use a graded, multi-tier mixture that spans a continuum of
difficulty and is, overall, substantially harder to fit:

  - Tier A (easy):   deterministic template phrases -- highly predictable given
                     context, low loss, small gradients.
  - Tier B (medium): stochastic-slot phrases whose words are drawn at random
                     from small pools, so context only *partially* predicts the
                     next character -- intermediate, branch-dependent gradients.
  - Tier C (hard):   longer high-entropy alphanumeric bursts -- near-random,
                     high loss, large gradients.

The medium tier is the new source of heterogeneity: it fills in the difficulty
spectrum between the trivially-easy and the near-random, and lowering the easy
fraction makes the global task genuinely harder (it does not saturate within the
training budget). A uniform sampler spends most of its gradient budget on the
easy/medium-predictable majority; a good reweighter should concentrate on the
informative, high-gradient positions across the whole difficulty continuum.
"""

import numpy as np


def make_corpus(n_chars=200_000, seed=0):
    """Synthesise a corpus with a graded, multi-tier difficulty structure.

    Vocabulary: lowercase a-z, space, period, comma, and digits 0-9.

    Mixture per phrase (tunable knobs):
      - 45% Tier A: deterministic template phrases (easy).
      - 35% Tier B: stochastic-slot phrases -- a fixed skeleton with random
        word choices from small pools, so the next char is only partly
        determined by context (medium).
      - 20% Tier C: longer high-entropy alphanumeric bursts (hard).
    """
    rng = np.random.default_rng(seed)
    vocab = list("abcdefghijklmnopqrstuvwxyz .,0123456789")
    stoi = {c: i for i, c in enumerate(vocab)}

    # Tier A: fully deterministic, highly predictable templates.
    easy_words = [
        "the cat sat on the mat. ",
        "a dog ran in the park. ",
        "she sells sea shells. ",
        "we go to the shop now. ",
        "rain falls on the town. ",
    ]

    # Tier B: phrase skeletons with random slots. The skeleton is predictable but
    # each slot branches over a small pool, producing intermediate difficulty.
    subjects = ["a fox", "the bird", "my friend", "her sister", "the old man",
                "a sailor", "the queen", "his brother"]
    verbs = ["found", "lost", "painted", "chased", "carried", "counted",
             "traded", "hid"]
    objects = ["a red box", "some bright coins", "the small key", "a green leaf",
               "two grey stones", "the glass jar", "a silver ring", "the torn map"]
    places = ["near the river", "by the tall gate", "under the bridge",
              "behind the barn", "on the long road", "inside the cave",
              "above the bay", "across the field"]

    def medium_phrase():
        return (f"{subjects[rng.integers(len(subjects))]} "
                f"{verbs[rng.integers(len(verbs))]} "
                f"{objects[rng.integers(len(objects))]} "
                f"{places[rng.integers(len(places))]}, ")

    # Tier C: longer high-entropy bursts mixing digits and letters.
    hard_alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

    out = []
    while len(out) < n_chars:
        r = rng.random()
        if r < 0.45:
            # Tier A: easy, deterministic template
            out.extend(easy_words[rng.integers(len(easy_words))])
        elif r < 0.80:
            # Tier B: medium, stochastic-slot phrase
            out.extend(medium_phrase())
        else:
            # Tier C: hard, high-entropy alphanumeric burst
            k = rng.integers(8, 15)
            for _ in range(k):
                out.append(hard_alphabet[rng.integers(len(hard_alphabet))])
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
