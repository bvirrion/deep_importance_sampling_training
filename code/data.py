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


def _make_hard_codebook(n_hard_patterns, fixed_seed=12345):
    """Build a fixed set of K distinct, deterministic rare-letter phrases.

    The codebook is generated from a dedicated fixed RNG so the hard target is
    identical across corpus seeds: it is therefore a *learnable* (memorizable)
    target, not random noise. With several distinct patterns it is data-hungry --
    a model must see each pattern enough times to learn it, which uniform sampling
    (spending only ~1% of its budget here) is slow to do, while importance
    sampling reduces the gradient-estimate variance on these rare high-gradient
    examples and so reaches the same loss with fewer examples. Patterns use only
    the rare letters {j,k,q,v,w,x,z} so they are never taught by the
    (common-letter) easy text.
    """
    cb_rng = np.random.default_rng(fixed_seed)
    rare = list("jkqvwxz")
    phrases = []
    seen = set()
    while len(phrases) < n_hard_patterns:
        L = int(cb_rng.integers(18, 28))
        chars = []
        for _ in range(L):
            # occasional space to give word-like structure, else a rare letter
            if chars and cb_rng.random() < 0.18:
                chars.append(" ")
            else:
                chars.append(rare[cb_rng.integers(len(rare))])
        p = "".join(chars).strip()
        if not p or p in seen:
            continue
        seen.add(p)
        phrases.append(p + " ")
    return phrases


def make_corpus_concentrated(n_chars=200_000, seed=0, easy_frac=0.99,
                             n_hard_patterns=6):
    """Synthesise a *concentrated*-difficulty corpus: ~99% trivially easy and a
    rare (~1%) hard tier, where the rare hard cases dominate the error.

    This is the regime where the informative "pain" is a tiny fraction of the
    data. A uniform sampler spends ~99% of its gradient budget on examples that
    are already mastered (near-zero gradient); importance sampling reduces the
    variance of the gradient estimate by concentrating on the rare high-gradient
    examples, and so reaches a given loss on the hard tier with substantially
    fewer training examples.

      - Easy (~easy_frac): a few fully deterministic template phrases built from
        common letters only, deliberately EXCLUDING the rare letters
        {j,q,k,v,w,x,z}. Near-zero entropy, mastered quickly so L_easy -> ~0.
      - Hard (~1-easy_frac): a small fixed codebook of n_hard_patterns distinct,
        deterministic rare-letter phrases. Deterministic so the loss is reducible
        (learnable), but with several patterns it is data-hungry: under uniform
        sampling the rare cases stay under-trained for longer (high L_hard, so
        they make a big chunk of the total error despite being rare); importance
        sampling reaches the same hard-tier loss with fewer examples.

    Returns (data, vocab, stoi, char_is_hard), where char_is_hard is a per-character
    boolean array marking characters that belong to a hard segment. Use
    hard_example_mask() to turn it into an example-level mask aligned with the
    targets produced by build_examples().
    """
    rng = np.random.default_rng(seed)
    vocab = list("abcdefghijklmnopqrstuvwxyz .")
    stoi = {c: i for i, c in enumerate(vocab)}

    # Easy: deterministic, common-letter-only (no j,q,k,v,w,x,z).
    easy_phrases = [
        "the cat sat on the mat. ",
        "a dog ran to the den. ",
        "she sells sea shells. ",
        "the sun is on the hill. ",
        "ned has ten red hens. ",
    ]
    # Hard: a small fixed codebook of deterministic rare-letter phrases.
    hard_phrases = _make_hard_codebook(n_hard_patterns)

    out = []
    is_hard = []
    while len(out) < n_chars:
        if rng.random() < easy_frac:
            w = easy_phrases[rng.integers(len(easy_phrases))]
            out.extend(w)
            is_hard.extend([False] * len(w))
        else:
            w = hard_phrases[rng.integers(len(hard_phrases))]
            out.extend(w)
            is_hard.extend([True] * len(w))
    text = "".join(out)[:n_chars]
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    char_is_hard = np.array(is_hard[:n_chars], dtype=bool)
    return data, vocab, stoi, char_is_hard


def hard_example_mask(char_is_hard, context=8):
    """Example-level hard mask aligned with build_examples() targets.

    build_examples maps example i to the target character at position i+context,
    so example i is 'hard' iff that target character lies in a hard segment.
    """
    N = len(char_is_hard) - context
    return char_is_hard[context:context + N]


def _codebook(alphabet, n_patterns, fixed_seed=12345):
    """K distinct deterministic phrases over a given alphabet (fixed across seeds)."""
    rng = np.random.default_rng(fixed_seed)
    alpha = list(alphabet)
    phrases, seen = [], set()
    while len(phrases) < n_patterns:
        L = int(rng.integers(18, 28))
        chars = []
        for _ in range(L):
            if chars and rng.random() < 0.18:
                chars.append(" ")
            else:
                chars.append(alpha[rng.integers(len(alpha))])
        p = "".join(chars).strip()
        if p and p not in seen:
            seen.add(p); phrases.append(p + " ")
    return phrases


def make_corpus_reducible(n_chars=200_000, seed=0, easy_frac=0.90,
                          learn_frac=0.05, n_learn_patterns=6):
    """Corpus where high gradient does NOT imply learnable.

    Three tiers, with the two HIGH-GRADIENT tiers placed in disjoint content
    regions so a content-aware scorer can separate them but a gradient-norm
    sampler cannot:
      - tier 0, easy (~easy_frac): deterministic common-letter phrases (low grad).
      - tier 1, learnable-hard (~learn_frac): a FIXED codebook of deterministic
        phrases in rare-letter set A = {j,k,q,v} -- high gradient, *reducible*
        (a step on one generalises to the others).
      - tier 2, noise (~rest): freshly-RANDOM tokens in disjoint set B = {w,x,z},
        regenerated every occurrence -- high gradient, *irreducible* (a step on one
        does not transfer to held-out data).

    Returns (data, vocab, stoi, tier) with tier a per-character label in {0,1,2}.
    """
    rng = np.random.default_rng(seed)
    vocab = list("abcdefghijklmnopqrstuvwxyz .")
    stoi = {c: i for i, c in enumerate(vocab)}

    easy_phrases = [
        "the cat sat on the mat. ",
        "a dog ran to the den. ",
        "she sells sea shells. ",
        "the sun is on the hill. ",
        "ned has ten red hens. ",
    ]
    set_A = "jkqv"          # learnable-hard alphabet (fixed codebook)
    set_B = "wxz"           # noise alphabet (random)
    learn_phrases = _codebook(set_A, n_learn_patterns)

    out, tier = [], []
    while len(out) < n_chars:
        r = rng.random()
        if r < easy_frac:
            w = easy_phrases[rng.integers(len(easy_phrases))]
            out.extend(w); tier.extend([0] * len(w))
        elif r < easy_frac + learn_frac:
            w = learn_phrases[rng.integers(len(learn_phrases))]
            out.extend(w); tier.extend([1] * len(w))
        else:
            k = rng.integers(18, 28)
            seg = [set_B[rng.integers(len(set_B))] if rng.random() >= 0.18 else " "
                   for _ in range(int(k))]
            seg = list("".join(seg).strip()) + [" "]
            out.extend(seg); tier.extend([2] * len(seg))
    text = "".join(out)[:n_chars]
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    tier = np.array(tier[:n_chars], dtype=np.int64)
    return data, vocab, stoi, tier


def tier_example_mask(tier, context=8):
    """Per-example tier labels aligned with build_examples() targets."""
    N = len(tier) - context
    return tier[context:context + N]


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


def reweighter_features_content(model, cache, y):
    """Cheap features (reweighter_features) PLUS a content signature.

    The content signature is the mean context embedding E[context].mean(axis=1)
    (a model-aware, cheap, forward-pass byproduct). Unlike the loss/grad-norm
    features -- which look the same for any high-loss example -- the embedding
    encodes *which region of input space* the example lives in, so a scorer can
    learn that one region is learnable and another is unlearnable noise. Returns
    array (B, 5 + emb_dim).
    """
    cheap = reweighter_features(model, cache, y)         # (B, 5)
    content = cache["emb"].mean(axis=1)                  # (B, D)
    mu = content.mean(axis=0, keepdims=True)
    sd = content.std(axis=0, keepdims=True) + 1e-6
    content = (content - mu) / sd
    return np.concatenate([cheap, content], axis=1)      # (B, 5 + D)
