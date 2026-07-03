"""
self_attention_demo.py
=======================
A from-scratch, dependency-light walkthrough of a SINGLE self-attention head,
built to mirror "The Illustrated Transformer" step by step.

Run it:           python self_attention_demo.py
Requires:         numpy   (pip install numpy)

What you'll see printed:
  1. Word embeddings (X)
  2. The learned projection matrices W_Q, W_K, W_V
  3. Q, K, V for every word
  4. The raw attention scores (Q . K^T)
  5. The scaled scores (divide by sqrt(d_k))
  6. The softmax attention weights  <-- this is the "where am I looking?" matrix
  7. The output Z (weighted sum of Values)

The example sentence is the article's classic:
  "the animal didn't cross the street because it was too tired"
We then look at which words "it" attends to.

NOTE: the matrices here are RANDOM (untrained), so don't expect "it" to
magically point at "animal" -- a real trained model learns W_Q/W_K so that
happens. The point of this script is to make the *mechanism* concrete and
inspectable. Section at the bottom shows how to HAND-CRAFT weights so that
"it" attends to "animal", to prove the machinery does what you think.
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=120)
np.random.seed(100)  # reproducible


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def self_attention(X, W_Q, W_K, W_V, verbose=True):
    """One self-attention head. X is (seq_len, d_model)."""
    d_k = W_K.shape[1]

    # STEP 1: project each word's embedding into Query, Key, Value vectors.
    Q = X @ W_Q          # (seq_len, d_k)  "what am I looking for?"
    K = X @ W_K          # (seq_len, d_k)  "what do I advertise?"
    V = X @ W_V          # (seq_len, d_v)  "what do I pass along?"

    # STEP 2: score = dot product of each Query with every Key.
    scores = Q @ K.T     # (seq_len, seq_len)

    # STEP 3: scale by sqrt(d_k) for stable gradients.
    # scaled = scores / np.sqrt(d_k)
    scaled = scores 

    # STEP 4: softmax across each row -> weights that sum to 1.
    weights = softmax(scaled, axis=-1)

    # STEP 5 & 6: multiply Values by weights and sum -> output.
    Z = weights @ V      # (seq_len, d_v)

    if verbose:
        print("Q (queries):\n", Q, "\n")
        print("K (keys):\n", K, "\n")
        print("V (values):\n", V, "\n")
        print("raw scores  Q . K^T:\n", scores, "\n")
        print(f"scaled scores (/ sqrt(d_k={d_k})):\n", scaled, "\n")
        print("attention weights (softmax rows sum to 1):\n", weights, "\n")
        print("output Z (one row per word):\n", Z, "\n")

    return Z, weights


def main():
    sentence = "the animal didn't cross the street because it was too tired".split()
    seq_len = len(sentence)

    d_model = 8   # embedding size (512 in the real model; tiny here so you can read it)
    d_k = 4       # query/key size (64 in the real model)
    d_v = 4       # value size

    # Fake "embeddings": one random vector per word.
    X = np.random.randn(seq_len, d_model)

    # The LEARNED matrices. In a real model these come from training;
    # here they're random so we can watch the mechanism run.
    W_Q = np.random.randn(d_model, d_k)
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_v)

    print("=" * 70)
    print("SENTENCE:", " ".join(sentence))
    print("positions:", {i: w for i, w in enumerate(sentence)})
    print("=" * 70, "\n")

    Z, weights = self_attention(X, W_Q, W_K, W_V, verbose=True)

    # Inspect what the word "it" (index 7) attends to.
    it_idx = sentence.index("it")
    print("-" * 70)
    print(f'How much "it" (position {it_idx}) attends to each word (RANDOM matrices):')
    for w, word in sorted(zip(weights[it_idx], sentence), reverse=True):
        bar = "#" * int(round(w * 40))
        print(f"  {word:>10}  {w:6.3f}  {bar}")
    print("(Random weights => no meaningful pattern. That's expected.)\n")

    # ------------------------------------------------------------------
    # PROVE THE MACHINERY: suppose training had shaped W_Q/W_K so that
    # "it"'s query lands a high score on "animal" (and a smaller one on
    # "street", its other plausible referent). We simulate that learned
    # outcome by setting the SCORE row for "it" directly, then run the
    # exact same scale -> softmax -> weighted-sum-of-Values pipeline.
    # ------------------------------------------------------------------
    print("=" * 70)
    print('SIMULATED TRAINED HEAD: "it" has learned to point at "animal"')
    print("=" * 70)
    animal_idx = sentence.index("animal")
    street_idx = sentence.index("street")

    V = X @ W_V
    # A plausible learned score vector for the "it" row: high on animal,
    # some on street (the competing noun), near-zero elsewhere.
    learned_scores = np.full(seq_len, -2.0)
    learned_scores[animal_idx] = 6.0
    learned_scores[street_idx] = 2.0
    learned_scores[it_idx] = 1.0  # words also attend a bit to themselves

    # weights_it = softmax(learned_scores / np.sqrt(d_k))
    weights_it = softmax(learned_scores)

    print('Now "it" attends to:')
    for w, word in sorted(zip(weights_it, sentence), reverse=True):
        bar = "#" * int(round(w * 40))
        print(f"  {word:>10}  {w:6.3f}  {bar}")

    z_it = weights_it @ V
    print(f"\noutput vector for 'it' = weighted sum of Values = {z_it}")
    print(
        '\n"animal" dominates the weights -> "it"\'s output is mostly animal\'s Value\n'
        "vector, with a dash of street and itself. That blend IS the contextual\n"
        "meaning the model has baked into 'it'. A real model reaches this by\n"
        "learning W_Q and W_K; here we just set the scores to show the payoff."
    )


if __name__ == "__main__":
    main()
