# nanoGPT: From Bigram to a Mini GPT — A Comprehensive Tutorial

> **Author:** AI Teaching Assistant  
> **Target Audience:** ML/DL practitioners mastering LLMs  
> **Objective:** Understand every tensor operation, mathematical concept, and architectural decision in Karpathy's nanoGPT lecture code.

---

## Table of Contents

- [1. Executive Overview](#1-executive-overview)
- [2. Key Concepts & Mathematics](#2-key-concepts--mathematics)
  - [2.1 Character-Level Tokenization](#21-character-level-tokenization)
  - [2.2 The Language Modeling Task](#22-the-language-modeling-task)
  - [2.3 Tensor Shapes — The Universal Language](#23-tensor-shapes--the-universal-language)
  - [2.4 Self-Attention (The Core Innovation)](#24-self-attention-the-core-innovation)
  - [2.5 Multi-Head Attention](#25-multi-head-attention)
  - [2.6 Residual Connections & Layer Normalization](#26-residual-connections--layer-normalization)
  - [2.7 Feed-Forward Network (FFN)](#27-feed-forward-network-ffn)
- [3. Step-by-Step Code Walkthrough](#3-step-by-step-code-walkthrough)
  - [3.1 `bigram.py` — The Simplest Possible Language Model](#31-bigrampy--the-simplest-possible-language-model)
  - [3.2 `gpt.py` — The Full GPT (Decoder-Only Transformer)](#32-gptpy--the-full-gpt-decoder-only-transformer)
- [4. Architecture/Flow Diagram](#4-architectureflow-diagram)
- [5. Summary: Key Takeaways for Fine-Tuning Your Own LLM](#5-summary-key-takeaways-for-fine-tuning-your-own-llm)
- [6. Complete Tensor Shape Walkthrough](#6-complete-tensor-shape-walkthrough)
  - [6.1 Input: Raw Token IDs](#61-input-raw-token-ids)
  - [6.2 Token Embedding Lookup](#62-token-embedding-lookup)
  - [6.3 Position Embedding Lookup](#63-position-embedding-lookup)
  - [6.4 Add Token + Position Embeddings](#64-add-token--position-embeddings)
  - [6.5 Entering a Transformer Block (×6)](#65-entering-a-transformer-block-6)
  - [6.6 Multi-Head Attention — Concatenate All Heads](#66-multi-head-attention--concatenate-all-heads)
  - [6.7 LayerNorm 2](#67-layernorm-2)
  - [6.8 Feed-Forward Network](#68-feed-forward-network)
  - [6.9 Final LayerNorm](#69-final-layernorm)
  - [6.10 LM Head (Output Projection)](#610-lm-head-output-projection)
  - [6.11 Loss Computation (Training Only)](#611-loss-computation-training-only)
  - [6.12 Generation Loop](#612-generation-loop)
  - [6.13 Complete Shape Flowchart](#613-complete-shape-flowchart)
  - [6.14 Summary: Only 3 Operations Change Shapes](#614-summary-only-3-operations-change-shapes)

---

## 1. Executive Overview

This repository contains **two progressively complex language models** trained on Shakespeare's text:

| File | Model | Context window | Params | Architecture |
|------|-------|---------------|--------|--------------|
| `bigram.py` | Bigram Language Model | 8 tokens | ~5K | One embedding table — predicts next token using *only* the current token |
| `gpt.py` | Mini GPT (decoder-only Transformer) | 256 tokens | ~10M | Full Transformer: token embeddings + position embeddings + 6 self-attention blocks + feedforward layers |

Both models are **character-level** — they ingest and predict raw characters (not subwords/words like GPT-2/3). The training data is the complete works of Shakespeare (`input.txt`), providing a rich linguistic distribution for a small model to learn.

**Training outcome:** The bigram model produces gibberish (it only knows "given character X, what's the most likely next character?"). The GPT model generates surprisingly coherent Shakespearean-sounding text, demonstrating the power of self-attention and deep representations.

---

## 2. Key Concepts & Mathematics

### 2.1 Character-Level Tokenization

Before any neural computation, text must be converted to integers.

- **Vocabulary:** All unique characters in the training text (here: ~65 chars including punctuation, spaces, newlines).
- **`encode(s)`:** `str → List[int]` — maps each character to its integer ID via a lookup table `stoi`.
- **`decode(l)`:** `List[int] → str` — reverses via `itos`.

### 2.2 The Language Modeling Task

Given a sequence of tokens `[x₁, x₂, ..., x_T]`, predict the next token `x_{T+1}`. This is **autoregressive** — the model consumes its own previous outputs during generation.

The loss function is **cross-entropy**:

```
L = -∑_t log P(x_{t+1} | x₁, ..., x_t)
```

where the sum runs over all positions in the sequence.

### 2.3 Tensor Shapes — The Universal Language

Throughout the code, tensors follow a consistent shape convention:

```
[B, T, C] = [Batch, Time (sequence length), Channels (embedding dim)]
```

| Symbol | Meaning | Typical value (gpt.py) |
|--------|---------|----------------------|
| `B` | Batch size (independent sequences) | 64 |
| `T` | Block size / context length | 256 |
| `C` | Embedding dimension (`n_embd`) | 384 |

**Every transformation in the network can be understood as reshaping these three dimensions.**

### 2.4 Self-Attention (The Core Innovation)

A single attention head computes:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

Where:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What information do I pass along?"

All three are linear projections of the input `x` of shape `[B, T, C]`.

**The critical tensor operation:**
- `Q @ K^T` gives a `[B, T, T]` matrix of **attention scores** (affinities between every pair of positions).
- Multiplying by `1/√(d_k)` prevents softmax saturation (keeping gradients healthy).
- Masked fill with `-inf` above the diagonal enforces **causality** (token t can only attend to tokens ≤ t).
- `softmax` converts scores to probabilities (rows sum to 1).
- `weights @ V` produces a weighted sum: each position gets a blend of all previous positions' values.

### 2.5 Multi-Head Attention

Instead of one attention distribution, use `h` independent heads, each with head_size = C/h:

```
MultiHead(x) = Concat(head₁(x), ..., head_h(x)) · W_proj
```

Each head can learn different relationship patterns (e.g., syntactic vs. semantic).

### 2.6 Residual Connections & Layer Normalization

- **Residual connection:** `x = x + sublayer(x)` — allows gradients to flow directly through the network, enabling deep stacks (6 layers here).
- **Layer Normalization:** Normalizes across the feature dimension (C) for each token independently. Stabilizes training.

The **Pre-LN** arrangement (norm before sublayer) used here: `x = x + sublayer(norm(x))`.

### 2.7 Feed-Forward Network (FFN)

A simple 2-layer MLP expanding from `C → 4C → C` with ReLU:

```
FFN(x) = Linear(C, 4C) → ReLU → Linear(4C, C)
```

This is where the model "thinks" about the information gathered via attention.

---

## 3. Step-by-Step Code Walkthrough

### 3.1 `bigram.py` — The Simplest Possible Language Model

#### Imports & Hyperparameters

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8    # context: only 8 characters
max_iters = 3000
learning_rate = 1e-2
```

The bigram model has **no notion of context beyond the current token**. `block_size=8` is just the chunk length we train on, but the model doesn't use it for prediction — it only uses position `t` to predict `t+1`.

#### Character Tokenization (shared with gpt.py)

```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)          # ~65 unique characters

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

**Key insight:** This is a **char-level tokenizer**. No BPE, no WordPiece. Every character is a token. This keeps vocabulary small and the model simple.

#### Data Splitting & Batching

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

**Tensor transformation walkthrough:**
1. `ix` shape: `[B]` — random starting indices (e.g., `[4231, 8872, 150, ...]`)
2. `x[i] = data[ix[i]:ix[i]+block_size]` → shape `[block_size]` each → stacked: `[B, T]`
3. `y[i] = data[ix[i]+1:ix[i]+block_size+1]` → **targets shifted right by 1** → shape `[B, T]`

So if `x = [a, b, c, d]`, then `y = [b, c, d, e]`. The model sees `a` and must predict `b`; sees `b` and must predict `c`; etc.

#### BigramLanguageModel

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

This is the **entire model**: a single embedding table of shape `[vocab_size, vocab_size]`.

- Row `i` of the embedding is a `vocab_size`-dimensional vector of **logits** — raw scores for which token comes next given that the current token is `i`.
- There's **no context aggregation**: the prediction for position `t` depends only on `token_t`, not on `token_0 ... token_t`.

```python
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C) where C=vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)          # (B*T, C) — flatten batch & time
            targets = targets.view(B*T)            # (B*T,) — flatten targets
            loss = F.cross_entropy(logits, targets)
        return logits, loss
```

**Cross-entropy detail:**
- `F.cross_entropy` expects input of shape `[N, C]` and targets of shape `[N]`.
- We flatten `[B, T, C] → [B*T, C]` so every position in every sequence contributes to the loss.
- Each position independently computes `-log P(target | input)`.

```python
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)                # forward pass on full sequence
            logits = logits[:, -1, :]               # take only last position: (B, C)
            probs = F.softmax(logits, dim=-1)        # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # sample: (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # append: (B, T+1)
        return idx
```

**Generation loop:**
1. Feed the entire context `idx` through the model.
2. Take only the **last timestep's** logits (we only care about what comes next).
3. `softmax` converts logits to a probability distribution over the vocabulary.
4. `torch.multinomial` samples from this distribution (introducing stochasticity; without it the model would always pick the argmax token).
5. Append the sampled token and repeat.

**Why this doesn't work well:** Each prediction only sees one token of context — the immediately preceding token. There's no mechanism to look further back.

---

### 3.2 `gpt.py` — The Full GPT (Decoder-Only Transformer)

#### Hyperparameters (Scaled Up)

```python
batch_size = 64
block_size = 256       # up from 8
n_embd = 384           # embedding dimension
n_head = 6             # number of attention heads
n_layer = 6            # number of transformer blocks
dropout = 0.2
```

**Why these numbers?** This is roughly a 10M-parameter model, scaled down from GPT-2 (which had 12 layers, 768-dim embeddings, 12 heads). The ratios are preserved: `n_embd / n_head = 64` (head size).

#### Single Attention Head

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
```

- **Q, K, V projections:** Three separate learned linear transforms, each mapping `[B, T, C] → [B, T, head_size]`.
- **`register_buffer('tril', ...)`:** A lower-triangular mask matrix of 1s (below diagonal) and 0s (above). Registered as a buffer so it moves to GPU with the model but is not a parameter.

```python
    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)
```

**The critical tensor operation** `q @ k.transpose(-2, -1)`:
- `q` shape: `[B, T, hs]` — for each batch, each position has a query vector
- `k.transpose(-2, -1)` shape: `[B, hs, T]` — transposes the last two dims
- Result: `[B, T, T]` — the **attention matrix** where entry `[b, i, j]` = similarity between query at position `i` and key at position `j`

**Scaling** `* hs**-0.5`: The dot products grow with `hs` (variance ≈ hs). Dividing by `√hs` keeps the variance ≈ 1, preventing softmax from becoming a near-one-hot distribution.

```python
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        out = wei @ v  # (B, T, hs)
```

**Causal masking:** Entries above the diagonal are set to `-inf`, so `softmax` outputs 0 for those positions. This ensures position `t` can only attend to positions `≤ t`.

**Weighted aggregation:** `wei @ v` where:
- `wei` shape: `[B, T, T]` — attention probabilities (each row sums to 1)
- `v` shape: `[B, T, hs]` — values
- Result: `[B, T, hs]` — each output position is a weighted sum of previous values

#### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads * hs) = (B, T, C)
        out = self.dropout(self.proj(out))                     # (B, T, C) — projection back to n_embd
        return out
```

With `n_head=6` and `n_embd=384`, each head has `head_size = 384/6 = 64`.

- 6 independent heads run in parallel (conceptually; sequentially in code but with different learned Q/K/V weights).
- Outputs are concatenated: `[B, T, 64] × 6 → [B, T, 384]`.
- `self.proj` projects back to `n_embd` — this is a learned mixing of head outputs.

#### Feed-Forward Network

```python
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # expand: 384 → 1536
            nn.ReLU(),                         # non-linearity
            nn.Linear(4 * n_embd, n_embd),    # compress back: 1536 → 384
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

**Why the expansion factor of 4?** This is the standard Transformer FFN ratio. The ReLU activation introduces sparsity: roughly half the activations are exactly 0, creating a high-dimensional sparse representation where the model can "store" patterns.

#### Transformer Block

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # attention + residual
        x = x + self.ffwd(self.ln2(x))  # feed-forward + residual
        return x
```

Each block has two sub-layers, each wrapped in a **Pre-LN residual connection**:

1. **Self-Attention sub-layer:** `LayerNorm → MultiHeadAttention → + residual`
   - Purpose: **Communication** between tokens. Each token gathers information from all previous tokens.
   
2. **FFN sub-layer:** `LayerNorm → FeedForward → + residual`
   - Purpose: **Computation** on each token independently. The token "thinks" about the information it gathered.

**Why residual connections?** The gradient flows through the identity path (the `+ x`), allowing training of deep networks. Without residuals, 6 layers of attention would suffer from vanishing gradients.

**Why Pre-LN (norm before sub-layer)?** More stable training than Post-LN (original Transformer). Allows higher learning rates.

#### Full GPT Language Model

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
```

**Token Embedding:** `[vocab_size, n_embd]` — maps each character ID to a 384-dimensional vector. This is learned from scratch.

**Position Embedding:** `[block_size, n_embd]` — maps each position (0 to 255) to a 384-dimensional vector. Makes the model position-aware; without this, attention is permutation-invariant.

**Why learned position embeddings instead of sinusoidal?** Both work. Learned embeddings are simpler and perform similarly for moderate context lengths.

```python
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)              # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb                                  # (B, T, C) — broadcast

        x = self.blocks(x)                                     # (B, T, C)
        x = self.ln_f(x)                                       # (B, T, C)
        logits = self.lm_head(x)                               # (B, T, vocab_size)
```

**Token + Position:** `broadcast add` — `pos_emb` is `[T, C]` which broadcasts against `[B, T, C]`. Each position gets the same position embedding across all batch items.

**Final layer norm:** A learnable normalization before the output head.

**LM Head:** A linear projection from `n_embd` to `vocab_size` producing logits for each position.

```python
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)          # (B*T, vocab_size)
            targets = targets.view(B*T)           # (B*T,)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
```

Same cross-entropy calculation as the bigram model.

#### Weight Initialization

```python
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

Small normal initialization (std=0.02) is critical for Transformer training. Too-large initial weights cause attention softmax to saturate, and gradients vanish.

#### Generation (Improved)

```python
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]       # crop to last block_size tokens
            logits, loss = self(idx_cond)          # forward pass
            logits = logits[:, -1, :]              # take last position: (B, C)
            probs = F.softmax(logits, dim=-1)      # to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
```

**Crucial difference from bigram:** `idx_cond = idx[:, -block_size:]`. Since the model can attend up to 256 tokens back, we must crop the context to its maximum capacity. Feeding more than `block_size` tokens would exceed the position embedding table.

#### Training Loop

```python
model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # 3e-4
```

**AdamW hyperparameter note:** `lr=3e-4` is a standard starting point for Transformer training. Higher rates cause instability; lower rates converge too slowly.

```python
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

Standard PyTorch training loop. `estimate_loss()` averages loss over `eval_iters=200` batches for more stable evaluation.

#### Loss Estimation

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

- `@torch.no_grad()`: Disables gradient computation for efficiency.
- `model.eval() / model.train()`: Controls dropout behavior (dropout disabled during eval).
- Averaging over 200 batches gives a stable loss estimate.

---

## 4. Architecture/Flow Diagram

### Data Flow Through the GPT Model

```
Input: (B, T) tensor of token IDs
         │
         ▼
┌─────────────────────────┐
│  Token Embedding        │  shape (B, T) → (B, T, C=384)
│  Position Embedding     │  shape (T,) → (T, C)  [broadcast]
│          +              │
│  x = tok_emb + pos_emb  │  shape (B, T, C)
└─────────┬───────────────┘
          │
          ▼      × 6 (n_layer)
┌─────────────────────────┐
│    Transformer Block    │
│                         │
│   ┌─────────────────┐   │
│   │  LayerNorm      │   │  (B, T, C)
│   │  MultiHeadAttn  │   │  (B, T, C)
│   │  ┌ ─ ─ ─ ─ ─ ┐ │   │
│   │  │ Head 1     │ │   │  head_size=64
│   │  │ Head 2     │ │   │
│   │  │   ...      │ │   │
│   │  │ Head 6     │ │   │
│   │  └ ─ ─ ─ ─ ─ ┘ │   │
│   │  Concat + Proj  │   │  (B, T, 384)
│   │  + Dropout      │   │
│   │  + Residual     │   │  (B, T, C)
│   └─────────────────┘   │
│          +               │
│   ┌─────────────────┐   │
│   │  LayerNorm      │   │  (B, T, C)
│   │  Linear 384→1536│   │  (B, T, 1536)
│   │  ReLU           │   │  (B, T, 1536)  [sparse!]
│   │  Linear 1536→384│   │  (B, T, C)
│   │  Dropout        │   │
│   │  + Residual     │   │  (B, T, C)
│   └─────────────────┘   │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Final LayerNorm        │  (B, T, C)
│  LM Head (Linear)       │  (B, T, vocab_size ~65)
└─────────┬───────────────┘
          │
          ▼
     Logits: (B, T, vocab_size)
          │
          ▼
  ┌─── targets=None? ───┐
  │         │            │
  │        Yes           │ No
  │         │            │
  │      return          │ CrossEntropyLoss
  │   (logits, None)     │ flatten (B*T, C) vs (B*T,)
  │                      │
  │                  return (logits, loss)
  └─────────────────────────────────────┘
```

### Attention Head Detail

```
Input x: (B, T, C=384)
       │
       ├──→ Linear(C, hs=64) → K: (B, T, 64)
       ├──→ Linear(C, hs=64) → Q: (B, T, 64)
       └──→ Linear(C, hs=64) → V: (B, T, 64)

wei = Q @ K^T                : (B, T, 64) @ (B, 64, T) → (B, T, T)
wei = wei / √64              : scaling
wei = masked_fill(tril==0, -inf) : causality
wei = softmax(wei, dim=-1)   : (B, T, T)  [each row sums to 1]
wei = dropout(wei)

out = wei @ V                : (B, T, T) @ (B, T, 64) → (B, T, 64)
```

### How the Bigram Model Differs

```
Input: (B, T)
    │
    ▼
Embedding(vocab_size, vocab_size)  : (B, T) → (B, T, vocab_size)
    │
    ▼
Logits: (B, T, vocab_size)
    │
No context mixing! Each position's prediction is independent.
```

---

## 5. Summary: Key Takeaways for Fine-Tuning Your Own LLM

1. **Tokenization matters:** Character-level is simple but inefficient. Modern LLMs use BPE or SentencePiece (subword tokens). For fine-tuning, you'll typically use the pre-trained model's tokenizer.

2. **Context window is king:** `block_size` determines how far back the model can look. GPT-3 uses 2048 tokens; GPT-4 uses 32K-128K. For fine-tuning, match your task's context needs.

3. **Self-attention is quadratic:** The `[T, T]` attention matrix means compute grows as `O(T²)`. This is the main bottleneck for long sequences.

4. **Pre-training vs. Fine-tuning:** The code here performs **pre-training from scratch** (learning language from raw text). Fine-tuning takes a pre-trained model and trains it further on a specific dataset/task with a much smaller learning rate (1e-5 vs 3e-4).

5. **What you'd change for fine-tuning:**
   - Load pre-trained weights instead of random init.
   - Use a lower learning rate (1e-5 to 5e-5).
   - Usually freeze early layers or use LoRA (Low-Rank Adaptation).
   - Keep the same tokenizer and vocabulary.
   - Adjust `block_size` to match the base model's context window.

6. **The magic of scaling:** This 10M-parameter model shows glimmers of coherent text. GPT-3 has 175B parameters — ~17,500× more. Emergent abilities (translation, reasoning, code generation) appear at scale.

---

*Happy hacking! You now have a deep understanding of every line in nanoGPT. The jump from here to fine-tuning a real LLM (like LLaMA or GPT-2) is a matter of scale, data, and tooling — the core architecture is the same.*

---

## 6. Complete Tensor Shape Walkthrough

*This section traces every single tensor shape change through the full forward pass of `GPTLanguageModel`. It's the most detailed view of how data actually flows."*

### Setup: The Numbers

```python
batch_size = 64      # B
block_size = 256     # T
n_embd = 384         # C
n_head = 6
head_size = 64       # 384 / 6
vocab_size = 65
n_layer = 6
```

### 6.1 Input: Raw Token IDs

```python
idx = torch.randint(0, vocab_size, (64, 256))  # random example
# idx.shape = (64, 256) = (B, T)
```

Just a matrix of integers (character IDs 0–64).

---

### 6.2 Token Embedding Lookup

```python
self.token_embedding_table = nn.Embedding(vocab_size=65, n_embd=384)
# weight.shape = (65, 384)  — one 384-dim vector per character

tok_emb = self.token_embedding_table(idx)
# idx.shape      = (64, 256)
# tok_emb.shape = (64, 256, 384) = (B, T, C)
```

Each integer ID is replaced by its corresponding 384-dim vector from the lookup table. Think of it as: "character 'h' (ID 46) → its learned semantic vector."

---

### 6.3 Position Embedding Lookup

```python
self.position_embedding_table = nn.Embedding(block_size=256, n_embd=384)
# weight.shape = (256, 384)  — one 384-dim vector per position

pos_emb = self.position_embedding_table(torch.arange(256))
# torch.arange(256).shape = (256,)
# pos_emb.shape = (256, 384) = (T, C)
```

Note: no batch dimension yet. The same position 0 vector is used for position 0 in *every* sequence.

---

### 6.4 Add Token + Position Embeddings

```python
x = tok_emb + pos_emb
# (64, 256, 384) + (256, 384)
# Broadcasting: (256, 384) → (64, 256, 384) automatically
# x.shape = (64, 256, 384) = (B, T, C)
```

Each token now carries **two** pieces of information: "what token am I?" + "where am I in the sequence?" The subsequent layers learn to decode these combined signals.

---

### 6.5 Entering a Transformer Block (×6)

Each block processes `x` of shape `(64, 256, 384)`, preserves the shape, and passes it to the next block.

#### 6.5a LayerNorm 1

```python
self.ln1 = nn.LayerNorm(384)  # normalizes the last dimension

x_normed = self.ln1(x)
# x.shape = (64, 256, 384) → x_normed.shape = (64, 256, 384)
```

**No shape change.** Each of the 384 features is independently normalized (mean=0, std=1) for every position in every sequence. This stabilizes training.

#### 6.5b Multi-Head Attention — Inside ONE Head

```python
class Head(nn.Module):
    def __init__(self, head_size=64):
        self.key   = nn.Linear(384, 64, bias=False)   # weight: (64, 384)
        self.query = nn.Linear(384, 64, bias=False)   # weight: (64, 384)
        self.value = nn.Linear(384, 64, bias=False)   # weight: (64, 384)
```

##### Linear Projections: Q, K, V

```python
k = self.key(x_normed)
# x_normed.shape = (64, 256, 384)
# Linear(384→64) operates on last dim: weight (64,384) @ input (384,) → (64,)
# k.shape = (64, 256, 64) = (B, T, head_size)

q = self.query(x_normed)
# q.shape = (64, 256, 64)

v = self.value(x_normed)
# v.shape = (64, 256, 64)
```

**Matrix multiplication view** (for one position):
```
k[b, t, :] = W_key @ x_normed[b, t, :]    # 64×384 @ 384×1 → 64×1
```

The position is now represented as a **query** (what am I looking for?), a **key** (what do I contain?), and a **value** (what info do I pass along?).

##### Attention Scores: Q @ K^T

```python
wei = q @ k.transpose(-2, -1)
# q.shape             = (64, 256, 64)
# k.transpose(-2,-1)  = (64, 64, 256)   ← swapped last two dimensions
# wei.shape           = (64, 256, 256)   = (B, T, T)
```

**What this multiplication does:**

```
For batch b:
               K^T (64×256)
          ┌──────────────────────┐
          │ k₀₀  k₁₀  ...  k₂₅₅₀│
          │ k₀₁  k₁₁  ...  k₂₅₅₁│
          │  ...                  │
          │ k₀₆₃ k₁₆₃ ... k₂₅₅₆₃│
          └──────────────────────┘
    ┌──┐  ┌──────────────────────┐
Q   │q₀─┼──→ wei[0,:] = q₀·k₀ ... q₀·k₂₅₅ │  ← similarity between
(256×64)│q₁─┼──→ wei[1,:] = q₁·k₀ ... q₁·k₂₅₅ │     pos 0 and ALL positions
    │  …│  │  …                                 │
    │q₂₅₅┼──→ wei[255,:] = q₂₅₅·k₀ ...        │  ← pos 255 and ALL positions
    └──┘  └──────────────────────────────────────┘
```

Entry `wei[b, i, j]` = **how much should position `i` attend to position `j`?**

##### Scaling

```python
wei = wei * k.shape[-1]**-0.5
# k.shape[-1] = 64
# 64^-0.5 = 1/8 = 0.125
# wei.shape = (64, 256, 256) — same shape, values scaled down
```

Each dot product sums 64 products (variance ≈ 64). Dividing by √64 = 8 brings variance back to ~1, preventing softmax from becoming near-one-hot (which would cause vanishing gradients).

##### Causal Masking

```python
self.register_buffer('tril', torch.tril(torch.ones(256, 256)))
# tril.shape = (256, 256) — lower triangular: 1s on and below diagonal, 0s above

wei = wei.masked_fill(self.tril[:256, :256] == 0, float('-inf'))
# wei.shape = (64, 256, 256) — same shape
```

**Before masking:** `wei[0, 3, 7]` = some positive number (position 3 can see the future! cheating!)

**After masking:** `wei[0, 3, 7]` = `-inf` → softmax gives it probability 0.

The `tril` matrix visualized:
```
Position:  0  1  2  3  4  5  ...
        0  1  0  0  0  0  0  ...   ← pos 0: attends only to pos 0
        1  1  1  0  0  0  0  ...   ← pos 1: attends to pos 0, 1
        2  1  1  1  0  0  0  ...   ← pos 2: attends to pos 0, 1, 2
        3  1  1  1  1  0  0  ...   ← pos 3: attends to pos 0, 1, 2, 3
        ...
        255 1  1  1  1  1  1  ...  ← pos 255: attends to ALL previous
```

##### Softmax

```python
wei = F.softmax(wei, dim=-1)
# wei.shape = (64, 256, 256) — same shape, now each row sums to exactly 1
```

For each batch item and each position, softmax normalizes across the last dimension (all T positions):
```
wei[0, 3, :] = softmax([-inf, -inf, -inf, 0.5, -inf, -inf, ...])
             = [0, 0, 0, 1, 0, 0, ...]   ← all probability on positions ≤ 3
```

##### Dropout on Attention Weights

```python
wei = self.dropout(wei)
# wei.shape = (64, 256, 256) — randomly zeros ~20% of entries
```

Forces the model to not rely on any single attention path. Each training step, some token pairs are forcibly ignored.

##### Weighted Aggregation: wei @ V

```python
out = wei @ v
# wei.shape = (64, 256, 256)
# v.shape   = (64, 256, 64)
# out.shape = (64, 256, 64) = (B, T, head_size)
```

Each output position is a **weighted sum** of **all previous positions' value vectors**:
```
out[b, t, :] = wei[b, t, 0]·v[b, 0, :] + wei[b, t, 1]·v[b, 1, :] + ...
           + wei[b, t, t]·v[b, t, :]
```

Positions that get high attention weight contribute more to the output; positions with zero attention (masked future) contribute nothing.

---

### 6.6 Multi-Head Attention — Concatenate All Heads

```python
# 6 heads, each producing (64, 256, 64)
head_outputs = [h(x) for h in self.heads]  # 6 × (64, 256, 64)

out = torch.cat(head_outputs, dim=-1)
# out.shape = (64, 256, 384) = (B, T, 6 × 64) = (B, T, C)
```

**Concatenation along the last dimension:** 6 independent 64-dim views → one 384-dim combined view. Each head captured a different relationship pattern (syntax, position, semantics, etc.), now they're all together.

#### Projection Back to n_embd

```python
self.proj = nn.Linear(384, 384)  # mixes information across heads
out = self.proj(out)
# out.shape = (64, 256, 384) — same shape, but now cross-head mixing
```

Despite input and output having the same shape, this is a **learned weighted combination** of the 6 heads' outputs. The model decides how to blend the specialized patterns.

#### Dropout + Residual Connection

```python
out = self.dropout(out)
# out.shape = (64, 256, 384)

x = x + out   # residual connection
# x.shape = (64, 256, 384)
```

The residual connection creates a **gradient highway** — gradients can flow directly through the `+ x` path, bypassing the attention layer. This is what allows training 6+ layers deep without vanishing gradients.

---

### 6.7 LayerNorm 2

```python
x_normed = self.ln2(x)
# (64, 256, 384) → (64, 256, 384)
```

---

### 6.8 Feed-Forward Network

```python
class FeedFoward(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(384, 1536),    #  384 → 1536 (4× expansion)
            nn.ReLU(),               #  no shape change
            nn.Linear(1536, 384),    # 1536 → 384  (back to C)
            nn.Dropout(0.2),         #  no shape change
        )

out = self.ffwd(x_normed)
# x_normed.shape = (64, 256, 384)
# After Linear(384, 1536):  (64, 256, 1536)   ← EXPAND   
# After ReLU:               (64, 256, 1536)   ← sparsity: ~half become 0
# After Linear(1536, 384):  (64, 256, 384)    ← COMPRESS
# After Dropout:            (64, 256, 384)
# out.shape = (64, 256, 384)
```

**Why expand to 1536 then compress back?** The expansion to 4× creates a **high-dimensional scratchpad** where the model can store and transform patterns. ReLU introduces sparsity (many activations become exactly 0), which acts as a learned feature selection mechanism.

#### Second Residual Connection

```python
x = x + out
# x.shape = (64, 256, 384)
```

**End of one Block.** After 6 blocks, `x` is still `(64, 256, 384)`. The shape never changes through the entire transformer stack.

---

### 6.9 Final LayerNorm

```python
self.ln_f = nn.LayerNorm(384)
x = self.ln_f(x)
# (64, 256, 384) → (64, 256, 384)
```

---

### 6.10 LM Head (Output Projection)

```python
self.lm_head = nn.Linear(384, 65)   # n_embd → vocab_size

logits = self.lm_head(x)
# x.shape      = (64, 256, 384)
# logits.shape = (64, 256, 65) = (B, T, vocab_size)
```

Each of the 256 positions now has 65 raw scores — one for each possible next character. The model scores "what character is most likely to come next, given all the context so far?"

---

### 6.11 Loss Computation (Training Only)

```python
B, T, C = logits.shape     # (64, 256, 65)
logits = logits.view(B*T, C)    # (16384, 65)   ← flatten batch × time
# Each of 64×256=16384 positions is now an independent prediction example

targets = targets.view(B*T)     # (16384,)      ← flatten ground truth

loss = F.cross_entropy(logits, targets)
# For each position: -log(P(correct_next_char))
# Average across all 16384 positions → single scalar
```

---

### 6.12 Generation Loop

```python
# Start with: context = torch.zeros((1, 1), dtype=torch.long)
# context.shape = (1, 1) = (B=1, T=1)

for _ in range(500):  # generate 500 new tokens
    idx_cond = idx[:, -256:]   # (1, min(current_len, 256))
    # Crop to last 256 tokens (model's context limit)

    logits, _ = self(idx_cond)  # forward pass
    # logits.shape = (1, current_T, 65)

    logits = logits[:, -1, :]   # (1, 65) — only the LAST position
    # We only care about what comes NEXT, not what we already generated

    probs = F.softmax(logits, dim=-1)        # (1, 65) — probability distribution
    idx_next = torch.multinomial(probs, 1)   # (1, 1)  — sample one token

    idx = torch.cat((idx, idx_next), dim=1)  # (1, T+1) — append to sequence
```

---

### 6.13 Complete Shape Flowchart

```
Input tokens idx:                    (B=64, T=256)
    │
    ▼
Token Embedding:                     (64, 256, 384)
Position Embedding:                  (256, 384) → broadcast
    │
    ▼
x = tok_emb + pos_emb:              (64, 256, 384)
    │
    ▼  [×6 Blocks]
┌──────────────────────────────────────────────────────────────────┐
│  LayerNorm:                          (64, 256, 384)              │
│  ┌─ Multi-Head Attention ────────────────────────────────────┐  │
│  │  Q: Linear(384→64)  →  (64,256,64)   For each of 6 heads  │  │
│  │  K: Linear(384→64)  →  (64,256,64)                        │  │
│  │  V: Linear(384→64)  →  (64,256,64)                        │  │
│  │  Q @ K^T: (64,256,64) @ (64,64,256) → (64,256,256)       │  │
│  │  Scale × 1/√64:                     (64,256,256)           │  │
│  │  Masked fill (tril):                (64,256,256)           │  │
│  │  Softmax (dim=-1):                  (64,256,256)           │  │
│  │  Dropout:                           (64,256,256)           │  │
│  │  wei @ V: (64,256,256) @ (64,256,64) → (64,256,64)        │  │
│  │  [Repeat for all 6 heads]                                  │  │
│  │  Concat heads (dim=-1):             (64,256,384)           │  │
│  │  Proj Linear(384→384):              (64,256,384)           │  │
│  │  Dropout:                           (64,256,384)           │  │
│  └────────────────────────────────────────────────────────────┘  │
│  + Residual:                          (64,256,384)              │
│  LayerNorm:                            (64,256,384)              │
│  ── Feed-Forward ──                                              │
│  Linear(384→1536):                     (64,256,1536)             │
│  ReLU:                                 (64,256,1536)             │
│  Linear(1536→384):                     (64,256,384)              │
│  Dropout:                              (64,256,384)              │
│  + Residual:                           (64,256,384)              │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
Final LayerNorm:                     (64, 256, 384)
    │
    ▼
LM Head Linear(384→65):             (64, 256, 65)   ← LOGITS
    │
    ▼
CrossEntropyLoss: flatten (16384, 65) vs (16384,) → scalar loss
```

### 6.14 Summary: Only 3 Operations Change Shapes

| Operation | What it does | Shape change example |
|-----------|-------------|---------------------|
| **Linear / Embedding** | Matrix multiply on last dim | `(64,256,384)` → `(64,256,64)` or `(64,256,65)` |
| **Q @ K^T** | Batched matrix multiply | `(64,256,64) @ (64,64,256)` → `(64,256,256)` |
| **Concat** | Stack along last dim | 6× `(64,256,64)` → `(64,256,384)` |
| **View/Reshape** | Flatten batch & time | `(64,256,65)` → `(16384,65)` |

Everything else (LayerNorm, ReLU, softmax, dropout, masking, residual add, scaling) **preserves the shape** — it only transforms values.
