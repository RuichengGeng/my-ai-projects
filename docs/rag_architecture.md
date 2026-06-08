# RAG System — Architecture & Retrieval Quality Roadmap

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [Current Pipeline (as built)](#2-current-pipeline-as-built)
3. [Where the current pipeline falls short](#3-where-the-current-pipeline-falls-short)
4. [P2 — Retrieval Quality Improvements](#4-p2--retrieval-quality-improvements)
   - 4.1 [Confidence Threshold](#41-confidence-threshold)
   - 4.2 [Reranking with a Cross-Encoder](#42-reranking-with-a-cross-encoder)
   - 4.3 [Hybrid Search — BM25 + Semantic](#43-hybrid-search--bm25--semantic)
5. [Combined P2 pipeline](#5-combined-p2-pipeline)
6. [Expected impact on eval metrics](#6-expected-impact-on-eval-metrics)

---

## 1. What is RAG?

Retrieval-Augmented Generation (RAG) is a pattern for answering questions
from a private document corpus. The idea is simple: instead of relying on
what an LLM memorised during training, we **retrieve** the relevant passages
first, then **give them to the LLM** as context so it can produce a grounded
answer.

Without RAG the LLM can only hallucinate or say "I don't know" for anything
outside its training data. With RAG it becomes a reading-comprehension engine
over your documents.

---

## 2. Current Pipeline (as built)

### 2.1 Indexing (offline, run once per new document)

```
PDF file
   │
   ▼
MinerU cloud API  ──►  Markdown file  (text + image paths)
   │
   ▼
Split by H2 section headers
   │
   ▼  for each chunk:
Sentence-Transformer embedding  (BAAI/bge-base-en-v1.5)
   │
   ▼
ChromaDB  ─── stores: chunk text, embedding vector, metadata
              metadata: doc_name, section, topic, series_name, series_date
              manifest.json: tracks what is indexed + series/topic info
```

**Key design choice:** we chunk by H2 headers rather than a fixed character
count. Financial research PDFs are naturally structured this way (each section
is a coherent topic), so the chunks align with meaning rather than arbitrary
byte boundaries.

---

### 2.2 Querying (the general RAG agent)

```
User question
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  PLAN node                                           │
│  • Reads manifest to get known topics + date ranges  │
│  • Asks LLM: which topic? latest / date_range / all? │
│  • Produces: topic_filter, series_filter,            │
│              doc_filter, date_from / date_to         │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  RETRIEVE node                                       │
│  • Embeds the query with bge-base-en-v1.5            │
│  • Runs cosine similarity search in ChromaDB         │
│  • Applies where-filter (topic / series / doc / $in) │
│  • Returns top-N chunks (default N=5)                │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  GENERATE node                                       │
│  • Concatenates chunks as context string             │
│  • Prompts DeepSeek: answer from context only,       │
│    cite document/section for each claim              │
│  • Produces a draft answer                           │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  EVALUATE node                                       │
│  • Asks DeepSeek: is this answer complete?           │
│  • If incomplete: proposes a refined query           │
│  • Loop back to RETRIEVE (up to max_iterations=3)    │
│  • If complete: return final answer                  │
└──────────────────────────────────────────────────────┘
     │
     ▼
Final answer + token usage summary
```

### 2.3 Oil Market Skill (structured output variant)

Instead of a single query, the `OilMarketSummaryAgent` runs **six** sub-queries
in sequence — one per product category (crude, heavy, middle distillates, light
ends, arbitrage, fundamentals) — deduplicates the chunks, then generates a
structured markdown report with fixed sections.

---

## 3. Where the current pipeline falls short

To understand why we need P2, it helps to understand how embedding-based
similarity search works and where it breaks.

### 3.1 How embedding search works (and why it is not enough)

The sentence transformer converts every chunk and every query into a vector of
~768 numbers. Semantically similar text produces vectors that point in a
similar direction; cosine similarity measures that angle.

**The model is trained on sentence pairs.** It learns what "similar meaning"
looks like. The result is that it works well for paraphrase and topic matching,
but it is fundamentally a **coarse filter**. It retrieves candidates that are
*about the same topic* as the query — it does not ask "is this the best passage
to answer this specific question?"

### 3.2 Three concrete failure modes we observe

| Failure mode | Example | Why it happens |
|---|---|---|
| **Low-quality chunks make it through** | A chunk about "crude oil market history" comes back for a question about "current HSFO crack spread" | Embedding similarity is high (both are about oil) but the chunk is useless for this specific question |
| **Exact terms are missed** | Query: "ESPO crude vs Brent spread this week" — chunks mentioning "ESPO" as a passing reference score lower than generic crude market commentary | The transformer weights semantic meaning; it cannot do exact term matching |
| **Top chunk is generic, best chunk is rank 8** | A table with exact price data is at rank 8 because its text is sparse; a long narrative is rank 1 | Embedding search favours text length and semantic richness, not answer quality |

These failures show up directly in our eval metrics:

- **Context Precision < 1.0** — some of the top-N retrieved chunks are not
  actually useful (failure mode 1)
- **Faithfulness < 5** — the LLM sometimes drifts because it is looking at
  noisy context (failure mode 1)
- **Completeness < 5** — key numeric data is buried at rank 8 and never reaches
  the context window (failure mode 3)

---

## 4. P2 — Retrieval Quality Improvements

The three improvements address the three failure modes in order of
implementation complexity.

---

### 4.1 Confidence Threshold

**What:** After embedding retrieval, drop any chunk whose cosine similarity
score is below a minimum threshold (proposed default: `0.45`).

**Why it matters:**

Cosine similarity ranges from 0 (completely unrelated) to 1 (identical). In
practice, a score below ~0.45 means the chunk is topically adjacent at best —
it shares vocabulary with the query but likely does not contain the answer. Yet
currently these chunks are passed to the LLM anyway, adding noise.

```
Current flow:
  query ──► retrieve top-5 ──► all 5 go to LLM context
                                (even if score 3 = 0.31, score 4 = 0.28, score 5 = 0.22)

With threshold:
  query ──► retrieve top-5 ──► filter: score >= 0.45 ──► 2 chunks to LLM
                                                          (plus a clear log that 3 were dropped)
```

**Trade-off:** If the threshold is too high, you might drop relevant chunks and
produce incomplete answers. The right value is empirical — we will tune it
against the eval metrics. For financial docs with specialised vocabulary, 0.45
is a reasonable starting point.

**Impact on eval metrics:** Direct improvement to **Context Precision** and
**Faithfulness** by reducing noise sent to the LLM.

---

### 4.2 Reranking with a Cross-Encoder

**What:** After the bi-encoder (bge-base-en-v1.5) retrieves the top-N
candidates, pass each `(question, chunk)` pair through a **cross-encoder**
model to get a more accurate relevance score. Re-sort chunks by this new score
before sending to the LLM.

**Why bi-encoder + cross-encoder is the standard architecture:**

| | Bi-encoder | Cross-encoder |
|---|---|---|
| What it does | Encodes query and chunk **independently** | Encodes query and chunk **together** |
| Speed | Fast — embed once, compare with dot product | Slow — full transformer inference per pair |
| Quality | Coarse: good for broad recall | Fine: much more accurate relevance |
| Typical use | First-stage retrieval over millions of chunks | Second-stage reranking of top-20 candidates |

The bi-encoder is efficient enough to scan all 10,000+ chunks in the database.
The cross-encoder would be too slow for that — but it only needs to score the
top-20 candidates the bi-encoder already found, which takes < 1 second.

```
Current flow:
  query ──► bi-encoder similarity ──► top-5 by embedding score ──► LLM

With reranking:
  query ──► bi-encoder similarity ──► top-20 candidates
                │
                ▼
          cross-encoder scores each (question, chunk) pair
                │
                ▼
          re-sorted: top-5 by cross-encoder score ──► LLM
```

**Model:** `cross-encoder/ms-marco-MiniLM-L-12-v2` — a 33M parameter model
trained on MS MARCO passage ranking. `sentence-transformers` (already in our
dependencies) ships this out of the box.

**Why this fixes failure mode 3:** The cross-encoder reads the full question and
chunk together. It can recognise that a sparse table with exact price numbers
directly answers the question even if its embedding score was mediocre.

**Impact on eval metrics:** The most direct improvement to **Completeness** and
**Context Precision**. Expect the cross-encoder to surface the price tables and
specific data points that embedding search pushes to rank 8+.

---

### 4.3 Hybrid Search — BM25 + Semantic

**What:** Run a BM25 keyword index in parallel with the embedding search.
Merge the two ranked lists using Reciprocal Rank Fusion (RRF) before
passing to the reranker.

**Why embedding search alone misses exact terms:**

Embedding models are trained to capture *meaning*, not to match exact strings.
"ESPO crude" and "Russian Far East blend" might have similar embeddings because
they mean the same thing — but if your question is specifically about "ESPO"
and the document uses that exact term, BM25 will find it more reliably.

BM25 (Best Match 25) is the algorithm behind most traditional search engines
(Elasticsearch, Solr). It scores documents by how often the query terms appear,
weighted by how rare those terms are across the corpus (TF-IDF variant). It has
no notion of meaning — it is purely lexical — but for exact financial
terminology it is extremely reliable.

```
Hybrid flow:

                    ┌─────────────────────┐
                    │  User query          │
                    └──────────┬──────────┘
                               │
               ┌───────────────┴────────────────┐
               │                                │
               ▼                                ▼
    Semantic search (bi-encoder)      BM25 keyword search
    top-20 by cosine similarity       top-20 by TF-IDF score
               │                                │
               └───────────────┬────────────────┘
                               │
                               ▼
               Reciprocal Rank Fusion (RRF)
               score = 1/(k + rank_semantic) + 1/(k + rank_bm25)
               k = 60  (standard constant, dampens high ranks)
                               │
                               ▼
               Merged top-20 unique chunks
                               │
                               ▼
               Cross-encoder reranker  (from 4.2)
                               │
                               ▼
               Top-5 by cross-encoder score ──► LLM
```

**Why RRF instead of score averaging:**

The embedding score and BM25 score live on different scales. RRF avoids this
by using only the **rank** (position in sorted list), not the raw score. A
chunk ranked #1 by BM25 and #5 by semantic gets a strong combined score
regardless of whether BM25 gave it a score of 12.4 or 0.87.

**Impact on eval metrics:** Improvement to **Context Precision** for
domain-specific queries. For oil market documents specifically, exact product
names (HSFO, VLSFO, 380cst, ESPO, Murban, UDC), date references, and
numerical values are all better served by BM25.

**Implementation note:** BM25 indexes only the text already in ChromaDB — there
is no new storage needed. The `rank_bm25` library provides a pure-Python BM25
implementation. The index is built in memory at startup (~1 second for 10K
chunks) or cached on disk.

---

## 5. Combined P2 Pipeline

After all three improvements, the full retrieval path looks like this:

```
User question
     │
     ▼
[PLAN] topic / date detection  (unchanged)
     │
     ▼
[RETRIEVE — stage 1: recall]
  ├─ Semantic search via bge-base-en-v1.5   → top-20 candidates
  └─ BM25 keyword search                    → top-20 candidates
            │
            ▼
     Reciprocal Rank Fusion → merged top-20 unique chunks
            │
            ▼
   Confidence threshold filter  (drop score < 0.45)
            │
            ▼
[RETRIEVE — stage 2: precision]
  Cross-encoder reranker on (question, chunk) pairs
            │
            ▼
   Top-5 by cross-encoder score
            │
            ▼
[GENERATE]  same as today
            │
            ▼
[EVALUATE]  same as today
```

The changes are entirely inside the retrieve step. The generate/evaluate loop,
topic/date filtering, and token tracking are all untouched.

---

## 6. Expected Impact on Eval Metrics

| Metric | Current behaviour | After confidence threshold | After + reranking | After + BM25 |
|---|---|---|---|---|
| **Context Precision** | Noisy top-5 | Fewer, better chunks | Strong improvement | Catches exact-term misses |
| **Faithfulness** | Occasional drift from irrelevant context | Improves (less noise) | Steady | Steady |
| **Completeness** | Price tables often missed | Partial improvement | Strong (cross-encoder finds them) | Marginal additional gain |
| **Relevance** | Generally good | No change | No change | Improves for specific product queries |
| **Latency** | ~2–5s retrieval | Slightly faster (fewer chunks) | +0.3–0.8s (cross-encoder inference) | +0.1s (BM25 lookup) |

The ordering (threshold → rerank → BM25) is deliberate: each step is
independently useful and can be measured in isolation via `eval/compare.py`
before moving to the next.
