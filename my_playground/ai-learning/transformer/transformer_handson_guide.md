# Transformers: A Hands-On Learning Path

You've read *The Illustrated Transformer* and understand the mechanism. The fastest way to deepen that into real understanding is to **run code and watch the pieces move**. This guide orders resources from "easiest to play with" to "build the whole thing," and tells you exactly what to look at in each so you're not just staring at a repo.

A good rule: don't just run things — **change a number and predict what happens, then check.** That prediction-and-check loop is where understanding actually forms.

---

## 0. Start here (included): `self_attention_demo.py`

The companion script in this folder implements **one self-attention head from scratch in NumPy** on the article's "the animal didn't cross the street because it was *it*..." sentence. No deep-learning framework needed — just `pip install numpy`.

Run it and read top to bottom. It prints every intermediate the article draws as a diagram: the Q/K/V vectors, the raw Q·Kᵀ scores, the scaled scores, the softmax weights, and the final output Z.

**Experiments to try (edit the file and re-run):**

- Change `d_k` from 4 and watch the scaled-score magnitudes change — see why dividing by √d_k matters.
- Delete the `/ np.sqrt(d_k)` scaling and look at the softmax weights: they get spikier (closer to a hard argmax). That's the problem scaling fixes.
- In the "simulated trained head" section, raise the `street` score above `animal` and watch "it"'s output vector shift toward street's Value — this is the trophy/suitcase coreference flip in miniature.

Once the printed numbers feel unsurprising, you've internalized the core. Move on.

---

## 1. SEE attention in a real, trained model — **BertViz**

The single highest insight-per-minute tool. BertViz renders the attention weights of real pretrained models (BERT, GPT-2) as interactive lines between words — exactly the colored-line visualizations in the article, but live and on your own sentences.

- Repo: https://github.com/jessevig/bertviz
- Easiest path: open its Colab notebook (linked from the README) so there's nothing to install.

**What to do:** Feed in *"The animal didn't cross the street because it was too tired"*, then change the last word to *"wide"* and see whether "it" re-points from animal to street. Use the "head view" to confirm the article's claim that **different heads attend to different things** — some track syntax, some track the previous/next token, some track coreference.

---

## 2. READ a clean, runnable implementation — **The Annotated Transformer**

Harvard NLP's line-by-line PyTorch implementation of the original "Attention Is All You Need" paper, with the paper's prose interleaved with the code that implements each sentence. The 2022 refresh uses modern PyTorch.

- Blog (read this version): http://nlp.seas.harvard.edu/annotated-transformer/
- Code: https://github.com/harvardnlp/annotated-transformer

**What to look at, in order:** the `attention()` function (you'll recognize it instantly from the demo above), then `MultiHeadedAttention`, then how the encoder and decoder `LayerNorm`/residual wrappers (`SublayerConnection`) are stitched together, then the `subsequent_mask` that enforces the decoder's look-backward-only rule. Seeing the mask as an actual triangular matrix makes the decoder section of the article click.

---

## 3. BUILD a GPT from an empty file — **Karpathy's "Let's build GPT"**

The best end-to-end "I made the whole thing" experience. Andrej Karpathy starts from a blank file and builds a working decoder-only Transformer that generates Shakespeare-like text, explaining every line on video. This is decoder-only — the perfect complement to the article's encoder-decoder focus, and it directly answers your "why not just one stack?" question.

- Video + code, commit by commit: https://github.com/karpathy/build-nanogpt
- The "Let's build GPT: from scratch, in code, spelled out" YouTube lecture is linked from that repo.

**Why it's worth the ~2 hours:** you'll implement masked self-attention yourself, see the autoregressive generation loop run, and watch the loss drop in real time. After this, nothing in the architecture is a black box.

---

## 4. TRAIN / FINETUNE something real — **nanoGPT**

The "teeth over education" sibling of the above: a tight ~600 lines that can actually reproduce GPT-2 (124M). Use it once you want to train on your own text or feel how the knobs (layers, heads, embedding size, context length) affect results.

- Repo: https://github.com/karpathy/nanoGPT
- Its older, more heavily-commented cousin **minGPT** (https://github.com/karpathy/minGPT) is also great for reading.

**A concrete first project:** train the char-level Shakespeare model from the README on a laptop in a few minutes, then swap in your own text file (song lyrics, your chat logs, code) and see what it generates. Then bump `n_head` from 6 to 1 and observe quality drop — a felt demonstration of why multi-head matters.

---

## Suggested route by time available

- **30 minutes:** run `self_attention_demo.py`, then play in BertViz (steps 0–1).
- **An afternoon:** add the Annotated Transformer read-through (step 2).
- **A weekend:** do Karpathy's build-GPT video and train nanoGPT on your own text (steps 3–4).

## If you want to go past the original architecture

The article is from 2018; production models have evolved. Once the above is solid, look up: **RoPE** (rotary positional embeddings — replaced the sine/cosine scheme), **RMSNorm** and **pre-norm** (replaced post-LayerNorm), **KV-caching** (the trick that makes the autoregressive loop fast at inference), **Grouped/Multi-Query Attention** (cheaper attention), and **FlashAttention** (a faster exact attention implementation). Jay Alammar's own book, *Hands-On Large Language Models* (llm-book.com), covers these as an updated version of the post you read.

---

*Sources: [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) · [annotated-transformer repo](https://github.com/harvardnlp/annotated-transformer) · [BertViz](https://github.com/jessevig/bertviz) · [build-nanogpt](https://github.com/karpathy/build-nanogpt) · [nanoGPT](https://github.com/karpathy/nanoGPT) · [minGPT](https://github.com/karpathy/minGPT)*
