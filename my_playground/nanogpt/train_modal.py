"""nanoGPT training on Modal.

Usage:
    modal run my_playground/nanogpt/train_modal.py
    modal run my_playground/nanogpt/train_modal.py --gpt2  # use GPT-2 tokenizer
"""

from pathlib import Path
import modal

# ── Modal app & image ─────────────────────────────────────────────
app = modal.App("nanogpt-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0.0",
        "tiktoken",
    )
)

# ── Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
EVAL_ITERS = 200
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@app.function(
    image=image,
    gpu="a10g",             # A10G (24GB) handles full batch_size=64 comfortably
    timeout=600 * 6,            # 10 minutes max
    secrets=[],
)
def train(use_gpt2_tokenizer: bool = True):
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")

    # ── Download data ────────────────────────────────────────────
    import urllib.request
    data_path = "/tmp/input.txt"
    urllib.request.urlretrieve(DATA_URL, data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters")

    # ── Tokenization ─────────────────────────────────────────────
    if use_gpt2_tokenizer:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode_fn = enc.encode
        decode_fn = enc.decode
        vocab_size = enc.n_vocab  # 50257
        print(f"Tokenizer: GPT-2 (vocab_size={vocab_size})")
    else:
        # Char-level tokenizer (original nanoGPT)
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode_fn = lambda s: [stoi[c] for c in s]
        decode_fn = lambda l: "".join(itos[i] for i in l)
        print(f"Tokenizer: char-level (vocab_size={vocab_size})")

    # ── Data preparation ─────────────────────────────────────────
    data = torch.tensor(encode_fn(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Total tokens: {len(data):,}  Train: {len(train_data):,}  Val: {len(val_data):,}")

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([d[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([d[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # ── Model definition ─────────────────────────────────────────
    class Head(nn.Module):
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(N_EMBD, head_size, bias=False)
            self.query = nn.Linear(N_EMBD, head_size, bias=False)
            self.value = nn.Linear(N_EMBD, head_size, bias=False)
            self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
            self.dropout = nn.Dropout(DROPOUT)

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            v = self.value(x)
            return wei @ v

    class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, N_EMBD)
            self.dropout = nn.Dropout(DROPOUT)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            return self.dropout(self.proj(out))

    class FeedForward(nn.Module):
        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(DROPOUT),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self, n_embd, n_head):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedForward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class GPTLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
            self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
            self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
            self.ln_f = nn.LayerNorm(N_EMBD)
            self.lm_head = nn.Linear(N_EMBD, vocab_size)
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            if targets is None:
                return logits, None
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            return logits, loss

        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -BLOCK_SIZE:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

    # ── Training ─────────────────────────────────────────────────
    torch.manual_seed(1337)
    model = GPTLanguageModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params/1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(model)
            print(f"step {step:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ── Generate sample text ─────────────────────────────────────
    print("\n─── Generated Text ───")
    if use_gpt2_tokenizer:
        start = torch.tensor([[enc.eot_token]], dtype=torch.long, device=device)
    else:
        start = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(start, max_new_tokens=500)
    print(decode_fn(generated[0].tolist()))

    # Save model checkpoint
    torch.save(model.state_dict(), "/tmp/nanogpt_model.pt")
    print("\nModel saved to /tmp/nanogpt_model.pt")

    return {
        "params_m": params / 1e6,
        "final_train_loss": float(losses["train"]),
        "final_val_loss": float(losses["val"]),
        "vocab_size": vocab_size,
        "use_gpt2": use_gpt2_tokenizer,
    }


@app.local_entrypoint()
def main(gpt2: bool = False):
    """Run nanoGPT training on Modal.

    Args:
        gpt2: Use GPT-2 tokenizer instead of char-level.
    """
    print(f"🚀 Starting nanoGPT training {'with GPT-2 tokenizer' if gpt2 else 'with char-level tokenizer'}...")
    result = train.remote(gpt2)
    print(f"\n✅ Training complete!")
    print(f"   Params: {result['params_m']:.2f}M")
    print(f"   Train loss: {result['final_train_loss']:.4f}")
    print(f"   Val loss:   {result['final_val_loss']:.4f}")
    print(f"   Vocab:      {'GPT-2 (50K)' if result['use_gpt2'] else 'char-level'}")
