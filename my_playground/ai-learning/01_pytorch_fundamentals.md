# Lesson 01 — PyTorch Fundamentals: Tensors, Autograd, and Training Loops

## Topic
**Roadmap phase:** 3 — Deep Learning and Computational Frameworks
**Prerequisites:** `nn_toy.py` (completed), NumPy proficiency
**Outcome:** Write a PyTorch training loop from scratch. Debug tensor shapes. Inspect activations with hooks. Handle imbalanced finance data properly.

## Why It Matters

You hand-coded backpropagation in `nn_toy.py`. Every professional deep learning engineer uses PyTorch's `autograd` to do the same thing — automatically. The five-step training loop (`zero_grad → forward → loss → backward → step`) you'll learn here is the scaffolding for **everything**: custom architectures, Transformers, LoRA fine-tuning, reinforcement learning, and multi-agent training.

This is also where quant intuition meets deep learning reality: imbalanced defaults, precision vs. recall tradeoffs, and activation inspection that mirrors the model diagnostics you already do.

## Core Ideas

### 1. Tensors = NumPy arrays with gradient tracking
If `requires_grad=True`, every arithmetic operation is recorded in a computation graph. Calling `.backward()` runs the chain rule automatically — replacing 50+ lines of manual gradient derivation from `nn_toy.py`.

### 2. The 5-step training loop
Every PyTorch training script follows this pattern:

```python
for epoch in range(epochs):
    optimizer.zero_grad()      # ① reset accumulated gradients
    output = model(data)       # ② forward: build compute graph
    loss = criterion(output, y)  # ③ scalar loss
    loss.backward()            # ④ backward: populate .grad fields
    optimizer.step()           # ⑤ update: θ -= η * ∇θ
```

Forgetting `zero_grad()` is the most common beginner bug — gradients **accumulate** by default (useful for gradient accumulation over micro-batches).

### 3. `nn.Module` organizes parameters and forward logic
You used `self.w1`, `self.b1` as plain arrays in `nn_toy.py`. `nn.Module` does the same but registers them properly so `model.parameters()`, `.to(device)`, `.state_dict()`, and checkpointing all work automatically.

### 4. BCEWithLogitsLoss vs. BCELoss
`BCEWithLogitsLoss` = `sigmoid + BCE` in a single numerically stable operation. This is the same trick you learned in `nn_toy.py`: the combined gradient simplifies to `ŷ - y`, avoiding the vanishing gradient of sigmoid saturation. **Always prefer it** over manual sigmoid + BCELoss.

### 5. DataLoader handles batching, shuffling, prefetching
For small toy data you can train on the entire dataset at once. For real problems, you need mini-batches. `DataLoader` + `TensorDataset` is the standard pattern.

## Minimal Math

### Autograd: the chain rule, automated

For your XOR network `output = sigmoid(W2 @ sigmoid(W1 @ x + b1) + b2)`:

In `nn_toy.py`, you manually computed:
```
∂L/∂W2 = hidden_a.T @ (pred - y) / N
∂L/∂W1 = X.T @ (∂L/∂hidden_a ⊙ sigmoid'(hidden_z))
```

In PyTorch, `loss.backward()` does all of this by walking backward through the computation graph. The dynamic graph means it works for **any** PyTorch operation — loops, conditionals, custom CUDA kernels.

### Shape rules for linear layers

`nn.Linear(in_features, out_features)` stores weight as `[out_features, in_features]` internally. The forward is `x @ W^T + b`, so:

```
input:  [N, in_features]  @  W^T [in_features, out_features]  +  b [out_features]
output: [N, out_features]
```

This is the opposite convention from `nn_toy.py` where you stored `W` as `[in_features, out_features]`. The transpose is internal — just know that `model.fc1.weight.shape` is `[out, in]`.

### Why AdamW?

| Optimizer | Key idea | When |
|-----------|----------|------|
| SGD | θ -= η·∇L | Simple, needs careful LR tuning |
| SGD + Momentum | θ -= η·(∇L + β·velocity) | Smoother descent |
| Adam | Adaptive per-parameter LR from 1st/2nd moment estimates | Default for most problems |
| AdamW | Adam + decoupled weight decay | Better generalization, standard for transformers |

## Implementation

### Files

| File | Purpose |
|------|---------|
| `01_pytorch_fundamentals.py` | Runnable script: tensor ops, XOR in PyTorch, credit risk classifier |

### Run it

```bash
python my_playground/ai-learning/01_pytorch_fundamentals.py
```

### What the script demonstrates

**Part 1 — Tensors and Autograd**
- Creating tensors, NumPy interop
- Autograd: define a function, call `.backward()`, verify gradients manually
- Shape operations: `view` vs `reshape`, the `permute` contiguity trap

**Part 2 — XOR in PyTorch**
- Exact parallel to `nn_toy.py`: same architecture, same init, same data
- The 5-step training loop
- `nn.BCELoss` to match the NumPy version (purely for comparison — in practice use `BCEWithLogitsLoss`)

**Part 3 — Credit Risk Classifier**
- Synthetic credit data with realistic class imbalance (~15% default rate)
- `BatchNorm1d` + `Dropout` for regularization
- `AdamW` + `CosineAnnealingLR` for optimization
- Train/val/test split with `DataLoader`
- Full evaluation: accuracy, precision, recall, F1 — with an explanation of why accuracy alone lies on imbalanced data
- Activation hook inspection: mean, std, dead neurons per layer
- Gradient clipping and best-model checkpointing
- Shape assertions as a debugging habit

## Exercises

### 1. Concept check (5 min)
In `nn_toy.py`, if you forget to update `b2`, what happens? Does the loss converge? Run the experiment and explain.

### 2. Code modification (10 min)
In the credit risk model, remove `BatchNorm1d` from all layers. Retrain. What changes?
- Does the loss converge faster or slower?
- Do activations have larger or smaller variance?
- Does validation loss become noisier?

### 3. Applied mini-task (20 min)
Add a 4th hidden layer (16 → 8) to `CreditRiskMLP`. Measure:
- Does it improve validation loss?
- Count dead ReLU neurons per layer
- Try replacing ReLU with GELU — does dead neuron count go to zero?

### 4. Shape debugging drill (10 min)
Intentionally introduce each of these bugs, observe the error message, then fix:
1. Pass input of shape `[batch, 10]` to a model expecting `[batch, 9]`
2. Call `.view()` on a permuted tensor without `.contiguous()`
3. Forget to move the model to the same device as the data

## Success Criteria

- [ ] You can write the 5-step training loop from memory
- [ ] You can explain why `zero_grad()` is necessary
- [ ] You can debug a shape mismatch error in under 30 seconds
- [ ] You understand why accuracy is misleading for a 15% default-rate dataset
- [ ] You can read `.grad` values after `backward()` to inspect what the optimizer is doing
- [ ] You can explain the difference between `BCELoss` and `BCEWithLogitsLoss`

## Next Step

**Lesson 02: Training Loop Internals** — hooks, mixed precision (`torch.cuda.amp`), gradient accumulation, learning rate finders, TensorBoard logging, and profiling with `torch.profiler`. This builds directly on the training loop pattern you now control.
