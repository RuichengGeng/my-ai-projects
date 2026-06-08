"""
Lesson 01 — PyTorch Fundamentals: Tensors, Autograd, and Training Loops
========================================================================
Roadmap phase: 3 — Deep Learning and Computational Frameworks
Prerequisites: nn_toy.py (completed), NumPy proficiency
Outcome: Write a PyTorch training loop from scratch, debug tensor shapes,
         and inspect gradients with hooks.

Two demos in one file:
  1. XOR in PyTorch — the exact parallel to nn_toy.py, now with autograd
  2. Credit risk classifier — synthetic default prediction with class imbalance,
     evaluation metrics, and activation hooks

Run:
    python my_playground/ai-learning/01_pytorch_fundamentals.py
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ═══════════════════════════════════════════════════════════════════════════════
# PART 0: Environment check
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 0: Environment")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"MPS available:   {torch.backends.mps.is_available()}")  # Apple Silicon

# Pick device: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device:    {DEVICE}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Tensors — the NumPy bridge (with gradient tracking)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: Tensor operations and autograd")
print()

# --- 1a: Creating tensors ---
# The torch API mirrors NumPy by design. Almost every NumPy function you know
# has a torch equivalent. The key addition: requires_grad tracks gradients.

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
b = torch.randn(2, 2)  # same as np.random.randn

print(f"Tensor a:\n{a}")
print(f"Tensor b:\n{b}")
print(f"a @ b:\n{a @ b}")

# Convert NumPy <-> Torch (share memory when possible)
np_arr = np.array([[5.0, 6.0], [7.0, 8.0]])
t_from_np = torch.from_numpy(np_arr)
np_back = t_from_np.numpy()
print(f"NumPy -> Tensor (same memory): {np_arr.ctypes.data == t_from_np.data_ptr()}")
print()

# --- 1b: Autograd — the thing you hand-coded in nn_toy.py ---
# When requires_grad=True, every operation builds a computation graph.
# Calling .backward() runs the chain rule automatically.

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x[0] ** 2 + 3 * x[0] * x[1] + x[1] ** 2  # some scalar function
y.backward()  # computes dy/dx

# Manual check: dy/dx0 = 2*x0 + 3*x1 = 4 + 9 = 13
#               dy/dx1 = 3*x0 + 2*x1 = 6 + 6 = 12
print(f"x = {x.detach().numpy()}")
print(f"y = {y.item():.2f}")
print(f"dy/dx (autograd): {x.grad.numpy()}")
print(f"dy/dx (manual):   [13.0, 12.0]")
assert torch.allclose(x.grad, torch.tensor([13.0, 12.0])), "Autograd check failed!"
print("Autograd check: PASSED\n")

# --- 1c: Shape operations — the #1 source of bugs ---
# Key lesson: transpose, permute, view, reshape. Know the difference.
# view()   = reinterpret memory (must be contiguous -> may fail)
# reshape() = copy if needed (always works)
# transpose() / permute() = swap axes (makes tensor non-contiguous!)

t = torch.randn(4, 8, 16)
print(f"Original shape:     {t.shape}")
print(f"t.view(32, 16):     {t.reshape(32, 16).shape}")  # okay
print(f"t.permute(2, 0, 1): {t.permute(2, 0, 1).shape}")  # [16, 4, 8]
try:
    t.permute(2, 0, 1).view(32, 16)  # will fail — permute makes it non-contiguous
except RuntimeError as e:
    print(f"view on permuted fails: RuntimeError (expected)")
    print(f"  Fix: use .contiguous().view() or just .reshape()")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: XOR in PyTorch — the exact parallel to nn_toy.py
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 2: XOR in PyTorch (compare with nn_toy.py)")
print()


class XORModel(nn.Module):
    """2-layer network: input(2) -> hidden(4) -> output(1), same as nn_toy.py.

    Key differences from your NumPy version:
    - Parameters are registered via nn.Parameter (or nn.Linear auto-registers them)
    - No manual backward() math — autograd handles the chain rule
    - nn.Sequential works for simple stacks, but explicit __init__ gives
      you hooks, inspection, and debug access
    """

    def __init__(self) -> None:
        super().__init__()
        # nn.Linear = weights + bias, with proper initialization
        # Default init: Kaiming uniform for weights, uniform(-1/sqrt(fan_in), ...) for bias
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

        # Override init to match nn_toy.py's N(0, 0.5) for apples-to-apples
        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.5)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.5)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Linear already does: z = x @ W^T + b
        hidden = torch.sigmoid(self.fc1(x))
        output = torch.sigmoid(self.fc2(hidden))
        return output


def train_xor() -> None:
    """Train the XOR model and show the predicted probabilities."""
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=DEVICE)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=DEVICE)

    torch.manual_seed(42)
    model = XORModel().to(DEVICE)
    # BCEWithLogitsLoss = sigmoid + BCE in one numerically stable op
    # But we match nn_toy.py: manual sigmoid + BCELoss
    # (In production, always prefer BCEWithLogitsLoss)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    print("Before training:")
    with torch.no_grad():
        print(model(x).round(decimals=3))

    # --- The canonical PyTorch training loop ---
    for epoch in range(1, 8001):
        # Step 1: zero gradients (they accumulate by default!)
        optimizer.zero_grad()

        # Step 2: forward pass
        pred = model(x)

        # Step 3: compute loss
        loss = criterion(pred, y)

        # Step 4: backward pass (autograd does all the chain rule work)
        loss.backward()

        # Step 5: update parameters
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"epoch {epoch:>5} | loss {loss.item():.4f}")

    print("\nAfter training:")
    with torch.no_grad():
        probabilities = model(x)
        for i in range(4):
            print(
                f"input={x[i].int().tolist()} "
                f"expected={int(y[i].item())} "
                f"probability={probabilities[i].item():.3f}"
            )

    # Verify XOR is solved
    predictions = (probabilities >= 0.5).int()
    assert torch.equal(predictions, y.int()), "XOR not solved!"
    print("XOR solved: PASSED\n")


train_xor()

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Credit Risk Classifier — a realistic finance example
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 3: Credit Risk Classifier with Synthetic Data")
print()


def generate_credit_data(
    n_samples: int = 10_000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic credit default data with realistic characteristics.

    Features (9):
      - FICO score (rescaled to 0-1 range)
      - Debt-to-income ratio
      - Loan-to-value ratio
      - Number of recent inquiries
      - Credit history length (years)
      - Delinquency count (last 2 years)
      - Revolving utilization
      - Income (log scale, standardized)
      - Employment length (years, standardized)

    The target is a binary default indicator (1 = default).
    The problem has class imbalance (~15% default rate, realistic for subprime).
    """
    rng = np.random.default_rng(seed)
    n_default = int(n_samples * 0.15)
    n_good = n_samples - n_default

    # Good borrowers: higher FICO, lower DTI, lower utilization
    good = np.column_stack(
        [
            rng.normal(0.70, 0.10, n_good),  # FICO (high)
            rng.normal(0.30, 0.10, n_good),  # DTI (low)
            rng.normal(0.50, 0.15, n_good),  # LTV (moderate)
            rng.poisson(1, n_good),  # inquiries (few)
            rng.normal(10, 5, n_good),  # history (longer)
            rng.poisson(0.5, n_good),  # delinquencies (few)
            rng.normal(0.30, 0.15, n_good),  # utilization (low)
            rng.normal(0.0, 1.0, n_good),  # income (standardized)
            rng.normal(5, 3, n_good),  # employment (moderate)
        ]
    )
    y_good = np.zeros((n_good, 1))

    # Defaulted borrowers: lower FICO, higher DTI, more delinquencies
    default = np.column_stack(
        [
            rng.normal(0.45, 0.10, n_default),  # FICO (low)
            rng.normal(0.55, 0.12, n_default),  # DTI (high)
            rng.normal(0.75, 0.15, n_default),  # LTV (high)
            rng.poisson(3, n_default),  # inquiries (many)
            rng.normal(4, 3, n_default),  # history (shorter)
            rng.poisson(2, n_default),  # delinquencies (more)
            rng.normal(0.65, 0.15, n_default),  # utilization (high)
            rng.normal(-0.3, 1.0, n_default),  # income (lower)
            rng.normal(2, 2, n_default),  # employment (shorter)
        ]
    )
    y_default = np.ones((n_default, 1))

    # Combine, shuffle, convert
    x_np = np.vstack([good, default])
    y_np = np.vstack([y_good, y_default])
    idx = rng.permutation(n_samples)
    x_np, y_np = x_np[idx], y_np[idx]

    # Clip features to realistic ranges
    x_np = np.clip(x_np, 0.01, 0.99)

    return torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)


class CreditRiskMLP(nn.Module):
    """3-layer MLP for credit default prediction.

    Architecture choices explained:
    - BatchNorm1d after each linear layer: stabilizes training, reduces sensitivity
      to weight init, allows higher learning rates. Place BEFORE activation.
    - Dropout(0.3): regularization for imbalanced finance data where overfitting
      to majority class is easy.
    - ReLU vs sigmoid hidden: ReLU avoids vanishing gradient in deeper nets.
      Use sigmoid only at the output for binary probability.
    - 64 -> 32 -> 16: decreasing widths (bottleneck) is a common pattern
      that forces the network to learn compressed representations.
    """

    def __init__(self, input_dim: int = 9) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)

        self.out = nn.Linear(16, 1)

        # Store intermediate activations for hook inspection
        self._activations: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        z1 = self.fc1(x)
        a1 = F.relu(self.bn1(z1))
        a1 = self.drop1(a1)
        self._activations["layer1"] = a1.detach()

        # Layer 2
        z2 = self.fc2(a1)
        a2 = F.relu(self.bn2(z2))
        a2 = self.drop2(a2)
        self._activations["layer2"] = a2.detach()

        # Layer 3
        z3 = self.fc3(a2)
        a3 = F.relu(self.bn3(z3))
        self._activations["layer3"] = a3.detach()

        # Output (no sigmoid — BCEWithLogitsLoss handles it)
        return self.out(a3)


def compute_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute classification metrics for imbalanced data.

    Important: For imbalanced finance data, accuracy is misleading.
    Always check precision, recall, and especially PR-AUC.
    """
    y_pred = (y_prob >= threshold).float()
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "default_rate_true": y_true.mean().item(),
        "default_rate_pred": y_pred.float().mean().item(),
    }


def train_credit_model() -> None:
    """Full training pipeline: data -> model -> train -> evaluate -> inspect."""
    torch.manual_seed(42)

    # --- Generate data ---
    x_all, y_all = generate_credit_data(n_samples=10_000)

    # Train/val/test split (70/15/15)
    n = len(x_all)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    indices = torch.randperm(n)
    train_idx, val_idx, test_idx = indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]

    x_train, y_train = x_all[train_idx].to(DEVICE), y_all[train_idx].to(DEVICE)
    x_val, y_val = x_all[val_idx].to(DEVICE), y_all[val_idx].to(DEVICE)
    x_test, y_test = x_all[test_idx].to(DEVICE), y_all[test_idx].to(DEVICE)

    print(f"Train: {len(x_train)} samples, default rate: {y_train.mean().item():.1%}")
    print(f"Val:   {len(x_val)} samples, default rate: {y_val.mean().item():.1%}")
    print(f"Test:  {len(x_test)} samples, default rate: {y_test.mean().item():.1%}")
    print()

    # --- DataLoader: batching, shuffling, prefetching ---
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # --- Model, loss, optimizer, scheduler ---
    model = CreditRiskMLP(input_dim=9).to(DEVICE)
    # BCEWithLogitsLoss = sigmoid + BCE in one stable op (preferred)
    criterion = nn.BCEWithLogitsLoss()
    # AdamW = Adam + decoupled weight decay (better regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Cosine annealing: gradually reduce LR to near-zero by end of training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # --- Training loop with validation ---
    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, 51):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            # Optional: gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Save best model (simple checkpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:>3} | train_loss {epoch_loss:.4f} | val_loss {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
            )

    # Load best checkpoint
    model.load_state_dict(best_state)
    print(f"\nBest val loss: {best_val_loss:.4f}")

    # --- Evaluate on test set ---
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        test_loss = criterion(test_logits, y_test).item()
        test_prob = torch.sigmoid(test_logits)
        metrics = compute_metrics(y_test, test_prob)

    print(f"\nTest loss: {test_loss:.4f}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # --- Hook inspection: look at intermediate activations ---
    # Forward hooks let you inspect what the model is "thinking" at each layer.
    print("\n--- Activation statistics (inference on test set) ---")
    _ = model(x_test[:100])  # forward pass fills self._activations
    for layer_name, activations in model._activations.items():
        print(
            f"  {layer_name}: "
            f"mean={activations.mean().item():.3f}, "
            f"std={activations.std().item():.3f}, "
            f"dead_neurons={(activations.sum(dim=0) == 0).sum().item()}/{activations.shape[1]}"
        )

    # --- Shape assertions (debug habit) ---
    print("\n--- Shape checks ---")
    dummy = torch.randn(1, 9).to(DEVICE)
    out = model(dummy)
    assert out.shape == (1, 1), f"Output shape wrong: {out.shape}"
    assert 0.70 < metrics["recall"] < 1.0, f"Recall suspicious: {metrics['recall']:.3f}"
    print("All shape assertions passed.")

    # --- Key lesson: why not just use accuracy? ---
    print(f"\n--- Imbalanced data lesson ---")
    print(f"Default rate: {y_test.mean().item():.1%}")
    print(f"Accuracy:     {metrics['accuracy']:.1%}")
    print(f"Precision:    {metrics['precision']:.1%}")
    print(f"Recall:       {metrics['recall']:.1%}")
    print(f"F1:           {metrics['f1']:.1%}")
    print("Naive 'predict all good' accuracy would be: {:.1%}".format(1 - y_test.mean().item()))
    print(f"Model beats naive by: {metrics['accuracy'] - (1 - y_test.mean().item()):.1%}")


if __name__ == "__main__":
    train_credit_model()
    print()
    print("=" * 60)
    print("Lesson 01 complete. Next: hooks, mixed precision, and training loop internals.")
    print("=" * 60)
