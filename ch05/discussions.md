# Chapter 5 — Discussions

Questions and deep dives from studying pretraining.

---

## What shape does calc_loss_batch return?

**A scalar — shape `()`, a single number.**

The transformation chain:

```
input_batch:   (batch, seq_len)                    e.g. (4, 256)
target_batch:  (batch, seq_len)                    e.g. (4, 256)
    ↓ model forward
logits:        (batch, seq_len, vocab_size)         e.g. (4, 256, 50257)
    ↓ reshape (flatten batch × seq_len)
logits:        (batch * seq_len, vocab_size)        e.g. (1024, 50257)
targets:       (batch * seq_len,)                   e.g. (1024,)
    ↓ F.cross_entropy (reduction='mean')
loss:          ()                                   ← scalar!
```

Why reshape? `F.cross_entropy` requires 2D input `(N, C)` + 1D target `(N,)`. So batch and seq_len get merged: "4 sentences × 256 positions" → "1024 independent next-token predictions", averaged into one number.

---

## If loss is just a number, what does loss.backward() do?

**Starting from that single number, it walks backward through the computation graph and computes how much each parameter contributed to the loss (gradients).**

PyTorch secretly records every operation during forward pass. `backward()` uses the chain rule to compute:

```
∂loss/∂W₁  → stored in W₁.grad (same shape as W₁)
∂loss/∂W₂  → stored in W₂.grad (same shape as W₂)
...for every parameter
```

**One scalar loss → backward → every parameter gets a gradient tensor of its own shape.** The gradient tells each parameter "which direction to move to make loss smaller."

Why must loss be scalar? If loss were a vector `[8.2, 5.1, 7.0]`, the derivative `∂loss/∂W` would be ambiguous — derivative with respect to which component? A scalar gives an unambiguous optimization target.

The full training step:
1. `loss = calc_loss_batch(...)` — forward, get scalar loss
2. `loss.backward()` — compute all gradients
3. `optimizer.step()` — update: `W_new = W - lr × W.grad`
4. `optimizer.zero_grad()` — reset gradients for next iteration

---

## How does AdamW optimizer work?

**Adam + correctly decoupled Weight Decay.** The standard optimizer for GPT training.

### Evolution of optimizers

**SGD:** `W = W - lr × gradient` — simple but unstable, all params use same learning rate.

**Momentum:** Adds a "velocity" term — smooths direction using exponential moving average of gradients. Like a ball rolling — doesn't stop at every small bump.

**Adam:** Momentum + per-parameter adaptive learning rate. Maintains two states:
- `m` = moving average of gradients (direction)
- `v` = moving average of squared gradients (magnitude/volatility)

Key: dividing by `√v` means parameters with consistently large gradients get smaller learning rates (auto-dampening), and vice versa.

### The AdamW fix

Original Adam applied weight decay inside the gradient (L2 regularization) — but Adam's adaptive scaling distorted the decay signal. AdamW decouples them:

```
# Normal Adam update (no weight decay in gradient)
W = W - lr × m̂ / (√v̂ + ε)

# Weight decay applied separately (not filtered through Adam)
W = W - lr × λ × W
```

**Weight decay = continuously shrink all weights slightly each step.** Only weights that gradients actively push up will stay large. This prevents overfitting — the model can't memorize by growing huge weights.

### Typical GPT config

```python
optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
```

| Component | Role |
|-----------|------|
| m (momentum) | Smooth gradient direction |
| v (adaptive) | Per-parameter learning rate |
| Weight Decay | Compress weights, prevent overfitting |
| "W" in AdamW | Decouple weight decay from Adam's adaptivity |
