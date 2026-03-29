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

---

## How does Cross-Entropy Loss work?

**Measures the "distance" between the model's predicted probability distribution and the true answer.**

Core formula: `loss = -log(P(correct answer))`

```
Model predicts: cat=0.7, dog=0.2, the=0.05, sat=0.05
True answer:    cat

loss = -log(0.7) = 0.36   ← good prediction, low loss

Model predicts: cat=0.01, dog=0.8, the=0.1, sat=0.09
True answer:    cat

loss = -log(0.01) = 4.6   ← bad prediction, high loss
```

**Why -log?** Perfect match for "penalty" intuition:
- P → 1.0: loss → 0 (perfect)
- P → 0.0: loss → ∞ (catastrophic)
- Non-linear: the penalty for going from 0.1 to 0.01 is much larger than from 0.9 to 0.8

**Actual computation:** `F.cross_entropy(logits, target)` combines three steps internally:
1. softmax(logits) → probabilities
2. Take probability of correct class
3. Negate the log

These are fused for numerical stability (log-sum-exp trick avoids overflow from large logits).

**Information theory interpretation:** Cross-entropy `H(p,q) = -Σ p(x) log q(x)` measures how many extra bits are needed to encode the true distribution p using the model's distribution q. When p is one-hot, it simplifies to `-log(q[correct])`.

**Perplexity** = `e^(cross-entropy loss)` — the human-readable version:
- Loss 4.0 → perplexity ≈ 55 (model is "choosing among 55 options")
- Loss 2.0 → perplexity ≈ 7 (much better)
- Loss 0.0 → perplexity = 1 (perfect, zero confusion)
