# Chapter 4 — Discussions

Questions and deep dives from studying the GPT architecture.

---

## What does LayerNorm do in a TransformerBlock?

**Prevents numerical explosion/vanishing across layers.**

Without normalization, 12 layers of matrix multiplications cause values to grow (or shrink) uncontrollably. LayerNorm "resets" each layer's input to a stable range.

**How it works:** For each token's vector independently:
1. Compute mean μ and variance σ² across the embedding dimensions
2. Normalize: (x - μ) / σ → values centered around 0, spread ≈ 1
3. Apply learnable scale γ and shift β → model decides the final range

**Pre-Norm (GPT's approach):** Normalize *before* attention and feed-forward, not after.

```
x → LayerNorm → Attention → +residual → LayerNorm → FeedForward → +residual
```

**Why LayerNorm over BatchNorm?**
- LayerNorm normalizes across dimensions (within one token) — independent of batch size and sequence length
- BatchNorm normalizes across the batch (same dimension, different samples) — breaks with variable sequence lengths and small batches
- LayerNorm behaves identically during training and inference

**Analogy:** LayerNorm is a "blood pressure regulator" for each layer — no matter how extreme the previous layer's output was, it gets pulled back to a healthy range before the next layer processes it.

---

## What is the role of activation functions?

**Introduce non-linearity. Without them, stacking layers is pointless.**

The math:
```
Without activation:
  Layer 1: y = xW₁ + b₁
  Layer 2: z = yW₂ + b₂ = x(W₁W₂) + (b₁W₂ + b₂) = xW' + b'
  → Two linear layers = one linear layer. 100 layers = still one layer.

With activation:
  Layer 1: y = GELU(xW₁ + b₁)
  Layer 2: z = GELU(yW₂ + b₂)
  → Cannot be collapsed. Each layer adds expressive power.
```

**Linear transforms can only rotate, scale, and shift** — they draw straight lines/planes to separate data. Non-linear activations allow curved decision boundaries, enabling the model to learn arbitrarily complex functions.

**In Transformer's FeedForward:**
```
FFN = Linear(768→3072) → GELU → Linear(3072→768)
```
Without GELU, the two Linear layers collapse into one. With GELU, the network can learn complex feature transformations. Attention decides "where to get information"; FeedForward + activation decides "what to do with it."

**Why GELU over ReLU?** GELU is smooth everywhere (differentiable), while ReLU has a hard kink at x=0. GELU also allows small negative values through, preserving more gradient signal. See `gelu_vs_relu.png` for the comparison.
