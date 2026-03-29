"""
Chapter 4: Implementing a GPT Model from Scratch.

Builds the complete GPT architecture step by step:
  1. LayerNorm       — manual layer normalization (no nn.LayerNorm)
  2. GELU            — Gaussian Error Linear Unit activation
  3. FeedForward     — position-wise feed-forward network
  4. TransformerBlock — pre-norm transformer block with residual connections
  5. GPTModel        — full GPT model: embeddings → N blocks → head

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 4.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import MultiHeadAttention from Chapter 3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ch03.attention import MultiHeadAttention
from ch04.config import GPTConfig


# ──────────────────────────────────────────────────────────────
# 1. Layer Normalization (手动实现，理解原理)
# ──────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """
    Layer Normalization — normalizes across the feature dimension.

    For each token's embedding vector x of shape (d_model,):
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta

    Why LayerNorm (not BatchNorm)?
        - BatchNorm normalizes across the batch dimension → depends on batch size
        - LayerNorm normalizes across features → independent per sample
        - More stable for variable-length sequences in NLP

    Parameters:
        gamma (scale): learnable, initialized to 1
        beta (shift):  learnable, initialized to 0
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Args:
            d_model: dimension of the input features (embedding size)
            eps:     small constant for numerical stability in division
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))   # (d_model,) — scale
        self.beta = nn.Parameter(torch.zeros(d_model))    # (d_model,) — shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch, seq_len, d_model)
        Returns:
            normalized tensor, same shape (batch, seq_len, d_model)
        """
        # Compute mean and variance across the last dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)    # (batch, seq_len, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch, seq_len, 1)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # (batch, seq_len, d_model)

        # Scale and shift with learnable parameters
        return self.gamma * x_norm + self.beta  # (batch, seq_len, d_model)


# ──────────────────────────────────────────────────────────────
# 2. GELU Activation (GPT 用的激活函数)
# ──────────────────────────────────────────────────────────────

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    GELU(x) = x · Φ(x)

    where Φ(x) is the CDF of the standard normal distribution.
    Approximation (used in GPT-2):
        GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))

    Why GELU instead of ReLU?
        - ReLU: hard cutoff at 0 → "dead neurons" problem
        - GELU: smooth, probabilistic gating → allows small negative values
        - Better gradient flow during training
        - Empirically works better for transformers / NLP tasks
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, any shape
        Returns:
            GELU-activated tensor, same shape
        """
        return 0.5 * x * (1.0 + torch.tanh(
            (2.0 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)
        ))


# ──────────────────────────────────────────────────────────────
# 3. FeedForward Network (位置前馈网络)
# ──────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Architecture: Linear(d_model → 4·d_model) → GELU → Linear(4·d_model → d_model) → Dropout

    The inner dimension is 4× the model dimension — this is the standard
    design from "Attention Is All You Need" and used in all GPT models.

    Why 4×?
        - Empirically chosen ratio in the original Transformer paper
        - Gives the model more capacity to learn complex transformations
        - Each token is processed independently (position-wise)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Args:
            d_model: input and output dimension
            dropout: dropout rate after the second linear layer
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # Up-project: (d_model) → (4·d_model)
            GELU(),                              # Activation
            nn.Linear(4 * d_model, d_model),    # Down-project: (4·d_model) → (d_model)
            nn.Dropout(dropout),                 # Regularization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch, seq_len, d_model)
        Returns:
            output tensor, shape (batch, seq_len, d_model)
        """
        return self.net(x)  # (batch, seq_len, d_model)


# ──────────────────────────────────────────────────────────────
# 4. Transformer Block (Pre-norm 架构)
# ──────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single Transformer block with Pre-norm architecture.

    Pre-norm (GPT-2 style):
        x → LayerNorm → MultiHeadAttention → + residual
          → LayerNorm → FeedForward        → + residual

    vs. Post-norm (original Transformer):
        x → MultiHeadAttention → + residual → LayerNorm
          → FeedForward        → + residual → LayerNorm

    Why Pre-norm?
        - More stable training (gradients flow better through residual paths)
        - LayerNorm before attention/FF means inputs are always normalized
        - Used by GPT-2, GPT-3, and most modern LLMs
        - Allows training deeper models without warmup tricks

    Residual connections:
        output = x + sublayer(x)
        - Allows gradients to flow directly through skip connections
        - Solves vanishing gradient problem in deep networks
        - Each layer only needs to learn the "residual" (the difference)
    """

    def __init__(self, cfg: GPTConfig):
        """
        Args:
            cfg: GPTConfig with model hyperparameters
        """
        super().__init__()
        # Pre-norm layers
        self.ln1 = LayerNorm(cfg.d_model)  # Before attention
        self.ln2 = LayerNorm(cfg.d_model)  # Before feed-forward

        # Multi-head causal self-attention (from ch03)
        self.attn = MultiHeadAttention(
            d_in=cfg.d_model,
            d_out=cfg.d_model,
            num_heads=cfg.n_heads,
            max_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )

        # Feed-forward network
        self.ff = FeedForward(cfg.d_model, cfg.dropout)

        # Dropout for residual connections
        self.drop_resid = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape (batch, seq_len, d_model)
        Returns:
            output tensor, shape (batch, seq_len, d_model)
        """
        # ---- Attention sub-block with residual ----
        # Pre-norm → attention → dropout → add residual
        shortcut = x                                        # (batch, seq_len, d_model)
        x = self.ln1(x)                                     # (batch, seq_len, d_model)
        attn_out, _ = self.attn(x)                           # (batch, seq_len, d_model)
        x = shortcut + self.drop_resid(attn_out)             # (batch, seq_len, d_model)

        # ---- Feed-forward sub-block with residual ----
        # Pre-norm → FF → add residual (dropout is inside FeedForward)
        shortcut = x                                        # (batch, seq_len, d_model)
        x = self.ln2(x)                                     # (batch, seq_len, d_model)
        ff_out = self.ff(x)                                  # (batch, seq_len, d_model)
        x = shortcut + ff_out                                # (batch, seq_len, d_model)

        return x


# ──────────────────────────────────────────────────────────────
# 5. GPT Model (完整模型)
# ──────────────────────────────────────────────────────────────

class GPTModel(nn.Module):
    """
    Complete GPT Model.

    Architecture:
        Input token IDs
            ↓
        Token Embedding  (vocab_size → d_model)
            +
        Position Embedding  (max_seq_len → d_model)
            ↓
        Dropout
            ↓
        N × TransformerBlock
            ↓
        Final LayerNorm
            ↓
        Linear Head  (d_model → vocab_size)  → logits
            ↓
        Output logits  (batch, seq_len, vocab_size)

    Note: We do NOT tie the token embedding weights with the output head
    for simplicity. The original GPT-2 uses weight tying to reduce params.
    """

    def __init__(self, cfg: GPTConfig):
        """
        Args:
            cfg: GPTConfig with model hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        # Token and position embeddings
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # (vocab_size, d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)  # (max_seq_len, d_model)
        self.drop_emb = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        # Final layer norm (applied after all blocks, before the head)
        self.final_norm = LayerNorm(cfg.d_model)

        # Output projection: project from d_model to vocabulary size
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: input token indices, shape (batch, seq_len)
                 Each value is in [0, vocab_size)
        Returns:
            logits: shape (batch, seq_len, vocab_size)
                    Raw scores for each token in the vocabulary
        """
        batch, seq_len = idx.shape
        assert seq_len <= self.cfg.max_seq_len, (
            f"Sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        pos = torch.arange(seq_len, device=idx.device)  # (seq_len,)

        # Embeddings: token + position
        tok_embeddings = self.tok_emb(idx)    # (batch, seq_len, d_model)
        pos_embeddings = self.pos_emb(pos)    # (seq_len, d_model) — broadcast over batch
        x = self.drop_emb(tok_embeddings + pos_embeddings)  # (batch, seq_len, d_model)

        # Pass through all transformer blocks
        x = self.blocks(x)  # (batch, seq_len, d_model)

        # Final normalization
        x = self.final_norm(x)  # (batch, seq_len, d_model)

        # Project to vocabulary
        logits = self.head(x)  # (batch, seq_len, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    from ch04.config import GPT2_SMALL

    cfg = GPT2_SMALL
    model = GPTModel(cfg)

    print("=" * 60)
    print("GPT Model Architecture")
    print("=" * 60)
    print(model)

    print(f"\n{'=' * 60}")
    print(f"Total trainable parameters: {model.count_parameters():,}")
    print(f"{'=' * 60}")

    # Test forward pass
    batch_size, seq_len = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    logits = model(idx)
    print(f"\nInput shape:  {idx.shape}")        # (2, 16)
    print(f"Output shape: {logits.shape}")        # (2, 16, 50257)

    # Test individual components
    print(f"\n{'=' * 60}")
    print("Component Tests")
    print(f"{'=' * 60}")

    x = torch.randn(2, 8, cfg.d_model)  # (batch=2, seq_len=8, d_model=768)

    ln = LayerNorm(cfg.d_model)
    out = ln(x)
    print(f"LayerNorm:    {x.shape} → {out.shape}")

    gelu = GELU()
    out = gelu(x)
    print(f"GELU:         {x.shape} → {out.shape}")

    ff = FeedForward(cfg.d_model)
    out = ff(x)
    print(f"FeedForward:  {x.shape} → {out.shape}")

    block = TransformerBlock(cfg)
    out = block(x)
    print(f"TransBlock:   {x.shape} → {out.shape}")
