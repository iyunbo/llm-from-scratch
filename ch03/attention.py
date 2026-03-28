"""
Chapter 3: Coding Attention Mechanisms.

Implements self-attention from scratch, progressing through three levels:
  1. SimpleAttention     — manual QKV, no learnable parameters
  2. CausalAttention     — single-head with nn.Linear projections, causal mask, dropout
  3. MultiHeadAttention  — multi-head version with split heads + output projection

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    """
    Simplest self-attention: no learnable parameters.

    Uses the input directly as Q, K, V (or a simple linear transformation
    via raw matrix multiplication). This is for pedagogical purposes only —
    it shows the core attention math without any bells and whistles.
    """

    def __init__(self, d_in, d_out):
        """
        Args:
            d_in:  dimensionality of input embeddings
            d_out: dimensionality of output (Q, K, V projection size)
        """
        super().__init__()
        # Manual weight matrices (not nn.Linear — no bias, raw parameters)
        self.W_q = nn.Parameter(torch.randn(d_in, d_out))  # (d_in, d_out)
        self.W_k = nn.Parameter(torch.randn(d_in, d_out))  # (d_in, d_out)
        self.W_v = nn.Parameter(torch.randn(d_in, d_out))  # (d_in, d_out)

    def forward(self, x):
        """
        Args:
            x: input tensor, shape (batch, seq_len, d_in)
        Returns:
            context_vectors: shape (batch, seq_len, d_out)
            attention_weights: shape (batch, seq_len, seq_len)
        """
        # Project input to Q, K, V
        Q = x @ self.W_q  # (batch, seq_len, d_out)
        K = x @ self.W_k  # (batch, seq_len, d_out)
        V = x @ self.W_v  # (batch, seq_len, d_out)

        d_k = K.shape[-1]

        # Scaled dot-product attention
        # scores: (batch, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)

        # Attention weights via softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # Weighted sum of values
        context_vectors = attention_weights @ V  # (batch, seq_len, d_out)

        return context_vectors, attention_weights


class CausalAttention(nn.Module):
    """
    Single-head causal (masked) self-attention.

    Adds three improvements over SimpleAttention:
      1. nn.Linear for Q, K, V projections (with bias)
      2. Causal mask — prevents attending to future tokens
      3. Dropout on attention weights
    """

    def __init__(self, d_in, d_out, max_len=1024, dropout=0.0):
        """
        Args:
            d_in:     dimensionality of input embeddings
            d_out:    dimensionality of output
            max_len:  maximum sequence length (for causal mask buffer)
            dropout:  dropout rate on attention weights
        """
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out)
        self.W_k = nn.Linear(d_in, d_out)
        self.W_v = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)

        # Register causal mask as a buffer (not a parameter)
        # Upper-triangular mask: 1s above diagonal → positions to mask
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("mask", mask)  # (max_len, max_len)

    def forward(self, x):
        """
        Args:
            x: input tensor, shape (batch, seq_len, d_in)
        Returns:
            context_vectors: shape (batch, seq_len, d_out)
            attention_weights: shape (batch, seq_len, seq_len)
        """
        seq_len = x.shape[1]

        Q = self.W_q(x)  # (batch, seq_len, d_out)
        K = self.W_k(x)  # (batch, seq_len, d_out)
        V = self.W_v(x)  # (batch, seq_len, d_out)

        d_k = K.shape[-1]

        # Scaled dot-product scores
        scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (batch, seq_len, seq_len)

        # Apply causal mask: set future positions to -inf before softmax
        scores.masked_fill_(self.mask[:seq_len, :seq_len], float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ V  # (batch, seq_len, d_out)

        return context_vectors, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: runs multiple attention heads in parallel,
    then concatenates and projects the results.

    Each head operates on a slice of the embedding dimension:
        head_dim = d_out // num_heads

    This allows the model to jointly attend to information from
    different representation subspaces at different positions.
    """

    def __init__(self, d_in, d_out, num_heads, max_len=1024, dropout=0.0):
        """
        Args:
            d_in:      dimensionality of input embeddings
            d_out:     dimensionality of output (must be divisible by num_heads)
            num_heads: number of attention heads
            max_len:   maximum sequence length
            dropout:   dropout rate on attention weights
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # dimension per head

        # Combined Q, K, V projections (more efficient than separate per-head)
        self.W_q = nn.Linear(d_in, d_out)
        self.W_k = nn.Linear(d_in, d_out)
        self.W_v = nn.Linear(d_in, d_out)

        # Output projection: concat of heads → d_out
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Causal mask buffer
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        """
        Args:
            x: input tensor, shape (batch, seq_len, d_in)
        Returns:
            output: shape (batch, seq_len, d_out)
            attention_weights: shape (batch, num_heads, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_out)
        K = self.W_k(x)  # (batch, seq_len, d_out)
        V = self.W_v(x)  # (batch, seq_len, d_out)

        # Reshape: split d_out into (num_heads, head_dim), then transpose
        # (batch, seq_len, d_out) → (batch, seq_len, num_heads, head_dim)
        #                         → (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention per head
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # → (batch, num_heads, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)

        # Apply causal mask
        scores.masked_fill_(self.mask[:seq_len, :seq_len], float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # → (batch, num_heads, seq_len, head_dim)
        context = attention_weights @ V

        # Concatenate heads: transpose back and reshape
        # (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads, head_dim)
        #                                       → (batch, seq_len, d_out)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # Final output projection
        output = self.out_proj(context)  # (batch, seq_len, d_out)

        return output, attention_weights


# --- Demo ---

if __name__ == "__main__":
    torch.manual_seed(42)

    # Small example: batch=1, seq_len=4, d_in=8
    batch_size, seq_len, d_in, d_out = 1, 4, 8, 8
    x = torch.randn(batch_size, seq_len, d_in)

    print("=" * 60)
    print("1. Simple Self-Attention (no mask, no dropout)")
    print("=" * 60)
    simple = SimpleAttention(d_in, d_out)
    ctx, weights = simple(x)
    print(f"   Input shape:            {x.shape}")
    print(f"   Context vectors shape:  {ctx.shape}")
    print(f"   Attention weights shape: {weights.shape}")
    print(f"   Weights (row 0): {weights[0, 0].detach()}")
    print(f"   → Each token attends to ALL tokens (no mask)\n")

    print("=" * 60)
    print("2. Causal Attention (with mask + dropout)")
    print("=" * 60)
    causal = CausalAttention(d_in, d_out, dropout=0.1)
    ctx, weights = causal(x)
    print(f"   Input shape:            {x.shape}")
    print(f"   Context vectors shape:  {ctx.shape}")
    print(f"   Attention weights shape: {weights.shape}")
    print(f"   Weights (row 0): {weights[0, 0].detach()}")
    print(f"   Weights (row 3): {weights[0, 3].detach()}")
    print(f"   → Token 0 only attends to itself; token 3 attends to 0-3\n")

    print("=" * 60)
    print("3. Multi-Head Attention (4 heads)")
    print("=" * 60)
    num_heads = 4
    mha = MultiHeadAttention(d_in, d_out, num_heads=num_heads, dropout=0.1)
    out, weights = mha(x)
    print(f"   Input shape:            {x.shape}")
    print(f"   Output shape:           {out.shape}")
    print(f"   Attention weights shape: {weights.shape}")
    print(f"   → {num_heads} heads, each with head_dim={d_out // num_heads}")
    print(f"   Head 0, row 0 weights: {weights[0, 0, 0].detach()}")
    print(f"   Head 1, row 0 weights: {weights[0, 1, 0].detach()}")
