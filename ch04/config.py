"""
Chapter 4: GPT Model Configuration.

Defines the configuration dataclass for GPT models.
Default values match GPT-2 Small (124M parameters).

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 4.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration for a GPT model.

    Default values correspond to GPT-2 Small (124M params):
        - 12 transformer layers
        - 12 attention heads
        - 768-dimensional embeddings
        - 50,257 BPE vocabulary (GPT-2 tokenizer)
        - 1024 max sequence length

    Parameter count breakdown (GPT-2 Small):
        Token embeddings:      50,257 × 768    = 38,597,376
        Position embeddings:   1,024 × 768     =    786,432
        12 × TransformerBlock:                  = 85,054,464
            - LayerNorm (×2):  768 × 2 × 2     =      3,072
            - MHA (QKV + out): 768×768×4        =  2,359,296
            - FF (up + down):  768×3072×2       =  4,722,432
            ─────────────────────────────────────
            Per block total:                    =  7,084,800
        Final LayerNorm:       768 × 2         =      1,536
        Output head (tied):    768 × 50,257    = 38,597,376
        ─────────────────────────────────────────
        Total:                                 ≈ 163M (untied)
        With weight tying:                     ≈ 124M

    Note: GPT-2 ties token embedding weights with the output head,
    reducing the effective parameter count by ~38.6M.
    We keep them separate here for clarity.
    """

    vocab_size: int = 50_257      # GPT-2 BPE vocabulary size
    d_model: int = 768            # Embedding dimension / hidden size
    n_heads: int = 12             # Number of attention heads
    n_layers: int = 12            # Number of transformer blocks
    max_seq_len: int = 1024       # Maximum sequence length (context window)
    dropout: float = 0.1          # Dropout rate (attention + residual)


# Predefined configurations for quick experimentation
GPT2_SMALL = GPTConfig()  # 124M — default

GPT2_MEDIUM = GPTConfig(
    d_model=1024,
    n_heads=16,
    n_layers=24,
)  # 350M

GPT2_LARGE = GPTConfig(
    d_model=1280,
    n_heads=20,
    n_layers=36,
)  # 774M

GPT2_XL = GPTConfig(
    d_model=1600,
    n_heads=25,
    n_layers=48,
)  # 1.5B


if __name__ == "__main__":
    cfg = GPT2_SMALL
    print("GPT-2 Small Configuration:")
    print(f"  vocab_size:   {cfg.vocab_size:,}")
    print(f"  d_model:      {cfg.d_model}")
    print(f"  n_heads:      {cfg.n_heads}")
    print(f"  n_layers:     {cfg.n_layers}")
    print(f"  max_seq_len:  {cfg.max_seq_len}")
    print(f"  dropout:      {cfg.dropout}")
    print(f"  head_dim:     {cfg.d_model // cfg.n_heads}")
