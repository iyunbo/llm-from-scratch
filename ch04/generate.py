"""
Chapter 4: Text Generation Functions.

Implements autoregressive text generation strategies:
  1. Greedy decoding     — always pick the highest-probability token
  2. Temperature scaling — control randomness of sampling
  3. Top-k sampling      — sample from the k most likely tokens

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 4.
"""

import torch
import torch.nn.functional as F
from ch04.gpt_model import GPTModel


def generate_greedy(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Greedy decoding: always pick the token with the highest logit.

    Deterministic — same input always produces the same output.
    Fast but can be repetitive and lack diversity.

    Args:
        model:          GPT model instance
        idx:            initial token indices, shape (batch, seq_len)
        max_new_tokens: number of new tokens to generate

    Returns:
        Extended token sequence, shape (batch, seq_len + max_new_tokens)
    """
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to max sequence length if needed
            idx_crop = idx[:, -model.cfg.max_seq_len:]  # (batch, ≤ max_seq_len)

            # Forward pass → logits
            logits = model(idx_crop)  # (batch, seq_len, vocab_size)

            # Take logits for the last position only
            logits_last = logits[:, -1, :]  # (batch, vocab_size)

            # Greedy: pick the argmax
            next_token = logits_last.argmax(dim=-1, keepdim=True)  # (batch, 1)

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=-1)  # (batch, seq_len + 1)

    return idx


def generate_topk(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Top-k sampling with temperature scaling.

    Temperature controls randomness:
        - T < 1.0 → sharper distribution (more confident, less diverse)
        - T = 1.0 → original distribution
        - T > 1.0 → flatter distribution (more random, more diverse)

    Top-k limits the candidate pool:
        - Only the k tokens with highest logits are considered
        - Prevents sampling from the long tail of unlikely tokens
        - k=1 is equivalent to greedy decoding

    Args:
        model:          GPT model instance
        idx:            initial token indices, shape (batch, seq_len)
        max_new_tokens: number of new tokens to generate
        temperature:    temperature for scaling logits (default: 1.0)
        top_k:          number of top tokens to sample from (default: 50)

    Returns:
        Extended token sequence, shape (batch, seq_len + max_new_tokens)
    """
    assert temperature > 0, "Temperature must be positive"
    assert top_k >= 1, "top_k must be at least 1"

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_crop = idx[:, -model.cfg.max_seq_len:]  # (batch, ≤ max_seq_len)

            # Forward pass
            logits = model(idx_crop)  # (batch, seq_len, vocab_size)

            # Last position logits
            logits_last = logits[:, -1, :]  # (batch, vocab_size)

            # ---- Temperature scaling ----
            logits_last = logits_last / temperature  # (batch, vocab_size)

            # ---- Top-k filtering ----
            if top_k < logits_last.size(-1):
                # Find the top-k values and their indices
                top_values, _ = torch.topk(logits_last, top_k, dim=-1)  # (batch, top_k)

                # Get the minimum value in top-k (threshold)
                min_topk = top_values[:, -1].unsqueeze(-1)  # (batch, 1)

                # Set all logits below threshold to -inf
                logits_last = logits_last.where(
                    logits_last >= min_topk,
                    torch.full_like(logits_last, float("-inf")),
                )

            # ---- Sample from the filtered distribution ----
            probs = F.softmax(logits_last, dim=-1)  # (batch, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=-1)  # (batch, seq_len + 1)

    return idx


# ──────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ch04.config import GPTConfig

    torch.manual_seed(42)

    # Use a tiny config for quick testing
    cfg = GPTConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32,
        dropout=0.0,
    )

    model = GPTModel(cfg)
    print(f"Tiny model params: {model.count_parameters():,}")

    # Starting tokens
    start = torch.tensor([[1, 2, 3]])  # (1, 3)

    print("\n--- Greedy Decoding ---")
    out = generate_greedy(model, start, max_new_tokens=10)
    print(f"Input:  {start.tolist()}")
    print(f"Output: {out.tolist()}")

    print("\n--- Top-k Sampling (T=1.0, k=10) ---")
    out = generate_topk(model, start, max_new_tokens=10, temperature=1.0, top_k=10)
    print(f"Input:  {start.tolist()}")
    print(f"Output: {out.tolist()}")

    print("\n--- Top-k Sampling (T=0.5, k=5) — more focused ---")
    out = generate_topk(model, start, max_new_tokens=10, temperature=0.5, top_k=5)
    print(f"Input:  {start.tolist()}")
    print(f"Output: {out.tolist()}")

    print("\n--- Top-k Sampling (T=2.0, k=50) — more random ---")
    out = generate_topk(model, start, max_new_tokens=10, temperature=2.0, top_k=50)
    print(f"Input:  {start.tolist()}")
    print(f"Output: {out.tolist()}")
