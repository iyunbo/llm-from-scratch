"""
Chapter 2: Token and Positional Embeddings.

Demonstrates how raw token IDs get transformed into dense vectors
that a transformer can process:
  1. Token embedding  — maps each token ID to a learned vector
  2. Positional embedding — maps each position to a learned vector
  3. Input = token_embed + pos_embed
"""

import torch
import torch.nn as nn


class TokenPositionalEmbedding(nn.Module):
    """
    Combined token + positional embedding layer.

    Args:
        vocab_size: number of tokens in the vocabulary
        embed_dim: dimensionality of each embedding vector
        max_len: maximum sequence length (context window)
    """

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        """
        Args:
            x: tensor of token IDs, shape (batch_size, seq_len)
        Returns:
            tensor of shape (batch_size, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        # Token embeddings
        tok_embeddings = self.tok_emb(x)
        # Positional embeddings: positions 0, 1, ..., seq_len-1
        positions = torch.arange(seq_len, device=x.device)
        pos_embeddings = self.pos_emb(positions)
        # Combine: broadcast pos_embeddings across batch dimension
        return tok_embeddings + pos_embeddings


# --- Demo ---

if __name__ == "__main__":
    import tiktoken

    # Config (GPT-2 small-ish)
    VOCAB_SIZE = 50257  # GPT-2 vocab size
    EMBED_DIM = 256
    MAX_LEN = 1024

    # Tokenize a sample
    enc = tiktoken.get_encoding("gpt2")
    text = "Every journey begins with a single step."
    token_ids = enc.encode(text)
    print(f"Text: {text}")
    print(f"Token IDs: {token_ids}")

    # Create a batch (batch_size=1)
    batch = torch.tensor([token_ids])
    print(f"Input shape: {batch.shape}")  # (1, seq_len)

    # Embedding layer
    embedding = TokenPositionalEmbedding(VOCAB_SIZE, EMBED_DIM, MAX_LEN)
    output = embedding(batch)
    print(f"Output shape: {output.shape}")  # (1, seq_len, embed_dim)

    # Peek at the first token's embedding vector
    print(f"\nFirst token embedding (first 8 dims):")
    print(output[0, 0, :8].detach())

    print(f"\nToken embedding and positional embedding are added element-wise.")
    print(f"This gives the model both 'what token' and 'where in sequence' information.")
