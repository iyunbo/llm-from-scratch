"""
Chapter 5: Pretraining Entry Point.

Puts it all together:
  1. Load text data ("The Verdict" by Edith Wharton)
  2. Create train/val dataloaders (ch02)
  3. Initialize GPT model (ch04) with small config
  4. Run pretraining loop
  5. Show before/after text generation comparison

Usage:
    python -m ch05.pretrain

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 5.
"""

import os
import sys
import torch
import tiktoken

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ch02.dataloader import create_dataloader
from ch04.config import GPTConfig
from ch04.gpt_model import GPTModel
from ch05.train import train_model
from ch05.utils import generate_text


# ──────────────────────────────────────────────────────────────
# Small model config (CPU-friendly, trains in minutes)
# ──────────────────────────────────────────────────────────────

SMALL_CONFIG = GPTConfig(
    vocab_size=50257,     # GPT-2 BPE vocab (keep full for proper tokenization)
    d_model=128,          # small embedding dim (GPT-2 uses 768)
    n_heads=4,            # fewer heads (GPT-2 uses 12)
    n_layers=4,           # fewer layers (GPT-2 uses 12)
    max_seq_len=256,      # shorter context (GPT-2 uses 1024)
    dropout=0.1,          # standard dropout
)


def load_text(path: str) -> str:
    """Load raw text from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_train_val_loaders(
    text: str,
    tokenizer,
    max_len: int = 256,
    stride: int = 128,
    batch_size: int = 4,
    val_ratio: float = 0.15,
):
    """
    Split text into train/val and create dataloaders.

    We split at the character level (not token level) for simplicity.
    The validation set is the last val_ratio fraction of the text.

    Args:
        text:       raw text string
        tokenizer:  tiktoken encoding
        max_len:    context window size
        stride:     sliding window stride
        batch_size: batch size
        val_ratio:  fraction of text for validation

    Returns:
        (train_loader, val_loader)
    """
    split_idx = int(len(text) * (1 - val_ratio))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"  Text length:    {len(text):,} characters")
    print(f"  Train split:    {len(train_text):,} chars")
    print(f"  Val split:      {len(val_text):,} chars")

    train_loader = create_dataloader(
        train_text, tokenizer,
        max_len=max_len, stride=stride,
        batch_size=batch_size, shuffle=True,
    )
    # Val uses smaller batch and overlapping stride to ensure ≥1 batch
    val_loader = create_dataloader(
        val_text, tokenizer,
        max_len=max_len, stride=max_len // 2,
        batch_size=2, shuffle=False,
    )

    print(f"  Train batches:  {len(train_loader)}")
    print(f"  Val batches:    {len(val_loader)}")

    return train_loader, val_loader


def main():
    """Run the full pretraining pipeline."""
    torch.manual_seed(42)

    # ---- Device ----
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")

    # ---- Load data ----
    print(f"\n📖 Loading text data...")
    text_path = os.path.join(os.path.dirname(__file__), "..", "ch02", "the-verdict.txt")
    text = load_text(text_path)

    # ---- Tokenizer ----
    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(text))
    print(f"  Total tokens:   {total_tokens:,}")

    # ---- Create dataloaders ----
    print(f"\n📦 Creating dataloaders...")
    train_loader, val_loader = create_train_val_loaders(
        text, tokenizer,
        max_len=SMALL_CONFIG.max_seq_len,
        stride=SMALL_CONFIG.max_seq_len // 2,  # overlapping for more training data
        batch_size=4,
        val_ratio=0.15,
    )

    # ---- Initialize model ----
    print(f"\n🧠 Initializing model...")
    model = GPTModel(SMALL_CONFIG)
    print(f"  Config:         d_model={SMALL_CONFIG.d_model}, "
          f"n_heads={SMALL_CONFIG.n_heads}, "
          f"n_layers={SMALL_CONFIG.n_layers}")
    print(f"  Parameters:     {model.count_parameters():,}")

    # ---- Move model to device ----
    model.to(device)

    # ---- Generate BEFORE training (random weights) ----
    prompt = "Every effort moves you"
    print(f"\n🎲 Text generation BEFORE training (random weights):")
    before_text = generate_text(model, prompt, max_new_tokens=50,
                                temperature=1.0, top_k=50, device=device)
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Output: \"{before_text}\"")

    # ---- Train ----
    print(f"\n🔥 Starting pretraining...")
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=30,
        max_lr=5e-4,
        min_lr=1e-5,
        warmup_frac=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        eval_every=10,
        sample_every=50,
        checkpoint_dir="ch05/checkpoints",
        device=device,
        prompt=prompt,
    )

    # ---- Generate AFTER training ----
    print(f"\n✨ Text generation AFTER training:")
    after_text = generate_text(model, prompt, max_new_tokens=50,
                               temperature=0.8, top_k=25, device=device)
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Output: \"{after_text}\"")

    # ---- Comparison ----
    print(f"\n{'=' * 60}")
    print(f"  Before vs After Comparison")
    print(f"{'=' * 60}")
    print(f"  BEFORE: {before_text[:150]}")
    print(f"  AFTER:  {after_text[:150]}")
    print(f"{'=' * 60}")

    return history


if __name__ == "__main__":
    main()
