"""
Chapter 5: Training Utilities.

Helper functions for pretraining:
  1. Learning rate scheduler (linear warmup + cosine decay)
  2. Text generation helper for monitoring training quality
  3. Checkpoint save/load functions

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 5.
"""

import math
import os
import torch
import tiktoken
from ch04.generate import generate_topk


# ──────────────────────────────────────────────────────────────
# 1. Learning Rate Scheduler (Warmup + Cosine Decay)
# ──────────────────────────────────────────────────────────────

def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """
    Compute learning rate for a given training step.

    Schedule:
        1. Linear warmup:   0 → max_lr over warmup_steps
        2. Cosine decay:    max_lr → min_lr over remaining steps

    Why warmup?
        - At the start, weights are random → gradients are noisy
        - Large LR + noisy gradients = unstable training
        - Warmup lets the model "settle" before applying full LR

    Why cosine decay?
        - Smoothly reduces LR as training converges
        - Better than step decay (no sudden jumps)
        - Widely used in LLM pretraining (GPT-3, LLaMA, etc.)

    Args:
        step:         current training step (0-indexed)
        max_lr:       peak learning rate (reached after warmup)
        min_lr:       minimum learning rate (floor of cosine decay)
        warmup_steps: number of warmup steps
        total_steps:  total number of training steps

    Returns:
        Learning rate for the given step
    """
    # Phase 1: Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Phase 2: Cosine decay
    # Map step to [0, 1] range within the decay phase
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)  # clamp

    # Cosine decay: cos goes from 1 → -1, we map to max_lr → min_lr
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────
# 2. Text Generation Helper
# ──────────────────────────────────────────────────────────────

def generate_text(
    model: torch.nn.Module,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cpu",
) -> str:
    """
    Generate text from a prompt using the model.
    Useful for monitoring training progress.

    Args:
        model:          GPT model instance
        prompt:         text prompt to start generation
        max_new_tokens: number of tokens to generate
        temperature:    sampling temperature (lower = more deterministic)
        top_k:          top-k sampling parameter
        device:         device to run on

    Returns:
        Generated text string (prompt + generated tokens)
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(prompt)
    idx = torch.tensor([token_ids], device=device)  # (1, seq_len)

    # Generate
    output_ids = generate_topk(
        model, idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )  # (1, seq_len + max_new_tokens)

    # Decode back to text
    return tokenizer.decode(output_ids[0].tolist())


# ──────────────────────────────────────────────────────────────
# 3. Checkpoint Save / Load
# ──────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float,
    path: str,
):
    """
    Save a training checkpoint.

    Saves:
        - Model state dict (weights)
        - Optimizer state dict (momentum, etc.)
        - Training metadata (epoch, step, losses)

    Args:
        model:      the GPT model
        optimizer:  the optimizer
        epoch:      current epoch number
        step:       current global step
        train_loss: latest training loss
        val_loss:   latest validation loss
        path:       file path to save to
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)
    print(f"  💾 Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
) -> dict:
    """
    Load a training checkpoint.

    Args:
        path:      file path to load from
        model:     the GPT model (weights will be loaded in-place)
        optimizer: optional optimizer (state will be loaded if provided)

    Returns:
        Dictionary with metadata: epoch, step, train_loss, val_loss
    """
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"  📂 Checkpoint loaded: {path}")
    print(f"     Epoch {checkpoint['epoch']}, Step {checkpoint['step']}")
    print(f"     Train loss: {checkpoint['train_loss']:.4f}, Val loss: {checkpoint['val_loss']:.4f}")

    return {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "train_loss": checkpoint["train_loss"],
        "val_loss": checkpoint["val_loss"],
    }
