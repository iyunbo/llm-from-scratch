"""
Chapter 5: Evaluation Functions.

Functions for evaluating the GPT model during pretraining:
  1. calc_loss_batch     — loss on a single batch
  2. calc_loss_loader    — average loss over a dataloader
  3. calc_perplexity     — perplexity = exp(loss)
  4. evaluate_and_sample — full evaluation with text generation

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 5.
"""

import math
import torch
import torch.nn.functional as F
from ch05.utils import generate_text


# ──────────────────────────────────────────────────────────────
# 1. Loss Calculation
# ──────────────────────────────────────────────────────────────

def calc_loss_batch(
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Calculate cross-entropy loss on a single batch.

    Cross-entropy loss for next-token prediction:
        loss = -1/N Σ log P(target_i | context_i)

    This is the fundamental training objective: the model learns to
    predict the next token given all previous tokens.

    Args:
        model:        GPT model
        input_batch:  input token IDs, shape (batch, seq_len)
        target_batch: target token IDs, shape (batch, seq_len)
        device:       device to compute on

    Returns:
        Scalar loss tensor
    """
    input_batch = input_batch.to(device)    # (batch, seq_len)
    target_batch = target_batch.to(device)  # (batch, seq_len)

    # Forward pass: get logits for all positions
    logits = model(input_batch)  # (batch, seq_len, vocab_size)

    # Reshape for cross-entropy:
    #   logits:  (batch * seq_len, vocab_size)
    #   targets: (batch * seq_len,)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # (batch * seq_len, vocab_size)
        target_batch.view(-1),              # (batch * seq_len,)
    )

    return loss  # scalar


def calc_loss_loader(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    max_batches: int = None,
) -> float:
    """
    Calculate average loss over an entire dataloader.

    Args:
        model:       GPT model
        data_loader: PyTorch DataLoader yielding (input, target) pairs
        device:      device to compute on
        max_batches: optional limit on number of batches to evaluate

    Returns:
        Average loss (float)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ──────────────────────────────────────────────────────────────
# 2. Perplexity
# ──────────────────────────────────────────────────────────────

def calc_perplexity(avg_loss: float) -> float:
    """
    Calculate perplexity from average cross-entropy loss.

    Perplexity = exp(loss)

    Intuition:
        - Perplexity ≈ "how many tokens is the model choosing between"
        - Perfect prediction → loss = 0 → perplexity = 1
        - Random over vocab_size → perplexity ≈ vocab_size
        - Lower is better

    For GPT-2 (vocab_size = 50257):
        - Random baseline perplexity ≈ 50,257
        - Good LLM perplexity ≈ 10-30 on standard benchmarks

    Args:
        avg_loss: average cross-entropy loss

    Returns:
        Perplexity value
    """
    return math.exp(avg_loss)


# ──────────────────────────────────────────────────────────────
# 3. Full Evaluation with Text Generation
# ──────────────────────────────────────────────────────────────

def evaluate_and_sample(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    prompt: str = "Every effort moves you",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 25,
) -> dict:
    """
    Full evaluation: compute train/val loss and generate sample text.

    Used during training to monitor progress:
        - Train loss: should decrease (model is learning)
        - Val loss: should decrease but may diverge (overfitting)
        - Generated text: should become more coherent over time

    Args:
        model:          GPT model
        train_loader:   training data loader
        val_loader:     validation data loader
        device:         device to compute on
        prompt:         prompt for text generation
        max_new_tokens: tokens to generate
        temperature:    sampling temperature
        top_k:          top-k filtering

    Returns:
        Dictionary with train_loss, val_loss, perplexity, generated_text
    """
    train_loss = calc_loss_loader(model, train_loader, device, max_batches=5)
    val_loss = calc_loss_loader(model, val_loader, device)
    perplexity = calc_perplexity(val_loss)

    # Generate sample text
    model.eval()
    generated = generate_text(
        model, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "perplexity": perplexity,
        "generated_text": generated,
    }
