"""
Chapter 7: Instruction Tuning Training Loop.

Fine-tunes a GPT model to follow instructions using:
  1. Instruction dataset (Alpaca format)
  2. Cross-entropy loss on the full sequence (simplified approach)
  3. AdamW optimizer with cosine LR schedule
  4. Evaluation with instruction-following examples

Key difference from pretraining (ch05):
  - Data is instruction-response pairs, not raw text
  - Much smaller dataset, fewer epochs
  - Lower learning rate (but 5e-4 for randomly initialized models)
  - Goal: teach the model to follow a prompt template

Advanced approach (not implemented here):
  - Mask instruction tokens so loss is only computed on response tokens
  - This prevents the model from "wasting capacity" memorizing instructions

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 7.
"""

import sys
import os
import torch
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ch04.gpt_model import GPTModel
from ch04.config import GPTConfig
from ch04.generate import generate_topk
from ch05.utils import get_lr, save_checkpoint
from ch07.instruction_dataset import (
    create_dataloaders,
    format_instruction,
    INSTRUCTIONS,
)


# ──────────────────────────────────────────────────────────────
# 1. Loss Calculation (full sequence)
# ──────────────────────────────────────────────────────────────

def calc_loss_batch(
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Cross-entropy loss on the full sequence (instruction + response).

    This is the simplified approach: we compute loss on ALL tokens,
    including the instruction part. The model learns both to understand
    the instruction format and to generate appropriate responses.

    A more sophisticated approach would mask the instruction tokens
    so the model only learns to predict response tokens. This is what
    production systems like LLaMA-2 do, but adds complexity.

    Args:
        model:        GPT model
        input_batch:  input token IDs (batch, seq_len)
        target_batch: target token IDs (batch, seq_len)
        device:       device

    Returns:
        Scalar loss tensor
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)  # (batch, seq_len, vocab_size)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_batch.view(-1),
    )
    return loss


def calc_loss_loader(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    max_batches: int = None,
) -> float:
    """Average loss over a dataloader."""
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
# 2. Generate Response to an Instruction
# ──────────────────────────────────────────────────────────────

def generate_response(
    model: torch.nn.Module,
    instruction: str,
    tokenizer,
    device: str = "cpu",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 25,
) -> str:
    """
    Generate a response to an instruction using the fine-tuned model.

    Formats the instruction into the Alpaca template, then generates
    tokens autoregressively. Extracts only the response part.

    Args:
        model:          fine-tuned GPT model
        instruction:    user's instruction text
        tokenizer:      tiktoken tokenizer
        device:         device
        max_new_tokens: max tokens to generate
        temperature:    sampling temperature
        top_k:          top-k sampling

    Returns:
        Generated response text
    """
    # Format as prompt (without the response part)
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    token_ids = tokenizer.encode(prompt, allowed_special=set())
    idx = torch.tensor([token_ids], device=device)

    # IMPORTANT: model must be on the correct device before generation!
    model.to(device)
    model.eval()

    with torch.no_grad():
        output_ids = generate_topk(
            model, idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode full output and extract response
    full_text = tokenizer.decode(output_ids[0].tolist())

    # Extract the part after "### Response:\n"
    if "### Response:\n" in full_text:
        response = full_text.split("### Response:\n")[-1].strip()
    else:
        response = full_text[len(prompt):].strip()

    return response


# ──────────────────────────────────────────────────────────────
# 3. Training Loop
# ──────────────────────────────────────────────────────────────

def train_instruction_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    tokenizer,
    num_epochs: int = 15,
    max_lr: float = 5e-4,
    min_lr: float = 1e-5,
    warmup_frac: float = 0.1,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    eval_every: int = 20,
    device: str = "cpu",
    checkpoint_dir: str = "ch07/checkpoints",
) -> dict:
    """
    Instruction tuning training loop.

    Similar to ch05's pretraining loop, but adapted for instruction tuning:
        - Smaller dataset → fewer steps per epoch
        - More epochs to compensate
        - Evaluation includes instruction-following examples
        - Learning rate 5e-4 (randomly initialized model)

    Args:
        model:          GPT model to fine-tune
        train_loader:   training DataLoader (instruction pairs)
        val_loader:     validation DataLoader
        tokenizer:      tiktoken tokenizer for text generation
        num_epochs:     number of training epochs
        max_lr:         peak learning rate
        min_lr:         minimum learning rate
        warmup_frac:    fraction of steps for warmup
        weight_decay:   AdamW weight decay
        max_grad_norm:  gradient clipping threshold
        eval_every:     evaluate every N steps
        device:         device to train on
        checkpoint_dir: where to save checkpoints

    Returns:
        Training history dict
    """
    model.to(device)
    model.train()

    # ---- Optimizer: AdamW ----
    no_decay = {"bias", "gamma", "beta"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=max_lr)

    # ---- LR Schedule ----
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_frac)
    print(f"\n🎯 Instruction Tuning")
    print(f"   Epochs: {num_epochs} | Steps/epoch: {len(train_loader)} | Total: {total_steps}")
    print(f"   LR: {max_lr} → {min_lr} | Warmup: {warmup_steps} steps")
    print(f"   Device: {device}\n")

    # ---- Training ----
    history = {"train_losses": [], "val_losses": [], "lrs": []}
    global_step = 0
    best_val_loss = float("inf")

    # Test instructions for monitoring progress
    test_instructions = [
        "What is the capital of Japan?",
        "Write a haiku about snow.",
        "Explain what gravity is.",
    ]

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Update learning rate
            lr = get_lr(global_step, max_lr, min_lr, warmup_steps, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward + backward
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            history["train_losses"].append(loss.item())
            history["lrs"].append(lr)
            global_step += 1

            # ---- Periodic Evaluation ----
            if global_step % eval_every == 0:
                val_loss = calc_loss_loader(model, val_loader, device)
                history["val_losses"].append(val_loss)

                print(f"  Step {global_step:4d} | "
                      f"Train loss: {loss.item():.4f} | "
                      f"Val loss: {val_loss:.4f} | "
                      f"LR: {lr:.2e}")

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, epoch, global_step,
                        loss.item(), val_loss,
                        os.path.join(checkpoint_dir, "best_instruction.pt"),
                    )

                model.train()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n📖 Epoch {epoch+1}/{num_epochs} | Avg loss: {avg_epoch_loss:.4f}")

        # Show sample responses at end of each epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("\n  💬 Sample responses:")
            for instr in test_instructions:
                resp = generate_response(
                    model, instr, tokenizer,
                    device=device, max_new_tokens=60,
                )
                print(f"    Q: {instr}")
                print(f"    A: {resp[:150]}")
                print()
            model.train()

    # Final checkpoint
    save_checkpoint(
        model, optimizer, num_epochs, global_step,
        avg_epoch_loss, best_val_loss,
        os.path.join(checkpoint_dir, "final_instruction.pt"),
    )

    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    return history


# ──────────────────────────────────────────────────────────────
# 4. Main: Run Instruction Tuning
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ---- Small model config (CPU friendly) ----
    cfg = GPTConfig(
        vocab_size=50_257,
        d_model=128,       # Small for demo
        n_heads=4,
        n_layers=4,
        max_seq_len=256,
        dropout=0.1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    print(f"📐 Model config: d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}")

    # Create model
    model = GPTModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {total_params:,}")

    # Create dataloaders
    train_loader, val_loader, tokenizer = create_dataloaders(
        max_length=256,
        batch_size=4,
    )

    # Train!
    history = train_instruction_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        num_epochs=15,
        max_lr=5e-4,
        device=device,
    )

    # Test the trained model
    print("\n" + "=" * 60)
    print("🧪 Testing instruction-tuned model:")
    print("=" * 60)

    test_questions = [
        "What is the capital of France?",
        "Translate 'goodbye' to Spanish.",
        "Write a short poem about stars.",
        "What is 25 times 4?",
        "Explain what machine learning is.",
    ]

    for q in test_questions:
        response = generate_response(
            model, q, tokenizer,
            device=device, max_new_tokens=80,
        )
        print(f"\n📝 {q}")
        print(f"🤖 {response}")
