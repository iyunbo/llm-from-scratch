"""
Chapter 5: Training Loop for GPT Pretraining.

Complete training loop with:
  1. Cross-entropy loss (next-token prediction)
  2. AdamW optimizer (with weight decay)
  3. Learning rate warmup + cosine decay
  4. Gradient clipping
  5. Periodic evaluation and text generation
  6. Checkpoint saving

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 5.
"""

import torch
from ch05.evaluate import calc_loss_batch, evaluate_and_sample
from ch05.utils import get_lr, save_checkpoint


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    max_lr: float = 5e-4,
    min_lr: float = 1e-5,
    warmup_frac: float = 0.1,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    eval_every: int = 50,
    sample_every: int = 100,
    checkpoint_dir: str = "ch05/checkpoints",
    device: str = "cpu",
    prompt: str = "Every effort moves you",
) -> dict:
    """
    Full pretraining loop for the GPT model.

    Training objective: next-token prediction (causal language modeling).
    The model learns to predict token[t+1] given tokens[0:t].

    AdamW vs Adam:
        Adam:  updates weights with adaptive learning rates + momentum
        AdamW: same as Adam, but applies weight decay SEPARATELY
               (decoupled weight decay regularization)
        Why?   In Adam, weight decay is mixed with gradient updates,
               which makes it less effective. AdamW fixes this.

    Gradient clipping:
        - Limits the norm of gradients to prevent exploding gradients
        - Especially important early in training with random weights
        - max_norm=1.0 is standard for LLMs

    Args:
        model:          GPT model to train
        train_loader:   training DataLoader
        val_loader:     validation DataLoader
        num_epochs:     number of training epochs
        max_lr:         peak learning rate
        min_lr:         minimum learning rate (cosine decay floor)
        warmup_frac:    fraction of total steps for warmup
        weight_decay:   weight decay coefficient for AdamW
        max_grad_norm:  gradient clipping threshold
        eval_every:     evaluate every N steps
        sample_every:   generate sample text every N steps
        checkpoint_dir: directory to save checkpoints
        device:         device to train on
        prompt:         prompt for sample generation

    Returns:
        Dictionary with training history:
            train_losses, val_losses, lrs, perplexities
    """
    model.to(device)
    model.train()

    # ---- Optimizer: AdamW with weight decay ----
    # Weight decay is NOT applied to bias and LayerNorm params
    # (they don't benefit from regularization)
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

    # ---- Calculate total steps and warmup ----
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_frac)

    print(f"{'=' * 60}")
    print(f"  Pretraining Configuration")
    print(f"{'=' * 60}")
    print(f"  Device:          {device}")
    print(f"  Parameters:      {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs:          {num_epochs}")
    print(f"  Batches/epoch:   {len(train_loader)}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Warmup steps:    {warmup_steps}")
    print(f"  Max LR:          {max_lr}")
    print(f"  Min LR:          {min_lr}")
    print(f"  Weight decay:    {weight_decay}")
    print(f"  Grad clip norm:  {max_grad_norm}")
    print(f"{'=' * 60}\n")

    # ---- Training history ----
    history = {
        "train_losses": [],
        "val_losses": [],
        "lrs": [],
        "perplexities": [],
        "steps": [],
    }

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # ---- Update learning rate (per step) ----
            lr = get_lr(global_step, max_lr, min_lr, warmup_steps, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # ---- Forward pass ----
            loss = calc_loss_batch(model, input_batch, target_batch, device)

            # ---- Backward pass ----
            optimizer.zero_grad()
            loss.backward()

            # ---- Gradient clipping ----
            # Prevents exploding gradients by scaling down if norm > max_grad_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # ---- Optimizer step ----
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # ---- Periodic evaluation ----
            if global_step % eval_every == 0:
                result = evaluate_and_sample(
                    model, train_loader, val_loader,
                    device=device, prompt=prompt,
                    max_new_tokens=50, temperature=0.8, top_k=25,
                )

                history["train_losses"].append(result["train_loss"])
                history["val_losses"].append(result["val_loss"])
                history["perplexities"].append(result["perplexity"])
                history["lrs"].append(lr)
                history["steps"].append(global_step)

                print(f"  Step {global_step:4d}/{total_steps} | "
                      f"LR: {lr:.2e} | "
                      f"Train loss: {result['train_loss']:.4f} | "
                      f"Val loss: {result['val_loss']:.4f} | "
                      f"PPL: {result['perplexity']:.1f}")

                # ---- Show generated text at sample intervals ----
                if global_step % sample_every == 0:
                    print(f"  📝 Generated: {result['generated_text'][:200]}")
                    print()

                model.train()  # Back to training mode

        # ---- End of epoch ----
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n  ✅ Epoch {epoch + 1}/{num_epochs} done | "
              f"Avg train loss: {avg_epoch_loss:.4f}\n")

    # ---- Save final checkpoint ----
    final_result = evaluate_and_sample(
        model, train_loader, val_loader,
        device=device, prompt=prompt,
    )
    save_checkpoint(
        model, optimizer,
        epoch=num_epochs,
        step=global_step,
        train_loss=final_result["train_loss"],
        val_loss=final_result["val_loss"],
        path=f"{checkpoint_dir}/final.pt",
    )

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Final train loss: {final_result['train_loss']:.4f}")
    print(f"  Final val loss:   {final_result['val_loss']:.4f}")
    print(f"  Final perplexity: {final_result['perplexity']:.1f}")
    print(f"{'=' * 60}")

    return history
