"""
Chapter 6: Classifier Training Loop.

Fine-tuning pipeline for GPT-based text classification:
  1. Load pretrained GPT backbone (or use random init)
  2. Attach classification head
  3. Train with CrossEntropyLoss
  4. Track accuracy on train/val/test sets
  5. Demonstrate before/after classification quality

Key differences from pretraining (ch05):
  - Loss: CrossEntropyLoss on class labels (not next-token prediction)
  - Metric: Accuracy (not perplexity)
  - Data: (text, label) pairs (not raw text chunks)
  - Learning rate: typically lower for fine-tuning (we're refining, not learning from scratch)
  - Epochs: fewer needed (pretrained features are already useful)

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 6.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ch04.config import GPTConfig
from ch05.pretrain import SMALL_CONFIG
from ch06.classifier import GPTClassifier
from ch06.dataset import create_classification_loaders, SPAM_DATASET


# ──────────────────────────────────────────────────────────────
# 1. Accuracy Calculation
# ──────────────────────────────────────────────────────────────

def calc_accuracy(model: nn.Module, data_loader, device: str = "cpu") -> float:
    """
    Calculate classification accuracy over a dataloader.

    Accuracy = correct predictions / total predictions

    Args:
        model:       classifier model
        data_loader: DataLoader yielding (token_ids, labels)
        device:      device

    Returns:
        Accuracy as a float in [0, 1]
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for token_ids, labels in data_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            logits = model(token_ids)                    # (batch, num_classes)
            predictions = logits.argmax(dim=-1)          # (batch,)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / max(total, 1)


def calc_classification_loss(
    model: nn.Module, data_loader, device: str = "cpu"
) -> float:
    """
    Calculate average CrossEntropyLoss over a dataloader.

    Args:
        model:       classifier model
        data_loader: DataLoader yielding (token_ids, labels)
        device:      device

    Returns:
        Average loss as float
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for token_ids, labels in data_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            logits = model(token_ids)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ──────────────────────────────────────────────────────────────
# 2. Fine-Tuning Training Loop
# ──────────────────────────────────────────────────────────────

def train_classifier(
    model: GPTClassifier,
    train_loader,
    val_loader,
    num_epochs: int = 20,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    device: str = "cpu",
) -> dict:
    """
    Fine-tuning loop for text classification.

    Key design choices:
        - Lower learning rate than pretraining (5e-5 vs 5e-4)
          → we want to REFINE, not destroy pretrained features
        - CrossEntropyLoss instead of next-token prediction loss
        - Track accuracy as the primary metric
        - No LR warmup needed (already pretrained, stable gradients)

    Args:
        model:        GPTClassifier instance
        train_loader: training DataLoader
        val_loader:   validation DataLoader
        num_epochs:   number of fine-tuning epochs
        lr:           learning rate (lower than pretraining!)
        weight_decay: L2 regularization
        device:       device

    Returns:
        Training history dict
    """
    model.to(device)

    # Only optimize trainable parameters (frozen params excluded)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    print(f"{'=' * 60}")
    print(f"  Fine-Tuning Configuration")
    print(f"{'=' * 60}")
    print(f"  Device:              {device}")
    print(f"  Total params:        {model.count_total_parameters():,}")
    print(f"  Trainable params:    {model.count_trainable_parameters():,}")
    trainable_pct = model.count_trainable_parameters() / model.count_total_parameters() * 100
    print(f"  Trainable ratio:     {trainable_pct:.1f}%")
    print(f"  Epochs:              {num_epochs}")
    print(f"  Learning rate:       {lr}")
    print(f"  Num classes:         {model.num_classes}")
    print(f"{'=' * 60}\n")

    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for token_ids, labels in train_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            # Forward
            logits = model(token_ids)              # (batch, num_classes)
            loss = F.cross_entropy(logits, labels)  # scalar

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # ---- Evaluate at end of each epoch ----
        avg_train_loss = epoch_loss / max(num_batches, 1)
        val_loss = calc_classification_loss(model, val_loader, device)
        train_acc = calc_accuracy(model, train_loader, device)
        val_acc = calc_accuracy(model, val_loader, device)

        history["train_losses"].append(avg_train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        print(f"  Epoch {epoch + 1:2d}/{num_epochs} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | "
              f"Train acc: {train_acc:.1%} | "
              f"Val acc: {val_acc:.1%}")

    return history


# ──────────────────────────────────────────────────────────────
# 3. Inference Helper
# ──────────────────────────────────────────────────────────────

def classify_text(
    model: GPTClassifier,
    text: str,
    max_len: int = 64,
    device: str = "cpu",
) -> tuple[int, list[float]]:
    """
    Classify a single text string.

    Args:
        model:   trained GPTClassifier
        text:    input text to classify
        max_len: max sequence length (must match training)
        device:  device

    Returns:
        (predicted_class, probabilities)
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(text)

    # Truncate + pad
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    token_ids = token_ids + [50256] * (max_len - len(token_ids))

    # Forward pass
    model.eval()
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)  # (1, max_len)
    with torch.no_grad():
        logits = model(idx)                          # (1, num_classes)
        probs = F.softmax(logits, dim=-1)            # (1, num_classes)

    predicted = logits.argmax(dim=-1).item()
    return predicted, probs[0].tolist()


# ──────────────────────────────────────────────────────────────
# 4. Main Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    """Full fine-tuning pipeline: load → train → evaluate → demo."""
    torch.manual_seed(42)

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")

    # ---- Create dataset ----
    print(f"\n📦 Creating classification dataset...")
    train_loader, val_loader, test_loader = create_classification_loaders(
        max_len=64, batch_size=8
    )

    # ---- Create classifier ----
    print(f"\n🧠 Creating GPT classifier...")
    num_classes = 2  # spam / ham
    model = GPTClassifier(SMALL_CONFIG, num_classes=num_classes, unfreeze_last_n=-1)

    # ---- Try to load pretrained backbone ----
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "ch05", "checkpoints", "final.pt")
    if os.path.exists(checkpoint_path):
        print(f"  Loading pretrained backbone...")
        model.load_pretrained_backbone(checkpoint_path, device=device)
    else:
        print(f"  ⚠️  No pretrained checkpoint found. Using random initialization.")
        print(f"     (Run ch05 pretraining first for better results)")

    # ---- Evaluate BEFORE fine-tuning ----
    model.to(device)
    before_acc = calc_accuracy(model, test_loader, device)
    print(f"\n  📊 Test accuracy BEFORE fine-tuning: {before_acc:.1%}")
    print(f"     (Random chance for 2 classes: 50%)")

    # ---- Fine-tune ----
    print(f"\n🔥 Starting fine-tuning...")
    history = train_classifier(
        model, train_loader, val_loader,
        num_epochs=15,
        lr=5e-4,
        device=device,
    )

    # ---- Evaluate AFTER fine-tuning ----
    test_acc = calc_accuracy(model, test_loader, device)
    print(f"\n  📊 Test accuracy AFTER fine-tuning: {test_acc:.1%}")

    # ---- Demo: classify individual messages ----
    print(f"\n{'=' * 60}")
    print(f"  Live Classification Demo")
    print(f"{'=' * 60}")

    test_messages = [
        "Congratulations! You won a free cruise. Click to claim!",
        "Hey, want to grab lunch tomorrow?",
        "URGENT: Verify your account or it will be suspended!",
        "The meeting has been rescheduled to Thursday.",
        "Make $10000 a day with this simple trick!",
        "Thanks for the birthday wishes!",
    ]

    label_names = {0: "ham ✉️", 1: "spam 🚫"}
    for msg in test_messages:
        pred, probs = classify_text(model, msg, device=device)
        confidence = max(probs) * 100
        print(f"\n  [{label_names[pred]}] ({confidence:.0f}%) {msg[:60]}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  Before fine-tuning: {before_acc:.1%}")
    print(f"  After fine-tuning:  {test_acc:.1%}")
    print(f"  Improvement:        {test_acc - before_acc:+.1%}")
    print(f"{'=' * 60}")

    return history


if __name__ == "__main__":
    main()
