"""
Chapter 6: Classification Dataset.

Provides a simple spam detection dataset for fine-tuning:
  1. Synthetic spam/ham messages (no external downloads needed)
  2. Tokenization with GPT-2 BPE tokenizer
  3. Padding + truncation to fixed length
  4. PyTorch Dataset with train/val/test splits

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 6.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import tiktoken


# ──────────────────────────────────────────────────────────────
# 1. Synthetic Spam Dataset
# ──────────────────────────────────────────────────────────────

# Labels: 0 = ham (not spam), 1 = spam
SPAM_DATASET = [
    # ---- Spam (label=1) ----
    ("Congratulations! You've won a $1000 gift card. Click here to claim now!", 1),
    ("URGENT: Your account has been compromised. Verify your identity immediately.", 1),
    ("You have been selected for a special offer. Act now before it expires!", 1),
    ("FREE iPhone giveaway! Just enter your credit card details to participate.", 1),
    ("Make $5000 per week working from home! No experience needed!", 1),
    ("Your package is waiting. Pay $1.99 shipping fee to claim it now.", 1),
    ("WINNER! You've been chosen for our exclusive prize draw.", 1),
    ("Limited time offer: Buy one get three free! Use code SPAM2024.", 1),
    ("Dear user, your bank account needs immediate verification. Click below.", 1),
    ("Hot singles in your area want to meet you tonight!", 1),
    ("You're our 1,000,000th visitor! Claim your prize now!", 1),
    ("Act now! This incredible deal expires in 24 hours!", 1),
    ("Earn money fast with this one weird trick doctors hate!", 1),
    ("Your loan has been pre-approved! No credit check required.", 1),
    ("FREE trial! Cancel anytime. Enter payment info to start.", 1),
    ("Congratulations, you qualify for a government grant of $25,000!", 1),
    ("ALERT: Suspicious activity on your account. Verify now or be locked out.", 1),
    ("Double your investment in just 30 days! Guaranteed returns!", 1),
    ("You've been specially selected to receive a luxury vacation package!", 1),
    ("Lose 30 pounds in 30 days with this miracle supplement!", 1),
    ("Your email has won the international lottery! Send details to claim.", 1),
    ("Secret method to eliminate all your debt overnight!", 1),
    ("Exclusive VIP access! Only for the next 100 people who sign up.", 1),
    ("WARNING: Your computer is infected! Download our antivirus now!", 1),
    ("Get rich quick with cryptocurrency! Invest $100, earn $10,000!", 1),

    # ---- Ham (label=0) ----
    ("Hey, are you coming to the meeting tomorrow at 3pm?", 0),
    ("I just finished reading that book you recommended. It was great!", 0),
    ("Can you pick up some groceries on your way home?", 0),
    ("The project deadline has been moved to next Friday.", 0),
    ("Happy birthday! Hope you have an amazing day.", 0),
    ("Let's meet for coffee this weekend. Are you free Saturday?", 0),
    ("I sent you the report. Let me know if you have questions.", 0),
    ("Thanks for helping me move last weekend. I really appreciate it.", 0),
    ("The weather is supposed to be nice tomorrow. Want to go for a hike?", 0),
    ("Don't forget we have dinner reservations at 7 tonight.", 0),
    ("Just wanted to check in and see how you're doing.", 0),
    ("I'll be running about 15 minutes late to our lunch.", 0),
    ("Did you see the game last night? What an incredible finish!", 0),
    ("Your presentation was really well done. Great job!", 0),
    ("I'm thinking about taking a cooking class. Want to join?", 0),
    ("The kids had a great time at the park today.", 0),
    ("Can you send me the address for the restaurant?", 0),
    ("I found a really interesting article about machine learning.", 0),
    ("Remember to bring your laptop to the workshop tomorrow.", 0),
    ("The flight lands at 6pm. I'll text you when I arrive.", 0),
    ("How's the new job going? Let's catch up soon.", 0),
    ("I'll pick up the tickets on my way to the venue.", 0),
    ("The cat knocked over the plant again this morning.", 0),
    ("Could you review my code when you get a chance?", 0),
    ("Let me know when you're free to discuss the budget.", 0),
]


# ──────────────────────────────────────────────────────────────
# 2. PyTorch Dataset
# ──────────────────────────────────────────────────────────────

class SpamDataset(Dataset):
    """
    Text classification dataset for spam detection.

    Each sample is tokenized, padded/truncated to a fixed length,
    and paired with a binary label (0=ham, 1=spam).

    Padding strategy:
        - Sequences shorter than max_len → pad with pad_token_id on the RIGHT
        - Sequences longer than max_len → truncate from the RIGHT
        - We pad on the right because GPT uses causal (left-to-right) attention,
          and we want the last NON-padding token to be the "summary" position.

    Note: For a real application, you'd also want an attention mask to
    ignore padding tokens. For this small demo, the model learns to
    handle padding naturally since all sequences are short.
    """

    def __init__(self, data: list[tuple[str, int]], max_len: int = 64, pad_token_id: int = 50256):
        """
        Args:
            data:         list of (text, label) tuples
            max_len:      maximum sequence length (tokens)
            pad_token_id: token ID for padding (GPT-2 uses 50256 = <|endoftext|>)
        """
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.max_len = max_len
        self.pad_token_id = pad_token_id

        # Pre-tokenize all texts
        self.samples = []
        for text, label in data:
            token_ids = self.tokenizer.encode(text)

            # Truncate if too long
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]

            # Pad if too short
            padding_len = max_len - len(token_ids)
            token_ids = token_ids + [pad_token_id] * padding_len

            self.samples.append((torch.tensor(token_ids, dtype=torch.long), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            token_ids: shape (max_len,)
            label:     scalar tensor (0 or 1)
        """
        token_ids, label = self.samples[idx]
        return token_ids, torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────
# 3. Create Train/Val/Test Splits
# ──────────────────────────────────────────────────────────────

def create_classification_loaders(
    data: list[tuple[str, int]] = None,
    max_len: int = 64,
    batch_size: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from text classification data.

    Split ratios:
        train: 70% — model learns from these examples
        val:   15% — monitor for overfitting during training
        test:  15% — final evaluation (never seen during training)

    Args:
        data:        list of (text, label) tuples (defaults to SPAM_DATASET)
        max_len:     maximum sequence length
        batch_size:  batch size for training
        train_ratio: fraction for training
        val_ratio:   fraction for validation
        seed:        random seed for reproducible splits

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if data is None:
        data = SPAM_DATASET

    dataset = SpamDataset(data, max_len=max_len)

    # Calculate split sizes
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    print(f"  Dataset splits: train={train_size}, val={val_size}, test={test_size}")

    # Random split with fixed seed
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Spam Classification Dataset — Demo")
    print("=" * 60)

    print(f"\n  Total samples: {len(SPAM_DATASET)}")
    print(f"  Spam:  {sum(1 for _, l in SPAM_DATASET if l == 1)}")
    print(f"  Ham:   {sum(1 for _, l in SPAM_DATASET if l == 0)}")

    # Create loaders
    print(f"\n  Creating dataloaders...")
    train_loader, val_loader, test_loader = create_classification_loaders(
        max_len=64, batch_size=4
    )

    # Inspect a batch
    for token_ids, labels in train_loader:
        print(f"\n  Sample batch:")
        print(f"    Token IDs shape: {token_ids.shape}")  # (4, 64)
        print(f"    Labels shape:    {labels.shape}")      # (4,)
        print(f"    Labels:          {labels.tolist()}")

        # Decode first sample
        tokenizer = tiktoken.get_encoding("gpt2")
        text = tokenizer.decode(token_ids[0].tolist())
        print(f"    First text:      {text[:80]}...")
        print(f"    First label:     {'spam' if labels[0] == 1 else 'ham'}")
        break
