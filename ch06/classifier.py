"""
Chapter 6: GPT-Based Text Classifier.

Adapts a pretrained GPT model for text classification:
  1. Freeze pretrained weights (optional: unfreeze last N layers)
  2. Replace the language model head with a classification head
  3. Use the last token's hidden state for classification (like [CLS])

Key insight: GPT is a causal (left-to-right) model, so the LAST token
has seen all previous tokens through self-attention. This makes it
the best candidate for sequence-level classification — it's the most
"informed" position in the sequence.

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 6.
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ch04.config import GPTConfig
from ch04.gpt_model import GPTModel


class GPTClassifier(nn.Module):
    """
    GPT model adapted for text classification.

    Architecture:
        Input token IDs
            ↓
        [Pretrained GPT backbone]
            Token Embedding + Position Embedding
            N × TransformerBlock
            Final LayerNorm
            ↓
        Hidden states (batch, seq_len, d_model)
            ↓
        Extract LAST token hidden state  ← key design choice
            ↓
        Classification Head:
            Linear(d_model → num_classes)
            ↓
        Logits (batch, num_classes)

    Why the last token?
        In causal (left-to-right) attention, each token can only attend
        to tokens before it. The last token is special because it has
        attended to ALL previous tokens — it's the most "complete"
        representation of the entire sequence.

        This is analogous to BERT's [CLS] token, but naturally emerges
        from the causal attention pattern.

    Transfer learning strategy:
        - Freeze most pretrained layers (preserve learned representations)
        - Optionally unfreeze the last few layers for task adaptation
        - Train only the new classification head + unfrozen layers
        - This is much faster than training from scratch
    """

    def __init__(self, cfg: GPTConfig, num_classes: int, unfreeze_last_n: int = 1):
        """
        Args:
            cfg:              GPTConfig matching the pretrained model
            num_classes:      number of classification categories
            unfreeze_last_n:  number of transformer blocks to unfreeze from the end
                              0 = freeze everything, only train the head
                              -1 = unfreeze all (full fine-tuning)
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # ---- Load GPT backbone ----
        self.backbone = GPTModel(cfg)

        # ---- Remove the original LM head (vocab projection) ----
        # We don't need it for classification
        # But we keep it in the backbone since GPTModel creates it;
        # we'll just ignore it and add our own classification head.

        # ---- Classification head ----
        # Simple linear layer: d_model → num_classes
        # No activation here — CrossEntropyLoss expects raw logits
        self.classifier_head = nn.Linear(cfg.d_model, num_classes)

        # ---- Freeze/unfreeze strategy ----
        self._freeze_backbone(unfreeze_last_n)

    def _freeze_backbone(self, unfreeze_last_n: int):
        """
        Freeze pretrained weights selectively.

        Why freeze?
            - Lower layers learn general language features (syntax, semantics)
            - These features transfer well across tasks
            - Training fewer params = faster + less overfitting
            - Upper layers are more task-specific, benefit from fine-tuning

        Args:
            unfreeze_last_n: number of blocks to keep trainable
                             0  = freeze all backbone weights
                             -1 = unfreeze everything
        """
        if unfreeze_last_n == -1:
            # Full fine-tuning: everything is trainable
            return

        # Freeze ALL backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            # Unfreeze the last N transformer blocks
            num_blocks = len(self.backbone.blocks)
            for i in range(num_blocks - unfreeze_last_n, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

            # Also unfreeze the final LayerNorm (it sits after the blocks)
            for param in self.backbone.final_norm.parameters():
                param.requires_grad = True

        # The classifier_head is always trainable (it's new, not frozen)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            idx: input token IDs, shape (batch, seq_len)

        Returns:
            logits: classification logits, shape (batch, num_classes)
        """
        # ---- Get hidden states from the backbone ----
        # We need the hidden states BEFORE the LM head
        # So we replicate the backbone forward but stop before head projection

        batch, seq_len = idx.shape
        assert seq_len <= self.cfg.max_seq_len

        pos = torch.arange(seq_len, device=idx.device)

        # Embeddings
        tok_emb = self.backbone.tok_emb(idx)       # (batch, seq_len, d_model)
        pos_emb = self.backbone.pos_emb(pos)       # (seq_len, d_model)
        x = self.backbone.drop_emb(tok_emb + pos_emb)  # (batch, seq_len, d_model)

        # Transformer blocks
        x = self.backbone.blocks(x)                # (batch, seq_len, d_model)

        # Final layer norm
        x = self.backbone.final_norm(x)            # (batch, seq_len, d_model)

        # ---- Extract last token's hidden state ----
        # x[:, -1, :] → shape (batch, d_model)
        # This is the "most informed" token due to causal attention
        last_hidden = x[:, -1, :]                  # (batch, d_model)

        # ---- Classification ----
        logits = self.classifier_head(last_hidden)  # (batch, num_classes)

        return logits

    def load_pretrained_backbone(self, checkpoint_path: str, device: str = "cpu"):
        """
        Load pretrained weights into the backbone.

        Only loads weights that exist in both the checkpoint and the model.
        The classifier_head weights remain randomly initialized.

        Args:
            checkpoint_path: path to a pretrained GPT checkpoint (.pt file)
            device:          device to load weights on
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Load into the backbone (strict=False to ignore missing/extra keys)
        self.backbone.load_state_dict(state_dict, strict=True)
        print(f"  ✅ Loaded pretrained backbone from {checkpoint_path}")

    def count_trainable_parameters(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        """Returns the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ch05.pretrain import SMALL_CONFIG

    torch.manual_seed(42)
    num_classes = 2  # e.g., spam / not spam

    print("=" * 60)
    print("GPT Classifier — Architecture Demo")
    print("=" * 60)

    # Test different freeze strategies
    for unfreeze_n in [0, 1, 2, -1]:
        clf = GPTClassifier(SMALL_CONFIG, num_classes=num_classes, unfreeze_last_n=unfreeze_n)
        label = "all" if unfreeze_n == -1 else str(unfreeze_n)
        print(f"\n  Unfreeze last {label} blocks:")
        print(f"    Total params:     {clf.count_total_parameters():,}")
        print(f"    Trainable params: {clf.count_trainable_parameters():,}")
        ratio = clf.count_trainable_parameters() / clf.count_total_parameters() * 100
        print(f"    Trainable ratio:  {ratio:.1f}%")

    # Test forward pass
    clf = GPTClassifier(SMALL_CONFIG, num_classes=num_classes, unfreeze_last_n=1)
    idx = torch.randint(0, SMALL_CONFIG.vocab_size, (2, 16))
    logits = clf(idx)
    print(f"\n  Forward pass:")
    print(f"    Input shape:  {idx.shape}")       # (2, 16)
    print(f"    Output shape: {logits.shape}")     # (2, 2)
    print(f"    Logits: {logits.detach()}")
