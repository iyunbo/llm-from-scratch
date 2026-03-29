# Chapter 6: Fine-Tuning for Classification

Adapting a pretrained GPT model for text classification — the bridge from general language understanding to solving specific tasks.

## Core Concept: Transfer Learning

```
Pretraining (ch05)          Fine-tuning (ch06)
────────────────           ─────────────────
Massive text data    →     Small labeled data
Next-token prediction →    Classification
General knowledge    →     Task-specific skill
Expensive (hours)    →     Cheap (minutes)
```

**The key insight:** A model pretrained on language modeling has already learned syntax, semantics, and world knowledge. Fine-tuning transfers this knowledge to a specific task with very little additional training.

## Why Freeze Bottom Layers?

```
Layer 12 (top)    ← Task-specific features      ← UNFREEZE (fine-tune)
Layer 11          ← More abstract features       ← UNFREEZE (optional)
...
Layer 3           ← Syntactic patterns           ← FREEZE
Layer 2           ← Word relationships           ← FREEZE
Layer 1 (bottom)  ← Basic token features         ← FREEZE
Embeddings        ← Token representations        ← FREEZE
```

**Lower layers** learn general language features (word meanings, grammar, syntax) that are useful for ANY NLP task. These transfer well without modification.

**Upper layers** learn more abstract, task-specific features. These benefit from fine-tuning on the target task.

**Why not train everything?**
- Risk of **catastrophic forgetting** — overwriting useful pretrained features
- Small datasets can cause **overfitting** if too many params are trainable
- Frozen layers = **faster training** (no gradient computation needed)

## Classification Head Design

```
GPT Backbone (pretrained)
    ↓
Hidden states: (batch, seq_len, d_model)
    ↓
Extract LAST token: (batch, d_model)    ← Why last?
    ↓
Linear(d_model → num_classes)
    ↓
Logits: (batch, num_classes)
```

### Why the Last Token?

GPT uses **causal (left-to-right) attention**. Each token can only see tokens before it:

```
Token:    [The] [cat] [sat] [on] [the] [mat]
Sees:      ←    ←←   ←←←  ←←←← ←←←←← ←←←←←←
```

The **last token** has attended to ALL previous tokens — it holds the most complete representation of the entire sequence. It's the natural choice for sequence-level classification.

(BERT uses `[CLS]` for the same reason, but it's bidirectional so any position works.)

### Why Replace the Head?

The pretrained head maps `d_model → vocab_size` (50,257 outputs for next-token prediction). For classification, we only need `d_model → num_classes` (e.g., 2 for spam detection).

## Training Differences: Pretraining vs Fine-Tuning

| Aspect | Pretraining (ch05) | Fine-Tuning (ch06) |
|--------|-------------------|---------------------|
| **Data** | Raw text (unlabeled) | (text, label) pairs |
| **Objective** | Next-token prediction | Classification |
| **Loss** | Cross-entropy on vocab | Cross-entropy on classes |
| **Metric** | Perplexity | Accuracy |
| **Learning rate** | 5e-4 (higher) | 5e-5 (10× lower) with pretrained backbone; 5e-4 without |

> **Note:** The 5e-5 lr assumes pretrained weights are loaded. When fine-tuning from random initialization (no pretrained backbone), use 5e-4 — otherwise the model won't learn.
| **Epochs** | 30+ | 10-20 |
| **Trainable params** | All | Head + last few layers |

### Why Lower Learning Rate?

The pretrained weights are in a good region of the loss landscape. A large LR would "kick" them out, destroying the useful representations. A small LR makes gentle adjustments.

## Files

| File | Description |
|------|-------------|
| `classifier.py` | `GPTClassifier` — GPT backbone + classification head |
| `dataset.py` | `SpamDataset` — tokenized spam/ham dataset with padding |
| `train_classifier.py` | Fine-tuning loop with accuracy tracking |
| `demo.ipynb` | Interactive notebook: train, evaluate, classify |

## Quick Start

```bash
# Run the full pipeline
python -m ch06.train_classifier

# Or explore interactively
jupyter notebook ch06/demo.ipynb
```

## Key Takeaways

1. **Transfer learning is powerful** — pretrained features save enormous training time
2. **Freeze strategically** — keep general features, fine-tune task-specific layers
3. **Small data is enough** — classification needs far less data than pretraining
4. **The last token matters** — in causal models, it's the most informed position
5. **Lower LR for fine-tuning** — gentle adjustments preserve pretrained knowledge

## Reference

- 📖 Chapter 6 of "Build a Large Language Model from Scratch" by Sebastian Raschka
- 📝 [ULMFiT paper](https://arxiv.org/abs/1801.06146) — pioneered gradual unfreezing for NLP
- 📝 [GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — first GPT for classification
