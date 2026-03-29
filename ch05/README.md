# Chapter 5: Pretraining on Unlabeled Data 🔥

The "soul injection" chapter — training random weights into a meaningful language model.

## Core Concept: Next-Token Prediction

Pretraining objective: given a sequence of tokens, predict the next one.

```
Input:   [The] [cat] [sat] [on]  [the]
Target:  [cat] [sat] [on]  [the] [mat]
```

The model learns language structure purely from predicting the next word — no labels, no human annotation. This is **self-supervised learning**: the training signal comes from the data itself.

## Loss Function: Cross-Entropy

For next-token prediction, we use cross-entropy loss:

```
loss = -1/N Σ log P(target_i | context_i)
```

- The model outputs a probability distribution over the entire vocabulary (50,257 tokens)
- Cross-entropy measures how far the predicted distribution is from the true (one-hot) distribution
- Lower loss = model assigns higher probability to the correct next token

### Perplexity

Perplexity = exp(loss) — an intuitive metric:
- **Perplexity ≈ 1**: perfect prediction
- **Perplexity ≈ 50,257**: random guessing over the vocab
- **Perplexity ≈ 10-30**: good language model on standard benchmarks

## AdamW vs Adam

### Adam (Adaptive Moment Estimation)
- Maintains per-parameter learning rates using gradient moments
- First moment (mean of gradients) + second moment (variance of gradients)
- Problem: weight decay is coupled with the adaptive learning rate

### AdamW (Decoupled Weight Decay)
- Separates weight decay from the gradient-based update
- Result: more effective regularization
- Why it matters: in Adam, weight decay is scaled by the adaptive learning rate, making it inconsistent across parameters. AdamW applies it uniformly.

```python
# Adam: weight decay is part of gradient update (problematic)
gradient += weight_decay * weight
weight -= lr * adam_update(gradient)

# AdamW: weight decay is separate (correct)
weight -= lr * adam_update(gradient)
weight -= lr * weight_decay * weight  # independent step
```

### Weight decay in practice
- Not applied to bias terms or LayerNorm parameters
- Typical value: 0.1
- Acts as L2 regularization to prevent overfitting

## Learning Rate Schedule

### Why not a constant learning rate?
- Too high: training diverges (loss explodes)
- Too low: training is slow and may get stuck
- Solution: start low, ramp up, then decay

### Warmup + Cosine Decay

```
LR
 ^
 |     /‾‾‾‾‾‾‾\
 |    /          \
 |   /            \
 |  /              \___________
 | /
 +──────────────────────────────→ Steps
   ← warmup →← cosine decay →
```

1. **Linear warmup** (first ~10% of training):
   - LR increases linearly from 0 → max_lr
   - Why: random weights produce noisy gradients; large LR would destabilize
   - Lets the model "settle" before applying full learning rate

2. **Cosine decay** (remaining ~90%):
   - LR smoothly decreases from max_lr → min_lr following a cosine curve
   - Why: as the model converges, smaller updates prevent overshooting
   - Smoother than step decay (no sudden jumps)

## Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

- Limits the total norm of all gradients to `max_norm`
- If ‖g‖ > max_norm: g ← g × (max_norm / ‖g‖)
- Prevents **exploding gradients** — especially critical with:
  - Random initialization (early training)
  - Long sequences (gradients compound through many timesteps)
  - Deep networks (gradients multiply through many layers)

## Overfitting on Small Data

When training on a small text (like "The Verdict" — ~20K tokens):

- **Training loss** drops quickly → model memorizes the text
- **Validation loss** drops then rises → classic overfitting
- The model can reproduce passages from the training text verbatim
- **This is expected and fine for a demo** — real LLMs train on billions of tokens

### Why we do it anyway
- Demonstrates the training pipeline works correctly
- Shows the model can actually learn language patterns
- Rapid iteration (minutes, not weeks)
- Before/after text generation comparison is dramatic

### To generalize, you'd need:
- Much more data (books, web text, code)
- Larger model (more parameters)
- More training time (days/weeks on GPUs)
- Data augmentation and better regularization

## Files

| File | Description |
|------|-------------|
| `train.py` | Training loop with AdamW, warmup, cosine LR, gradient clipping |
| `evaluate.py` | Validation loss, perplexity, text generation monitoring |
| `utils.py` | LR scheduler, text generation helper, checkpoint save/load |
| `pretrain.py` | Main entry point — data → model → training → results |
| `demo.ipynb` | Interactive notebook with loss curves and generation comparison |

## Usage

```bash
# Run pretraining demo
python -m ch05.pretrain

# Expected output:
# - Loss decreasing from ~10 to ~1
# - Perplexity dropping from ~50,000 to ~3
# - Generated text becoming coherent
```
