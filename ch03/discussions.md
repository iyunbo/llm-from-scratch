# Chapter 3 — Discussions

Questions and deep dives from studying attention mechanisms.

---

## Why do we need three separate matrices (Q, K, V)?

At initialization, Q, K, V are just three random matrices — no semantic difference. Their "roles" emerge through training.

**Why three instead of one?** Decoupling. With one matrix, "who to match with" and "what information to pass" are locked together. Three matrices let the model learn independently:
- **Q·K** decides "who to attend to" (matching)
- **V** decides "what information to transmit" (content)

A token can strongly match another (high Q·K score) while passing completely different information (V). This flexibility is critical.

**Analogy — search engine:**
- Q = your search query
- K = page titles/tags (used for matching)
- V = page content (what you actually read)

Titles and content can be very different — one is for matching, the other for information. Same principle.

---

## How do heads specialize in multi-head attention?

They're not assigned roles. Gradient descent naturally pushes them apart because:

1. **The task is multi-faceted** — predicting the next token requires syntax, semantics, position, coreference, etc.
2. **Redundancy is wasteful** — if two heads learn the same pattern, one doesn't help reduce loss
3. **The output projection rewards diversity** — it mixes all heads, preferring diverse signals

Research on trained models (GPT-2, BERT) shows heads do specialize: some track syntax, some handle coreference, some focus on adjacent positions, some copy patterns. But no two training runs produce the same assignment.

~30-50% of heads can be pruned with minimal performance loss. This insight led to GQA (Grouped Query Attention) in LLaMA 2/3.

---

## Why is multi-head better than one big head?

**Same parameter count, more attention patterns.**

Core issue: **one softmax can only produce one probability distribution.** It can't simultaneously give high weight to two different positions — the sum-to-1 constraint forces compromise.

```
Single head on "it":
  cat: 0.40  tired: 0.15  ← has to split between coreference and semantics

Multi-head:
  Head 1: cat: 0.85       ← coreference, fully committed
  Head 2: tired: 0.60     ← semantics, fully committed
```

12 heads × 64 dims = same params as 1 head × 768 dims. But 12 independent softmax distributions = 12 simultaneous focus points.

**One head = one pair of eyes, one focus point.**
**Multi-head = twelve pairs of eyes, twelve focus points, same cost.**

---

## What is dropout and why use it in attention?

**Randomly zeroing out neurons during training to prevent over-reliance on any single feature.**

In attention: randomly zeroing some attention weights forces the model to not over-attend to any fixed position. Like randomly making team members "call in sick" during training — everyone else has to learn to compensate.

Key details:
- Only active during training, disabled at inference
- Surviving values are scaled up by `1/(1-p)` to maintain expected sum
- GPT-2 uses dropout=0.1 on attention weights
- Larger models with more data need less dropout

---

## Why does nn.Linear have bias by default?

`y = xW + b` — bias adds a translation (shift), giving the model more freedom than `y = xW` (which is locked to pass through origin).

**In attention:** GPT-2 uses `bias=False` for Q, K, V projections because softmax normalization largely cancels the shift effect. Rule of thumb: if a normalization layer (softmax, layernorm) follows immediately, bias can be dropped to save parameters.
