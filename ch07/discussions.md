# Chapter 7 — Discussions

Questions and deep dives from studying instruction tuning.

---

## Why does instruction tuning also need adaptive learning rate?

**Different parameters need different update speeds — this matters even more in fine-tuning than pretraining.**

During pretraining, all parameters start random and need large adjustments. During instruction tuning, the model already knows language — now it needs to learn "follow instructions":

- **Lower layers** (embeddings, early attention): encode basic language knowledge → need tiny updates, shouldn't be disrupted
- **Upper layers** (later attention, output head): need to learn new behavior patterns → need larger updates

AdamW solves this automatically: each parameter's effective learning rate adapts based on its gradient history. Lower layers with small gradients get small steps; upper layers with larger gradients get bigger steps.

With fixed LR (SGD): too large → destroys language knowledge (catastrophic forgetting); too small → can't learn the new task; compromise → both sides unsatisfied.

**Additional factor:** Instruction datasets are small and diverse ("translate this" vs "write a poem" vs "explain physics"). Batch-to-batch gradient variance is high. Adam's momentum smooths this noise; SGD would oscillate wildly.

**Analogy:** Instruction tuning = fine carving on existing sculpture. Different areas need different chisel pressures. Adaptive LR auto-selects the right chisel. Fixed LR is one chisel for everything.

---

## Why can't the trained chat.py actually talk?

**Three compounding reasons — all about scale, not about code correctness.**

### 1. No real language ability (root cause)

GPT-2 was trained on billions of tokens → fluent English. Our small model (128 dim, 4 layers) trained on ~20K tokens from one short story → it memorized fragments of "The Verdict", not English. It's like someone who read one book trying to hold a conversation.

### 2. Too little instruction data

ChatGPT: tens of thousands of high-quality instruction-response pairs. Ours: ~50 synthetic examples. The model learned the *format* ("see ### Instruction: → try to respond") but not the *content*. Format without substance.

### 3. Model too small

ChatGPT: 175B parameters → massive memory capacity for grammar + world knowledge + reasoning + instruction following. Ours: ~13M parameters → can barely memorize basic word frequency patterns. Even with perfect training data, 13M params cannot hold a conversation.

### Is this expected?

**Absolutely.** The project's goal isn't building a usable chatbot — it's understanding how every LLM component works. The code is correct; the scale is wrong. Scale up the model 1000× and data 100000×, and the same code produces ChatGPT-level results.

**Analogy:** We built a complete airplane model from LEGO. It can't fly, but the structure is identical to a real plane.
