# Chapter 7: Fine-Tuning to Follow Instructions 🎯

The final chapter — turning a language model into a useful assistant.

## What is Instruction Tuning?

Instruction tuning (also called **Supervised Fine-Tuning / SFT**) teaches a pretrained language model to follow human instructions. It's the step that turns a "text completion engine" into something that actually answers questions, follows orders, and feels like an assistant.

### The Three Stages of LLM Training

```
Stage 1: Pretraining (ch05)
    Data:   Raw text (books, web, code)
    Goal:   Learn language patterns, knowledge, grammar
    Result: Good at completing text, but doesn't follow instructions
    Scale:  Billions of tokens, weeks of compute

Stage 2: Instruction Tuning / SFT (ch07 — this chapter)
    Data:   (instruction, response) pairs
    Goal:   Learn to follow the instruction format
    Result: Model answers questions, follows commands
    Scale:  Thousands of examples, hours of compute

Stage 3: RLHF / Preference Alignment (not implemented)
    Data:   Human preferences (which response is better?)
    Goal:   Align with human values and preferences
    Result: Safer, more helpful, less harmful outputs
    Scale:  Thousands of comparisons, specialized training
```

### Why Does Instruction Tuning Work?

A pretrained model already "knows" things — it's absorbed vast knowledge from its training data. But it doesn't know **how to use that knowledge on command**. It's like a brilliant person who only knows how to continue sentences, not answer questions.

Instruction tuning teaches the model a new **interface**: "When you see `### Instruction:`, treat what follows as a task, and generate a helpful response after `### Response:`."

Key insight: **We're not teaching the model new knowledge — we're teaching it a new behavior.**

## Prompt Template

We use the Alpaca format:

```
### Instruction:
{instruction}

### Response:
{response}
```

The template is crucial because:
1. It gives the model a clear signal about what's expected
2. It separates the instruction from the response
3. At inference time, we provide the instruction and let the model complete the response

## How It Differs from Other Fine-Tuning

| Aspect | Pretraining (ch05) | Classification (ch06) | Instruction Tuning (ch07) |
|--------|--------------------|-----------------------|---------------------------|
| **Data** | Raw text | Text + labels | Instruction + response |
| **Loss** | Next-token on all text | Cross-entropy on class | Next-token (optionally masked) |
| **Output** | Next token | Class label | Free-form text |
| **Architecture** | Unchanged | + classification head | Unchanged |
| **Goal** | Learn language | Classify text | Follow instructions |

## Loss Masking (Advanced)

In our simplified implementation, we compute loss on the **entire sequence** (instruction + response). This means the model is penalized for not perfectly predicting the instruction tokens too.

A more sophisticated approach masks the instruction tokens:

```
### Instruction:          ← loss = 0 (masked)
What is the capital...    ← loss = 0 (masked)

### Response:             ← loss = 0 (masked)
The capital of France...  ← loss computed here!
```

This is what production systems (LLaMA-2, Mistral, etc.) do. The benefit is that the model focuses its learning capacity entirely on generating good responses.

## RLHF: Reinforcement Learning from Human Feedback

After SFT, many LLMs go through RLHF to align with human preferences:

### How RLHF Works

```
Step 1: Collect comparisons
    Show humans two model responses to the same prompt
    Human picks which one is better
    → Creates a preference dataset

Step 2: Train a reward model
    Learns to predict which response a human would prefer
    Input: (prompt, response) → Output: scalar reward score

Step 3: RL fine-tuning (PPO)
    Use the reward model to give feedback to the LLM
    LLM generates responses → reward model scores them
    PPO algorithm updates LLM to maximize reward
    KL penalty keeps the model close to the SFT version
```

### Why RLHF Matters

SFT teaches the model to follow instructions, but:
- It might give harmful or incorrect information confidently
- It might be verbose when concise answers are better
- It might not refuse dangerous requests

RLHF teaches the model **human preferences**: be helpful, be honest, be harmless.

## DPO: Direct Preference Optimization

DPO is a simpler alternative to RLHF that skips the reward model:

```
RLHF:  SFT → Reward Model → PPO Training (complex, unstable)
DPO:   SFT → Direct optimization on preferences (simpler, stable)
```

DPO directly optimizes the language model using preference pairs (chosen response vs rejected response), without needing a separate reward model or RL training. It reformulates the RLHF objective into a simple classification loss.

### The Evolution

```
2020: GPT-3        → Few-shot prompting (no fine-tuning)
2022: InstructGPT   → SFT + RLHF (the ChatGPT recipe)
2023: LLaMA-2      → SFT + RLHF with rejection sampling
2023: Zephyr       → SFT + DPO (simpler, competitive results)
2024: Most models  → SFT + DPO (industry standard)
```

## Files in This Chapter

| File | Description |
|------|-------------|
| `instruction_dataset.py` | Synthetic instruction dataset (Alpaca format) + DataLoader |
| `instruction_tuning.py` | Training loop for instruction tuning |
| `chat.py` | Interactive chat REPL |
| `demo.ipynb` | End-to-end demo notebook |

## Quick Start

```bash
# Train the model
python -m ch07.instruction_tuning

# Chat with it
python -m ch07.chat

# Or run the notebook
jupyter notebook ch07/demo.ipynb
```

## Key Takeaways

1. **Instruction tuning is surprisingly effective** — even a small dataset (~50 examples) can teach a model to follow a prompt template
2. **The prompt template matters** — consistent formatting is what makes it work
3. **SFT is necessary but not sufficient** — RLHF/DPO adds the alignment layer
4. **Small models can demonstrate the concept** — even if they lack the knowledge of larger models, the *behavior* of following instructions can be learned
5. **This is what makes LLMs "useful"** — without instruction tuning, even GPT-4 would just be a fancy autocomplete

## 🎉 Course Complete!

This is the final chapter. We've built a complete LLM pipeline:

```
Text → Tokens → Attention → GPT → Pretrain → Classify → Instruct
ch02    ch03      ch04      ch05     ch06       ch07
```

From raw text to an instruction-following model — all from scratch in PyTorch.
