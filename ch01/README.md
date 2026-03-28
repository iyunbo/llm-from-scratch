# Chapter 1: Understanding Large Language Models

> No code in this chapter — it's about building the mental model before building the actual model.

## Key Concepts

### What is an LLM?
- A neural network trained on massive text data to predict the next token
- "Large" = billions of parameters, trained on trillions of tokens
- The core insight: **next-token prediction is surprisingly powerful** — it forces the model to learn grammar, facts, reasoning, and even code

### The Transformer Architecture
- Introduced in "Attention Is All You Need" (Vaswani et al., 2017)
- Key innovation: **self-attention** — every token can attend to every other token
- Replaced RNNs/LSTMs for sequence tasks because it can be parallelized
- Components: input embedding → positional encoding → attention layers → feed-forward → output

### GPT Architecture
- **G**enerative **P**re-trained **T**ransformer
- Decoder-only transformer (vs encoder-decoder like original Transformer)
- Autoregressive: generates one token at a time, feeds it back as input
- Pre-trained on unlabeled text, then fine-tuned for specific tasks

### Evolution
| Model | Year | Parameters | Key Innovation |
|-------|------|-----------|----------------|
| GPT-1 | 2018 | 117M | Pre-training + fine-tuning paradigm |
| GPT-2 | 2019 | 1.5B | Zero-shot task transfer |
| GPT-3 | 2020 | 175B | In-context learning, few-shot prompting |
| GPT-4 | 2023 | ~1.8T (rumored) | Multimodal, RLHF |

### Stages of Building an LLM
1. **Data collection & preprocessing** — tokenization, cleaning, deduplication
2. **Pre-training** — next-token prediction on massive unlabeled corpus
3. **Fine-tuning** — adapt to specific tasks (classification, instruction-following)
4. **Alignment** — RLHF, constitutional AI, making it helpful and harmless

## My Takeaways

- **LLMs are compression engines.** They compress the statistical patterns of human language into weights. When you prompt them, you're decompressing.
- **Scale is not magic** — it's a tradeoff. More parameters = more capacity to memorize patterns, but also more compute, more data, more ways to go wrong.
- **The "from scratch" approach matters** because when you build every layer yourself, you understand *why* each design choice was made — not just *what* it does.

## What's Next

Chapter 2: Working with text data — tokenization, embeddings, data loaders. This is where the code starts.
