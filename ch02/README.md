# Chapter 2: Working with Text Data

This chapter covers the **text-to-numbers pipeline** — how raw text gets transformed into the tensor inputs that a GPT model actually consumes.

## Key Concepts

### 1. Tokenization

**Problem:** Neural networks work with numbers, not text.

**Solution:** Convert text into a sequence of integer token IDs.

- **Word-level tokenizer** (`SimpleTokenizer`): Split on whitespace/punctuation → build vocabulary → map tokens to IDs. Simple but limited — can't handle unknown words.
- **BPE (Byte Pair Encoding)**: Iteratively merges frequent character pairs. Handles any text, no unknown tokens. GPT-2 uses BPE with ~50,257 tokens.
- We use **tiktoken** (OpenAI's fast BPE implementation) for the GPT-2 tokenizer.

### 2. Token Embeddings

Each token ID gets mapped to a dense vector via `nn.Embedding`:

```
token_id: 15496 → [0.12, -0.34, 0.56, ..., 0.78]  (embed_dim values)
```

These vectors are **learned** during training — the model figures out what each token "means" in context.

### 3. Positional Embeddings

Transformers process all tokens in parallel (no recurrence), so they have no notion of order. Positional embeddings fix this:

```
position 0 → [0.01, 0.23, ...]
position 1 → [0.11, -0.05, ...]
...
```

**Final input = token_embedding + positional_embedding**

This gives the model both *what* each token is and *where* it appears in the sequence.

### 4. Data Loading (Sliding Window)

GPT is trained on **next-token prediction**. We create training pairs with a sliding window:

```
Text tokens: [A, B, C, D, E, F, G]
max_len = 4, stride = 2

Sample 1: input=[A,B,C,D] → target=[B,C,D,E]
Sample 2: input=[C,D,E,F] → target=[D,E,F,G]
```

- `stride < max_len` → overlapping windows (more training data)
- `stride = max_len` → no overlap

## Files

| File | Description |
|------|-------------|
| `tokenizer.py` | SimpleTokenizer + tiktoken BPE demo |
| `embeddings.py` | Token + positional embedding layer |
| `dataloader.py` | Sliding window dataset & dataloader |
| `demo.ipynb` | Full pipeline walkthrough |
| `the-verdict.txt` | Sample text (Edith Wharton's "The Verdict") |

## Usage

```bash
pip install tiktoken torch

# Run individual modules
cd ch02
python tokenizer.py
python embeddings.py
python dataloader.py

# Or run the notebook
jupyter notebook demo.ipynb
```

## Key Takeaways

1. **Tokenization is not trivial** — BPE balances vocabulary size with coverage
2. **Embeddings are learned** — not hand-crafted features
3. **Position matters** — without positional embeddings, "dog bites man" = "man bites dog"
4. **Next-token prediction** is a self-supervised objective — no labels needed, just raw text
