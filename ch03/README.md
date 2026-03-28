# Chapter 3: Coding Attention Mechanisms

## What is Self-Attention?

In a standard neural network, each token is processed independently. But language is inherently contextual — the meaning of a word depends on the words around it.

**Self-attention** lets every token in a sequence "look at" every other token and decide how much to attend to each one. The result is a new representation of each token that incorporates information from the entire sequence.

Example: In *"The cat sat on the mat because **it** was tired"*, self-attention helps the model understand that "it" refers to "the cat" by assigning a high attention weight between them.

### Why do we need it?

- **RNNs** process tokens sequentially → slow, struggle with long-range dependencies
- **Self-attention** processes all tokens in parallel → fast, captures any-distance relationships
- It's the core mechanism that makes Transformers work

## Query, Key, Value (QKV)

The attention mechanism borrows concepts from information retrieval:

| Component | Intuition | Role |
|-----------|-----------|------|
| **Query (Q)** | "What am I looking for?" | The current token's question |
| **Key (K)** | "What do I contain?" | Each token's label/descriptor |
| **Value (V)** | "What information do I carry?" | The actual content to retrieve |

### How they work together:

1. Each token generates a Q, K, and V vector (via learned linear projections)
2. The **Query** of token *i* is compared against all **Keys** → produces attention scores
3. Scores are normalized (softmax) → attention weights
4. Weights are used to take a weighted sum of **Values** → context vector

```
score(i, j) = Q_i · K_j / sqrt(d_k)
weights = softmax(scores)
output_i = Σ_j weights(i,j) · V_j
```

**Analogy:** Imagine searching a library. Your Query is what you're looking for, Keys are the book titles, and Values are the book contents. You match your query against titles, then read the most relevant books.

## Scaled Dot-Product: Why divide by √d_k?

The raw dot product `Q · K^T` grows in magnitude as the dimensionality `d_k` increases. If two random vectors each have `d_k` dimensions, their dot product has a variance proportional to `d_k`.

**Problem:** Large dot products → softmax produces extremely peaked distributions (close to one-hot) → vanishing gradients.

**Solution:** Divide by `√d_k` to normalize the variance back to ~1:

```
attention = softmax(Q @ K^T / √d_k) @ V
```

For `d_k = 64`, we divide by `8`. This keeps the softmax in a well-behaved range where gradients can flow.

## Causal Mask

In language modeling, we generate text left-to-right. Token at position *t* should only attend to positions *0, 1, ..., t* — never to future tokens.

**Implementation:**
1. Create an upper-triangular matrix of `-inf` values
2. Add it to the attention scores **before** softmax
3. After softmax, `-inf` positions become 0 → effectively invisible

```
scores:           after masking:       after softmax:
[0.5  0.3  0.8]  [0.5  -inf  -inf]   [1.0   0.0   0.0]
[0.2  0.6  0.1]  [0.2   0.6  -inf]   [0.40  0.60  0.0]
[0.9  0.4  0.7]  [0.9   0.4   0.7]   [0.39  0.24  0.37]
```

**Why is this important?**
- During training, we process entire sequences at once (teacher forcing)
- Without the mask, the model would "cheat" by seeing future tokens
- The mask enforces autoregressive behavior: predict the next token using only past context

## Multi-Head Attention

Instead of one big attention operation, we split into multiple **heads**, each attending to different aspects of the input.

### Why multiple heads?

A single attention head can only focus on one type of relationship at a time. With multiple heads:
- Head 1 might learn **syntactic** relationships (subject-verb agreement)
- Head 2 might learn **semantic** relationships (pronoun resolution)
- Head 3 might learn **positional** patterns (nearby words)

### How it works:

1. **Split:** Divide `d_model` into `num_heads` pieces, each of size `head_dim = d_model / num_heads`
2. **Attend:** Each head runs its own independent scaled dot-product attention
3. **Concat:** Concatenate all head outputs back to `d_model` dimensions
4. **Project:** Pass through a final linear layer

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o

where head_i = Attention(Q @ W_qi, K @ W_ki, V @ W_vi)
```

### Practical example:

With `d_model=256` and `num_heads=8`:
- Each head works with `head_dim = 256/8 = 32` dimensions
- 8 parallel attention operations
- Results concatenated back to 256 dimensions
- Output projection mixes information across heads

## Summary: Three Levels of Attention

| Level | Parameters | Mask | Dropout | Use case |
|-------|-----------|------|---------|----------|
| **SimpleAttention** | Manual W_q, W_k, W_v | ✗ | ✗ | Understanding the math |
| **CausalAttention** | nn.Linear projections | ✓ | ✓ | Single-head language modeling |
| **MultiHeadAttention** | Multi-head + output proj | ✓ | ✓ | Full transformer attention |

## Files

- `attention.py` — All three attention implementations with demo
- `demo.ipynb` — Interactive visualization of attention weights and causal masking
