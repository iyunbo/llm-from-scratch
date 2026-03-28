"""
Chapter 2: Tokenizer implementations.

1. SimpleTokenizer — a basic word-level tokenizer to understand the concept
2. BPE tokenizer via tiktoken (GPT-2) — what real models use
"""

import re


class SimpleTokenizer:
    """
    A minimal word-level tokenizer.
    Splits text on whitespace and punctuation, builds a vocabulary,
    and maps tokens <-> integer IDs.
    """

    def __init__(self, vocab):
        """
        Args:
            vocab: dict mapping token strings to integer IDs
        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """Convert text to a list of token IDs."""
        # Split on whitespace and common punctuation, keeping punctuation as tokens
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        ids = [self.str_to_int[t] for t in tokens]
        return ids

    def decode(self, ids):
        """Convert a list of token IDs back to text."""
        tokens = [self.int_to_str[i] for i in ids]
        text = " ".join(tokens)
        # Clean up spacing before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


def build_vocab(text):
    """
    Build a vocabulary (token -> ID mapping) from raw text.
    Includes special tokens <|unk|> and <|endoftext|>.
    """
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = sorted(set(t.strip() for t in tokens if t.strip()))

    # Add special tokens
    tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token: i for i, token in enumerate(tokens)}
    return vocab


# --- tiktoken (BPE) tokenizer demo ---

def tiktoken_demo(text):
    """
    Demonstrate GPT-2's BPE tokenizer via tiktoken.
    Returns encoded IDs and decoded text.
    """
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(text, allowed_special={"<|endoftext|>"})
    decoded = enc.decode(ids)
    return ids, decoded


# --- Quick test ---

if __name__ == "__main__":
    # Load sample text
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 1. Simple tokenizer
    print("=== Simple Word-Level Tokenizer ===")
    vocab = build_vocab(raw_text)
    print(f"Vocabulary size: {len(vocab)}")

    tokenizer = SimpleTokenizer(vocab)

    # Use text from the actual vocabulary
    snippet = '"It\'s the last day," she said.'
    try:
        ids = tokenizer.encode(snippet)
        print(f"Original: {snippet}")
        print(f"Encoded:  {ids}")
        print(f"Decoded:  {tokenizer.decode(ids)}")
    except KeyError as e:
        print(f"Token not in vocab: {e}")

    # Encode the first sentence of the text
    first_sentence = raw_text.split(".")[0] + "."
    ids = tokenizer.encode(first_sentence)
    print(f"\nFirst sentence → {len(ids)} tokens")
    print(f"Round-trip: {tokenizer.decode(ids)}")

    # 2. tiktoken (BPE)
    print("\n=== GPT-2 BPE Tokenizer (tiktoken) ===")
    sample_text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
    ids, decoded = tiktoken_demo(sample_text)
    print(f"Token IDs: {ids[:20]}...")
    print(f"Decoded:   {decoded[:80]}...")
    print(f"Num tokens: {len(ids)}")
