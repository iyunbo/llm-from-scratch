"""
Chapter 2: GPT Data Loader.

Creates input-target pairs using a sliding window approach:
  input  = tokens[i   : i + max_len]
  target = tokens[i+1 : i + max_len + 1]

This is how GPT learns next-token prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    """
    A PyTorch Dataset that produces (input, target) pairs
    from a flat list of token IDs using a sliding window.

    Args:
        token_ids: list or tensor of token IDs
        max_len: context window size (number of tokens per sample)
        stride: step size for the sliding window
    """

    def __init__(self, token_ids, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_len, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_len]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_len + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(text, tokenizer, max_len=256, stride=128, batch_size=4, shuffle=True):
    """
    End-to-end: text -> tokenize -> dataset -> dataloader.

    Args:
        text: raw text string
        tokenizer: tiktoken encoding object (must have .encode())
        max_len: context window size
        stride: sliding window step (< max_len for overlap)
        batch_size: batch size
        shuffle: whether to shuffle

    Returns:
        PyTorch DataLoader yielding (input, target) batches
    """
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    dataset = GPTDataset(token_ids, max_len, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader


# --- Demo ---

if __name__ == "__main__":
    import tiktoken

    # Load text
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Tokenize with GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")
    total_tokens = len(enc.encode(raw_text))
    print(f"Total tokens in text: {total_tokens}")

    # Create dataloader with small context for demo
    MAX_LEN = 4
    STRIDE = 1

    loader = create_dataloader(raw_text, enc, max_len=MAX_LEN, stride=STRIDE, batch_size=8)

    # Show one batch
    inputs, targets = next(iter(loader))
    print(f"\nBatch shape: inputs={inputs.shape}, targets={targets.shape}")
    print(f"\nFirst sample in batch:")
    print(f"  Input:  {inputs[0].tolist()}")
    print(f"  Target: {targets[0].tolist()}")
    print(f"  (target is input shifted right by 1)")

    # Larger context window
    print(f"\n--- With larger context window ---")
    MAX_LEN = 256
    STRIDE = 256  # no overlap

    loader = create_dataloader(raw_text, enc, max_len=MAX_LEN, stride=STRIDE, batch_size=2)
    inputs, targets = next(iter(loader))
    print(f"Batch shape: inputs={inputs.shape}, targets={targets.shape}")
    print(f"Number of batches: {len(loader)}")
