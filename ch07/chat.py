"""
Chapter 7: Simple Chat Interface.

Interactive REPL for chatting with an instruction-tuned GPT model.
Formats user input as an instruction prompt and generates responses.

Usage:
    python -m ch07.chat                          # Use default small model
    python -m ch07.chat --checkpoint path/to/pt   # Load a checkpoint

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 7.
"""

import sys
import os
import argparse
import torch
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ch04.gpt_model import GPTModel
from ch04.config import GPTConfig
from ch07.instruction_tuning import generate_response


def load_model(checkpoint_path: str = None, device: str = "cpu"):
    """
    Load a GPT model, optionally from a checkpoint.

    If no checkpoint is provided, creates a randomly initialized
    small model (useful for testing the interface).

    Args:
        checkpoint_path: path to a .pt checkpoint file
        device:          device to load model on

    Returns:
        (model, tokenizer)
    """
    # Small model config (must match the training config)
    cfg = GPTConfig(
        vocab_size=50_257,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_seq_len=256,
        dropout=0.0,  # No dropout during inference
    )

    model = GPTModel(cfg)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"   Step {checkpoint.get('step', '?')} | "
              f"Val loss: {checkpoint.get('val_loss', '?'):.4f}")
    else:
        print("⚠️  No checkpoint loaded — using random weights")
        print("   (Run instruction_tuning.py first to train a model)")

    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    return model, tokenizer


def chat_loop(model, tokenizer, device: str = "cpu"):
    """
    Interactive chat REPL.

    Type an instruction and get a response from the model.
    Special commands:
        - 'quit' or 'exit': end the session
        - 'temp <value>':   change temperature (e.g., 'temp 0.5')
        - 'topk <value>':   change top-k (e.g., 'topk 10')

    Args:
        model:     GPT model
        tokenizer: tiktoken tokenizer
        device:    device
    """
    temperature = 0.7
    top_k = 25
    max_new_tokens = 100

    print("\n" + "=" * 60)
    print("🤖 Instruction-Tuned GPT Chat")
    print("=" * 60)
    print("Type your instruction and press Enter.")
    print("Commands: 'quit', 'temp <val>', 'topk <val>'")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        # ---- Special commands ----
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if user_input.lower().startswith("temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"   🌡️ Temperature set to {temperature}")
            except (ValueError, IndexError):
                print("   ❌ Usage: temp <value> (e.g., temp 0.5)")
            continue

        if user_input.lower().startswith("topk "):
            try:
                top_k = int(user_input.split()[1])
                print(f"   🎯 Top-k set to {top_k}")
            except (ValueError, IndexError):
                print("   ❌ Usage: topk <value> (e.g., topk 10)")
            continue

        # ---- Generate response ----
        response = generate_response(
            model, user_input, tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        print(f"🤖 Bot: {response}\n")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with instruction-tuned GPT")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="ch07/checkpoints/best_instruction.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cpu/cuda/mps)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint, args.device)
    chat_loop(model, tokenizer, args.device)
