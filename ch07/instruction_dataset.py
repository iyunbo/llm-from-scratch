"""
Chapter 7: Instruction Dataset for Fine-Tuning.

Creates and manages instruction-response pairs in Alpaca format:
  1. Synthetic instruction dataset (~60 examples)
  2. Prompt template formatting
  3. PyTorch Dataset with tokenization and padding

Alpaca format:
    {"instruction": "...", "input": "...", "output": "..."}

The "input" field is optional context for the instruction.
Most examples use instruction + output only.

Reference: "Build a Large Language Model from Scratch" by Sebastian Raschka, Chapter 7.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import tiktoken


# ──────────────────────────────────────────────────────────────
# 1. Synthetic Instruction Dataset (~60 examples)
# ──────────────────────────────────────────────────────────────

INSTRUCTIONS = [
    # === Q&A / Factual ===
    {"instruction": "What is the capital of France?",
     "output": "The capital of France is Paris."},
    {"instruction": "What is the largest planet in our solar system?",
     "output": "The largest planet in our solar system is Jupiter."},
    {"instruction": "Who wrote Romeo and Juliet?",
     "output": "Romeo and Juliet was written by William Shakespeare."},
    {"instruction": "What is the speed of light?",
     "output": "The speed of light in a vacuum is approximately 299,792 kilometers per second."},
    {"instruction": "What is photosynthesis?",
     "output": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."},
    {"instruction": "How many continents are there?",
     "output": "There are seven continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America."},
    {"instruction": "What is the boiling point of water?",
     "output": "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."},
    {"instruction": "Who painted the Mona Lisa?",
     "output": "The Mona Lisa was painted by Leonardo da Vinci."},
    {"instruction": "What is the chemical formula for water?",
     "output": "The chemical formula for water is H2O."},
    {"instruction": "What year did World War II end?",
     "output": "World War II ended in 1945."},
    {"instruction": "What is the tallest mountain in the world?",
     "output": "Mount Everest is the tallest mountain in the world, standing at 8,849 meters above sea level."},
    {"instruction": "What is DNA?",
     "output": "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development and functioning of living organisms."},

    # === Translation ===
    {"instruction": "Translate 'hello' to Spanish.",
     "output": "Hello in Spanish is 'hola'."},
    {"instruction": "Translate 'thank you' to French.",
     "output": "Thank you in French is 'merci'."},
    {"instruction": "Translate 'goodbye' to Japanese.",
     "output": "Goodbye in Japanese is 'sayonara'."},
    {"instruction": "Translate 'good morning' to German.",
     "output": "Good morning in German is 'Guten Morgen'."},
    {"instruction": "How do you say 'I love you' in Italian?",
     "output": "I love you in Italian is 'Ti amo'."},

    # === Summarization ===
    {"instruction": "Summarize: The quick brown fox jumps over the lazy dog.",
     "output": "A fox jumps over a dog."},
    {"instruction": "Summarize: Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance without being explicitly programmed.",
     "output": "Machine learning lets systems learn and improve from data automatically, without explicit programming."},
    {"instruction": "Summarize: The Great Wall of China is a series of fortifications built along the northern borders of China to protect against various nomadic groups. Construction began in the 7th century BC.",
     "output": "The Great Wall of China is an ancient defensive structure built along China's northern borders starting in the 7th century BC."},
    {"instruction": "Summarize: Python is a high-level programming language known for its readability and versatility. It supports multiple programming paradigms and has a large standard library.",
     "output": "Python is a readable, versatile high-level language with multi-paradigm support and a rich standard library."},
    {"instruction": "Summarize: The human brain contains approximately 86 billion neurons, each connected to thousands of other neurons through synapses, forming an incredibly complex network.",
     "output": "The human brain has about 86 billion neurons connected through synapses in a complex network."},

    # === Creative Writing ===
    {"instruction": "Write a haiku about rain.",
     "output": "Drops fall from the sky\nPuddles form on the gray ground\nNature drinks the rain"},
    {"instruction": "Write a haiku about mountains.",
     "output": "Peaks touch cloudy skies\nSilent guardians of earth\nSnow crowns ancient stone"},
    {"instruction": "Write a short poem about the ocean.",
     "output": "Waves crash on the shore,\nEndless blue stretching beyond,\nSalt breeze fills the air,\nThe ocean sings its old song."},
    {"instruction": "Write a one-sentence story about a robot.",
     "output": "The last robot on Earth spent its days painting sunsets it had only seen in photographs."},
    {"instruction": "Write a limerick about a cat.",
     "output": "There once was a cat on a mat,\nWho dreamed it could fly like a bat,\nIt leaped in the air,\nWith hardly a care,\nAnd landed right back where it sat."},

    # === Math / Logic ===
    {"instruction": "What is 15 multiplied by 7?",
     "output": "15 multiplied by 7 is 105."},
    {"instruction": "What is the square root of 144?",
     "output": "The square root of 144 is 12."},
    {"instruction": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
     "output": "The train travels 150 miles (60 mph × 2.5 hours = 150 miles)."},
    {"instruction": "What is 20% of 250?",
     "output": "20% of 250 is 50."},
    {"instruction": "Convert 100 degrees Fahrenheit to Celsius.",
     "output": "100°F is approximately 37.8°C. The formula is: (F - 32) × 5/9 = (100 - 32) × 5/9 ≈ 37.8°C."},

    # === Explanation / How-to ===
    {"instruction": "Explain what a neural network is in simple terms.",
     "output": "A neural network is a computer system inspired by the human brain. It consists of layers of connected nodes that process information. Each connection has a weight that adjusts during learning, allowing the network to recognize patterns and make predictions."},
    {"instruction": "How does a transformer model work?",
     "output": "A transformer processes all input tokens simultaneously using self-attention, which lets each token attend to every other token. It uses positional encodings to understand word order. The architecture has an encoder-decoder structure, though modern LLMs often use decoder-only designs."},
    {"instruction": "What is the difference between supervised and unsupervised learning?",
     "output": "Supervised learning uses labeled data to train models to predict outputs from inputs. Unsupervised learning finds patterns in unlabeled data without predefined targets. Examples: classification (supervised) vs clustering (unsupervised)."},
    {"instruction": "Explain what an API is.",
     "output": "An API (Application Programming Interface) is a set of rules that allows different software programs to communicate with each other. It defines the methods and data formats that programs can use to request and exchange information."},
    {"instruction": "What is recursion in programming?",
     "output": "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems. It needs a base case to stop and a recursive case that moves toward the base case. Example: calculating factorial where n! = n × (n-1)!."},
    {"instruction": "Explain the concept of supply and demand.",
     "output": "Supply and demand is an economic model. When demand for a product increases and supply stays the same, prices rise. When supply increases and demand stays the same, prices fall. The market price settles where supply equals demand (equilibrium)."},

    # === Classification / Analysis ===
    {"instruction": "Is the following sentence positive or negative: 'I had a wonderful time at the park today.'",
     "output": "The sentence is positive. Words like 'wonderful' indicate a positive sentiment."},
    {"instruction": "Is the following sentence positive or negative: 'The movie was boring and a waste of time.'",
     "output": "The sentence is negative. Words like 'boring' and 'waste of time' indicate negative sentiment."},
    {"instruction": "Classify the following as animal, vegetable, or mineral: diamond.",
     "output": "Diamond is a mineral. It is a naturally occurring crystalline form of carbon."},
    {"instruction": "Classify the following as animal, vegetable, or mineral: broccoli.",
     "output": "Broccoli is a vegetable. It is an edible green plant in the cabbage family."},

    # === Coding ===
    {"instruction": "Write a Python function to reverse a string.",
     "output": "def reverse_string(s):\n    return s[::-1]"},
    {"instruction": "Write a Python function to check if a number is even.",
     "output": "def is_even(n):\n    return n % 2 == 0"},
    {"instruction": "What does the 'len' function do in Python?",
     "output": "The len() function in Python returns the number of items in an object, such as the length of a string, list, tuple, or dictionary."},
    {"instruction": "Write a Python list comprehension to get squares of numbers 1 to 5.",
     "output": "squares = [x**2 for x in range(1, 6)]  # Result: [1, 4, 9, 16, 25]"},

    # === Rewriting / Formatting ===
    {"instruction": "Rewrite in formal English: 'gonna grab some coffee brb'.",
     "output": "I will be stepping away momentarily to get some coffee. I shall return shortly."},
    {"instruction": "Make this sentence shorter: 'In my personal opinion, I think that the weather today is quite nice and pleasant.'",
     "output": "The weather is nice today."},
    {"instruction": "Fix the grammar: 'Me and him goes to the store yesterday.'",
     "output": "He and I went to the store yesterday."},
    {"instruction": "Convert to a bullet list: 'Python is popular for web development, data science, automation, and AI.'",
     "output": "Python is popular for:\n- Web development\n- Data science\n- Automation\n- AI"},

    # === Comparison ===
    {"instruction": "What is the difference between a list and a tuple in Python?",
     "output": "Lists are mutable (can be changed after creation) and use square brackets []. Tuples are immutable (cannot be changed) and use parentheses (). Tuples are faster and can be used as dictionary keys."},
    {"instruction": "Compare cats and dogs as pets.",
     "output": "Cats are independent, low-maintenance, and good for small spaces. Dogs are social, need more exercise and attention, but offer strong companionship. Both make excellent pets depending on lifestyle."},

    # === Advice / Suggestions ===
    {"instruction": "Give me three tips for learning a new programming language.",
     "output": "1. Start with the basics and build small projects to practice.\n2. Read other people's code to learn idioms and best practices.\n3. Be consistent — practice a little every day rather than cramming."},
    {"instruction": "What are some good habits for software engineers?",
     "output": "Write clean, readable code with meaningful names. Write tests for your code. Use version control (Git). Review others' code and accept reviews of yours. Keep learning new technologies and patterns."},

    # === Miscellaneous ===
    {"instruction": "List the primary colors.",
     "output": "The primary colors are red, blue, and yellow (in traditional color theory) or red, green, and blue (in additive/light color mixing - RGB)."},
    {"instruction": "Name three renewable energy sources.",
     "output": "Three renewable energy sources are: solar energy, wind energy, and hydroelectric power."},
    {"instruction": "What is the Fibonacci sequence?",
     "output": "The Fibonacci sequence is a series where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, and so on. It appears frequently in nature and mathematics."},
    {"instruction": "Explain what a black hole is.",
     "output": "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their life cycle. The boundary around a black hole is called the event horizon."},
    {"instruction": "What causes seasons on Earth?",
     "output": "Seasons are caused by Earth's axial tilt of about 23.5 degrees. As Earth orbits the Sun, different hemispheres receive varying amounts of direct sunlight throughout the year, creating seasonal changes in temperature and daylight."},
]


# ──────────────────────────────────────────────────────────────
# 2. Prompt Template
# ──────────────────────────────────────────────────────────────

def format_instruction(entry: dict) -> str:
    """
    Format an instruction entry into the Alpaca-style prompt template.

    Template:
        ### Instruction:
        {instruction}

        ### Response:
        {output}

    If the entry has an "input" field (additional context), it's appended
    after the instruction.

    Args:
        entry: dict with "instruction", optional "input", and "output" keys

    Returns:
        Formatted prompt string
    """
    instruction = entry["instruction"]
    if entry.get("input"):
        instruction += f"\n{entry['input']}"

    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{entry['output']}"
    )


# ──────────────────────────────────────────────────────────────
# 3. PyTorch Dataset
# ──────────────────────────────────────────────────────────────

class InstructionDataset(Dataset):
    """
    PyTorch Dataset for instruction tuning.

    Each sample is tokenized from the formatted prompt template.
    All sequences are padded to max_length for batching.

    The dataset returns (input_ids, target_ids) pairs where:
        - input_ids:  tokens[:-1]  (everything except last token)
        - target_ids: tokens[1:]   (everything except first token)

    This is the standard causal LM setup: predict next token.
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 256,
        pad_token_id: int = 50256,  # GPT-2 uses <|endoftext|> as pad
    ):
        """
        Args:
            data:         list of instruction dicts
            tokenizer:    tiktoken tokenizer
            max_length:   max sequence length (pad/truncate to this)
            pad_token_id: token ID used for padding
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        # Pre-tokenize all examples
        self.encoded = []
        for entry in data:
            text = format_instruction(entry)
            token_ids = tokenizer.encode(text, allowed_special=set())

            # Truncate if too long (leave room for shifting)
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            # Pad if too short
            padding_len = max_length - len(token_ids)
            token_ids = token_ids + [pad_token_id] * padding_len

            self.encoded.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        token_ids = self.encoded[idx]

        # Input: all tokens except the last
        input_ids = token_ids[:-1]    # (max_length - 1,)

        # Target: all tokens except the first (shifted by 1)
        target_ids = token_ids[1:]    # (max_length - 1,)

        return input_ids, target_ids


# ──────────────────────────────────────────────────────────────
# 4. Helper: Create DataLoaders
# ──────────────────────────────────────────────────────────────

def create_dataloaders(
    data: list[dict] = None,
    train_ratio: float = 0.85,
    max_length: int = 256,
    batch_size: int = 4,
    seed: int = 42,
) -> tuple:
    """
    Create train and validation DataLoaders from instruction data.

    Args:
        data:        list of instruction dicts (defaults to INSTRUCTIONS)
        train_ratio: fraction of data for training
        max_length:  max sequence length
        batch_size:  batch size
        seed:        random seed for reproducible split

    Returns:
        (train_loader, val_loader, tokenizer)
    """
    if data is None:
        data = INSTRUCTIONS

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = InstructionDataset(data, tokenizer, max_length=max_length)

    # Split into train / validation
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True,  # drop incomplete batches for stable training
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False,
    )

    print(f"📊 Dataset: {len(data)} examples")
    print(f"   Train: {train_size} | Val: {val_size}")
    print(f"   Max length: {max_length} | Batch size: {batch_size}")

    return train_loader, val_loader, tokenizer


# ──────────────────────────────────────────────────────────────
# 5. Main: Quick Test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Total instructions: {len(INSTRUCTIONS)}\n")

    # Show a few formatted examples
    for i in range(3):
        print(f"--- Example {i+1} ---")
        print(format_instruction(INSTRUCTIONS[i]))
        print()

    # Test DataLoader creation
    train_loader, val_loader, tokenizer = create_dataloaders(batch_size=2)

    # Peek at one batch
    for input_ids, target_ids in train_loader:
        print(f"\nBatch shapes: input={input_ids.shape}, target={target_ids.shape}")
        # Decode first example
        text = tokenizer.decode(input_ids[0].tolist())
        print(f"First example (decoded): {text[:200]}...")
        break
