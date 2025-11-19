"""
Generate synthetic arithmetic dataset.
"""
import random
import pandas as pd
import pickle
from typing import List, Dict
from config import (
    OPERATIONS, DIFFICULTY_LEVELS, DIFFICULTY_RANGES,
    EXAMPLES_PER_OP_DIFFICULTY, DATASET_PATH, set_seed,
    USE_FEW_SHOT, FEW_SHOT_EXAMPLES
)


def compute_answer(x1: int, x2: int, op: str) -> int:
    """Compute the correct answer for a given operation."""
    if op == "add":
        return x1 + x2
    elif op == "sub":
        return x1 - x2
    elif op == "mul":
        return x1 * x2
    elif op == "div":
        return x1 // x2  # Integer division
    elif op == "max":
        return max(x1, x2)
    elif op == "min":
        return min(x1, x2)
    else:
        raise ValueError(f"Unknown operation: {op}")


def format_prompt(x1: int, x2: int, op: str) -> str:
    """Format the prompt for a given operation."""
    # Base query without few-shot examples
    if op == "add":
        query = f"{x1} + {x2} ="
    elif op == "sub":
        query = f"{x1} - {x2} ="
    elif op == "mul":
        query = f"{x1} * {x2} ="
    elif op == "div":
        query = f"{x1} / {x2} ="
    elif op == "max":
        query = f"max({x1}, {x2}) ="
    elif op == "min":
        query = f"min({x1}, {x2}) ="
    else:
        raise ValueError(f"Unknown operation: {op}")

    # Add few-shot examples if enabled
    if USE_FEW_SHOT:
        few_shot = FEW_SHOT_EXAMPLES.get(op, "")
        return few_shot + query
    else:
        return query


def generate_example(op: str, difficulty: str) -> Dict:
    """Generate a single arithmetic example."""
    min_val, max_val = DIFFICULTY_RANGES[difficulty]

    # Sample two integers
    x1 = random.randint(min_val, max_val)
    x2 = random.randint(min_val, max_val)

    # For subtraction, ensure x1 >= x2 to avoid negative results
    if op == "sub" and x1 < x2:
        x1, x2 = x2, x1

    # For division, ensure x1 is divisible by x2 for integer results
    if op == "div":
        # Generate quotient and divisor, then compute x1 = quotient * x2
        quotient = random.randint(max(2, min_val // max_val), max_val // max(2, min_val))
        x2 = random.randint(max(2, min_val), min(max_val, max_val // 2))
        x1 = quotient * x2

    # Compute correct answer
    correct_answer = compute_answer(x1, x2, op)

    # Format prompt
    prompt = format_prompt(x1, x2, op)

    return {
        "prompt": prompt,
        "x1": x1,
        "x2": x2,
        "op": op,
        "difficulty": difficulty,
        "correct_answer": correct_answer
    }


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    """Generate the full arithmetic dataset."""
    set_seed(seed)

    examples = []

    for op in OPERATIONS:
        for difficulty in DIFFICULTY_LEVELS:
            for _ in range(EXAMPLES_PER_OP_DIFFICULTY):
                example = generate_example(op, difficulty)
                examples.append(example)

    df = pd.DataFrame(examples)

    print(f"Generated {len(df)} examples:")
    print(df.groupby(['op', 'difficulty']).size())

    return df


def save_dataset(df: pd.DataFrame, path: str = DATASET_PATH):
    """Save dataset to disk."""
    with open(path, 'wb') as f:
        pickle.dump(df, f)
    print(f"Dataset saved to {path}")


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load dataset from disk."""
    with open(path, 'rb') as f:
        df = pickle.load(f)
    print(f"Dataset loaded from {path}")
    return df


if __name__ == "__main__":
    # Generate and save dataset
    df = generate_dataset()
    save_dataset(df)

    # Display sample examples
    print("\nSample examples:")
    print(df.sample(10)[["prompt", "x1", "x2", "op", "difficulty", "correct_answer"]])
