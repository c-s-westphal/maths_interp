"""
Tokenization utilities and operand position finding.
"""
import torch
from transformers import AutoTokenizer
from typing import Tuple, List
from config import MODEL_NAME


def find_subsequence(seq: List[int], subseq: List[int]) -> int:
    """
    Find the starting index of a subsequence in a sequence.
    Returns -1 if not found.
    """
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            return i
    return -1


def find_operand_positions(prompt: str, x1: int, x2: int, tokenizer) -> Tuple[List[int], int, int]:
    """
    Tokenize the prompt and find the positions of the last tokens of x1 and x2.

    Returns:
        input_ids: List of token IDs
        pos_op1: Position of last token of operand 1
        pos_op2: Position of last token of operand 2
    """
    # Tokenize the full prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # Tokenize x1 and x2 separately
    str_x1 = str(x1)
    str_x2 = str(x2)

    # We need to tokenize with a space prefix to match context
    # Try different tokenization contexts to find the match
    x1_tokens_variants = [
        tokenizer.encode(str_x1, add_special_tokens=False),
        tokenizer.encode(" " + str_x1, add_special_tokens=False),
    ]

    x2_tokens_variants = [
        tokenizer.encode(str_x2, add_special_tokens=False),
        tokenizer.encode(" " + str_x2, add_special_tokens=False),
    ]

    # Find x1 position
    pos_op1 = -1
    x1_tokens = None
    for variant in x1_tokens_variants:
        idx = find_subsequence(input_ids, variant)
        if idx != -1:
            x1_tokens = variant
            pos_op1 = idx + len(variant) - 1  # Last token position
            break

    # Find x2 position (search after x1)
    pos_op2 = -1
    x2_tokens = None
    for variant in x2_tokens_variants:
        idx = find_subsequence(input_ids, variant)
        if idx != -1 and (pos_op1 == -1 or idx > pos_op1):
            x2_tokens = variant
            pos_op2 = idx + len(variant) - 1  # Last token position
            break

    # Sanity check
    if pos_op1 == -1 or pos_op2 == -1:
        raise ValueError(f"Could not find operand positions for prompt: {prompt}")

    return input_ids, pos_op1, pos_op2


def process_dataset_tokenization(df, tokenizer):
    """
    Process entire dataset and add tokenization information.

    Adds columns:
        - input_ids
        - pos_op1
        - pos_op2
    """
    print("Tokenizing dataset and finding operand positions...")

    input_ids_list = []
    pos_op1_list = []
    pos_op2_list = []

    skipped = 0
    for idx, row in df.iterrows():
        try:
            input_ids, pos_op1, pos_op2 = find_operand_positions(
                row['prompt'], row['x1'], row['x2'], tokenizer
            )
            input_ids_list.append(input_ids)
            pos_op1_list.append(pos_op1)
            pos_op2_list.append(pos_op2)
        except ValueError as e:
            # Skip examples with tokenization issues
            input_ids_list.append(None)
            pos_op1_list.append(-1)
            pos_op2_list.append(-1)
            skipped += 1

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(df)} examples")

    df['input_ids'] = input_ids_list
    df['pos_op1'] = pos_op1_list
    df['pos_op2'] = pos_op2_list

    # Remove skipped examples
    df = df[df['pos_op1'] != -1].reset_index(drop=True)

    print(f"Tokenization complete. Skipped {skipped} examples.")
    print(f"Final dataset size: {len(df)}")

    return df


if __name__ == "__main__":
    # Test tokenization
    from dataset_generator import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = load_dataset()

    # Test on a few examples
    for i in range(5):
        row = df.iloc[i]
        input_ids, pos_op1, pos_op2 = find_operand_positions(
            row['prompt'], row['x1'], row['x2'], tokenizer
        )
        print(f"\nPrompt: {row['prompt']}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids)}")
        print(f"x1={row['x1']} at position {pos_op1}, token: {tokenizer.decode([input_ids[pos_op1]])}")
        print(f"x2={row['x2']} at position {pos_op2}, token: {tokenizer.decode([input_ids[pos_op2]])}")
