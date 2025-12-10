"""
Run model inference and extract hidden states at operand positions.
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from config import MODEL_NAME, DEVICE, HIDDEN_STATES_PATH, set_seed


# Default batch size for inference
INFERENCE_BATCH_SIZE = 8


def load_model_and_tokenizer():
    """Load the language model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Use fp16 for faster inference
        device_map=None  # We'll manually move to device
    )
    model = model.to(DEVICE)
    model.eval()

    print(f"Model loaded with {model.config.num_hidden_layers} layers")
    print(f"Hidden dimension: {model.config.hidden_size}")

    return model, tokenizer


def extract_hidden_states_batch(model, input_ids_list, pos_op1_list, pos_op2_list, pad_token_id):
    """
    Extract hidden states for a batch of examples.

    Args:
        model: The language model
        input_ids_list: List of input_ids (variable length)
        pos_op1_list: List of operand 1 positions
        pos_op2_list: List of operand 2 positions
        pad_token_id: Token ID for padding

    Returns:
        h1_batch: [batch_size, num_layers, hidden_dim]
        h2_batch: [batch_size, num_layers, hidden_dim]
        h_output_batch: [batch_size, num_layers, hidden_dim]
        logits_batch: [batch_size, vocab_size]
    """
    batch_size = len(input_ids_list)

    # Pad sequences to same length
    max_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids = []
    attention_mask = []
    seq_lengths = []

    for ids in input_ids_list:
        seq_len = len(ids)
        seq_lengths.append(seq_len)
        padding_len = max_len - seq_len
        # Pad on the LEFT so the last token position is consistent
        padded_ids = [pad_token_id] * padding_len + list(ids)
        mask = [0] * padding_len + [1] * seq_len
        padded_input_ids.append(padded_ids)
        attention_mask.append(mask)

    # Convert to tensors
    input_tensor = torch.tensor(padded_input_ids, device=DEVICE)
    attention_tensor = torch.tensor(attention_mask, device=DEVICE)

    with torch.no_grad():
        outputs = model(
            input_tensor,
            attention_mask=attention_tensor,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states  # Tuple of [batch, seq_len, hidden_dim]
        num_layers = len(hidden_states)
        hidden_dim = hidden_states[0].shape[-1]

        # Pre-allocate output arrays
        h1_batch = np.zeros((batch_size, num_layers, hidden_dim), dtype=np.float32)
        h2_batch = np.zeros((batch_size, num_layers, hidden_dim), dtype=np.float32)
        h_output_batch = np.zeros((batch_size, num_layers, hidden_dim), dtype=np.float32)

        # Extract hidden states at specific positions for each example
        for i in range(batch_size):
            # Adjust positions for left-padding
            padding_offset = max_len - seq_lengths[i]
            adj_pos_op1 = pos_op1_list[i] + padding_offset
            adj_pos_op2 = pos_op2_list[i] + padding_offset
            adj_pos_output = max_len - 1  # Last position (same for all after padding)

            for layer_idx in range(num_layers):
                h1_batch[i, layer_idx] = hidden_states[layer_idx][i, adj_pos_op1, :].cpu().float().numpy()
                h2_batch[i, layer_idx] = hidden_states[layer_idx][i, adj_pos_op2, :].cpu().float().numpy()
                h_output_batch[i, layer_idx] = hidden_states[layer_idx][i, adj_pos_output, :].cpu().float().numpy()

        # Get logits at last position for each example
        logits_batch = outputs.logits[:, -1, :].cpu().float().numpy()

    return h1_batch, h2_batch, h_output_batch, logits_batch


def decode_model_answer(model, tokenizer, input_ids, max_new_tokens=10, debug=False):
    """
    Decode the model's answer by generating tokens.

    Returns:
        predicted_answer: The integer answer predicted by the model (or None if can't parse)
    """
    with torch.no_grad():
        input_tensor = torch.tensor([input_ids], device=DEVICE)

        # Generate answer
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode the generated tokens (excluding the prompt)
        generated_tokens = output_ids[0, len(input_ids):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if debug:
            prompt_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"Prompt: {prompt_text}")
            print(f"Generated: '{generated_text}'")

        # Try to extract an integer from the generated text
        import re
        match = re.search(r'-?\d+', generated_text.strip())
        if match:
            try:
                predicted_answer = int(match.group())
                if debug:
                    print(f"Parsed answer: {predicted_answer}")
                return predicted_answer
            except ValueError:
                pass

        if debug:
            print("Failed to parse answer")

    return None


def decode_model_answers_batch(model, tokenizer, input_ids_list, max_new_tokens=10):
    """
    Decode model answers for a batch using batched generation.
    """
    import re

    batch_size = len(input_ids_list)
    if batch_size == 0:
        return []

    # Pad sequences for batched generation (left-padding for causal LM)
    max_len = max(len(ids) for ids in input_ids_list)
    pad_token_id = tokenizer.pad_token_id

    padded_input_ids = []
    attention_mask = []

    for ids in input_ids_list:
        padding_len = max_len - len(ids)
        padded_ids = [pad_token_id] * padding_len + list(ids)
        mask = [0] * padding_len + [1] * len(ids)
        padded_input_ids.append(padded_ids)
        attention_mask.append(mask)

    input_tensor = torch.tensor(padded_input_ids, device=DEVICE)
    attention_tensor = torch.tensor(attention_mask, device=DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            attention_mask=attention_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id
        )

    # Decode each sequence
    answers = []
    for i in range(batch_size):
        # Get generated tokens (after the padded input)
        generated_tokens = output_ids[i, max_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Extract integer from generated text
        match = re.search(r'-?\d+', generated_text.strip())
        if match:
            try:
                answers.append(int(match.group()))
            except ValueError:
                answers.append(None)
        else:
            answers.append(None)

    return answers


def process_dataset_inference(df, model, tokenizer, batch_size=INFERENCE_BATCH_SIZE):
    """
    Run inference on entire dataset and extract hidden states.

    Args:
        df: DataFrame with input_ids, pos_op1, pos_op2, correct_answer
        model: The language model
        tokenizer: The tokenizer
        batch_size: Batch size for inference

    Returns:
        Dictionary with:
            - h1_all: Shape [num_examples, num_layers, hidden_dim]
            - h2_all: Shape [num_examples, num_layers, hidden_dim]
            - h_output_all: Shape [num_examples, num_layers, hidden_dim]
            - Z_all: Shape [num_examples] - logit of predicted token
            - predicted_answers: List of predicted answers
            - is_correct: Boolean array
    """
    print(f"Running model inference with batch_size={batch_size}...")

    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size
    num_examples = len(df)

    # Pre-allocate arrays
    h1_all = np.zeros((num_examples, num_layers, hidden_dim), dtype=np.float32)
    h2_all = np.zeros((num_examples, num_layers, hidden_dim), dtype=np.float32)
    h_output_all = np.zeros((num_examples, num_layers, hidden_dim), dtype=np.float32)
    Z_all = np.zeros(num_examples, dtype=np.float32)
    predicted_answers = [None] * num_examples
    is_correct = np.zeros(num_examples, dtype=bool)

    pad_token_id = tokenizer.pad_token_id

    # Process in batches
    num_batches = (num_examples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Extracting hidden states"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_examples)
        batch_df = df.iloc[start_idx:end_idx]

        # Prepare batch data
        input_ids_list = batch_df['input_ids'].tolist()
        pos_op1_list = batch_df['pos_op1'].tolist()
        pos_op2_list = batch_df['pos_op2'].tolist()
        correct_answers = batch_df['correct_answer'].tolist()

        # Extract hidden states for batch
        h1_batch, h2_batch, h_output_batch, logits_batch = extract_hidden_states_batch(
            model, input_ids_list, pos_op1_list, pos_op2_list, pad_token_id
        )

        # Store hidden states
        h1_all[start_idx:end_idx] = h1_batch
        h2_all[start_idx:end_idx] = h2_batch
        h_output_all[start_idx:end_idx] = h_output_batch

        # Get predicted token and logit for each example
        for i, (input_ids, logits, correct_answer) in enumerate(
            zip(input_ids_list, logits_batch, correct_answers)
        ):
            idx = start_idx + i
            predicted_token_id = np.argmax(logits)
            Z_all[idx] = logits[predicted_token_id]

        # Decode answers (still sequential for generation)
        batch_answers = decode_model_answers_batch(model, tokenizer, input_ids_list)

        for i, (answer, correct_answer) in enumerate(zip(batch_answers, correct_answers)):
            idx = start_idx + i
            predicted_answers[idx] = answer
            is_correct[idx] = (answer is not None and answer == correct_answer)

            # Debug first 5 examples
            if idx < 5:
                if is_correct[idx]:
                    print(f"✓ Correct! Expected: {correct_answer}")
                else:
                    print(f"✗ Incorrect. Expected: {correct_answer}, Got: {answer}")

    print(f"\nModel accuracy: {is_correct.mean():.2%}")
    print("Accuracy by operation:")
    for op in df['op'].unique():
        op_mask = df['op'] == op
        op_accuracy = is_correct[op_mask].mean()
        print(f"  {op}: {op_accuracy:.2%}")

    return {
        'h1_all': h1_all,
        'h2_all': h2_all,
        'h_output_all': h_output_all,
        'Z_all': Z_all,
        'predicted_answers': predicted_answers,
        'is_correct': is_correct
    }


def save_hidden_states(data_dict, path=HIDDEN_STATES_PATH):
    """Save hidden states and predictions to disk."""
    np.savez_compressed(path, **data_dict)
    print(f"Hidden states saved to {path}")


def load_hidden_states(path=HIDDEN_STATES_PATH):
    """Load hidden states from disk."""
    data = np.load(path, allow_pickle=True)
    print(f"Hidden states loaded from {path}")
    return {key: data[key] for key in data.files}


if __name__ == "__main__":
    from dataset_generator import load_dataset
    from tokenizer_utils import process_dataset_tokenization

    set_seed()

    # Load and tokenize dataset
    df = load_dataset()
    model, tokenizer = load_model_and_tokenizer()
    df = process_dataset_tokenization(df, tokenizer)

    # Run inference
    hidden_states_data = process_dataset_inference(df, model, tokenizer)

    # Save results
    save_hidden_states(hidden_states_data)

    # Update dataframe with predictions
    df['predicted_answer'] = hidden_states_data['predicted_answers']
    df['is_correct'] = hidden_states_data['is_correct']

    # Save updated dataset
    from dataset_generator import save_dataset
    save_dataset(df)
