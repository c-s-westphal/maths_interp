"""
Run model inference and extract hidden states at operand positions.
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from config import MODEL_NAME, DEVICE, HIDDEN_STATES_PATH, set_seed


def load_model_and_tokenizer():
    """Load the language model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None  # We'll manually move to device
    )
    model = model.to(DEVICE)
    model.eval()

    print(f"Model loaded with {model.config.num_hidden_layers} layers")
    print(f"Hidden dimension: {model.config.hidden_size}")

    return model, tokenizer


def extract_hidden_states_single(model, input_ids, pos_op1, pos_op2):
    """
    Extract hidden states for a single example.

    Returns:
        h1_all_layers: List of tensors, one per layer (shape: [hidden_dim])
        h2_all_layers: List of tensors, one per layer (shape: [hidden_dim])
        logits: Output logits at the last position (shape: [vocab_size])
    """
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor([input_ids], device=DEVICE)

        # Run model with output_hidden_states=True
        outputs = model(input_tensor, output_hidden_states=True)

        # Extract hidden states
        # hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor has shape [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states

        # Extract operand representations from each layer
        h1_all_layers = []
        h2_all_layers = []

        for layer_idx in range(len(hidden_states)):
            h1 = hidden_states[layer_idx][0, pos_op1, :].cpu().numpy()
            h2 = hidden_states[layer_idx][0, pos_op2, :].cpu().numpy()
            h1_all_layers.append(h1)
            h2_all_layers.append(h2)

        # Extract logits at the last position
        logits = outputs.logits[0, -1, :].cpu().numpy()

    return h1_all_layers, h2_all_layers, logits


def decode_model_answer(model, tokenizer, input_ids, max_new_tokens=10, debug=False):
    """
    Decode the model's answer by generating tokens.

    Returns:
        predicted_answer: The integer answer predicted by the model (or None if can't parse)
        is_correct: Will be filled in later when comparing to ground truth
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
        # Look for the first sequence of digits (possibly with a minus sign)
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


def process_dataset_inference(df, model, tokenizer, batch_size=1):
    """
    Run inference on entire dataset and extract hidden states.

    Returns:
        Dictionary with:
            - h1_all: Shape [num_examples, num_layers, hidden_dim]
            - h2_all: Shape [num_examples, num_layers, hidden_dim]
            - Z_all: Shape [num_examples] - logit of predicted token
            - predicted_answers: List of predicted answers
            - is_correct: Boolean array
    """
    print("Running model inference and extracting hidden states...")

    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size
    num_examples = len(df)

    # Pre-allocate arrays
    h1_all = np.zeros((num_examples, num_layers, hidden_dim), dtype=np.float32)
    h2_all = np.zeros((num_examples, num_layers, hidden_dim), dtype=np.float32)
    Z_all = np.zeros(num_examples, dtype=np.float32)
    predicted_answers = []
    is_correct = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting hidden states"):
        input_ids = row['input_ids']
        pos_op1 = row['pos_op1']
        pos_op2 = row['pos_op2']
        correct_answer = row['correct_answer']

        # Extract hidden states
        h1_layers, h2_layers, logits = extract_hidden_states_single(
            model, input_ids, pos_op1, pos_op2
        )

        # Store hidden states
        for layer_idx in range(num_layers):
            h1_all[idx, layer_idx] = h1_layers[layer_idx]
            h2_all[idx, layer_idx] = h2_layers[layer_idx]

        # Get predicted token and its logit
        predicted_token_id = np.argmax(logits)
        Z_all[idx] = logits[predicted_token_id]

        # Decode model's answer (debug first 5 examples)
        debug = (idx < 5)
        predicted_answer = decode_model_answer(model, tokenizer, input_ids, debug=debug)
        predicted_answers.append(predicted_answer)

        # Check if correct
        if predicted_answer is not None and predicted_answer == correct_answer:
            is_correct.append(True)
            if debug:
                print(f"✓ Correct! Expected: {correct_answer}\n")
        else:
            is_correct.append(False)
            if debug:
                print(f"✗ Incorrect. Expected: {correct_answer}, Got: {predicted_answer}\n")

    is_correct = np.array(is_correct)

    print(f"\nModel accuracy: {is_correct.mean():.2%}")
    print("Accuracy by operation:")
    for op in df['op'].unique():
        op_mask = df['op'] == op
        op_accuracy = is_correct[op_mask].mean()
        print(f"  {op}: {op_accuracy:.2%}")

    return {
        'h1_all': h1_all,
        'h2_all': h2_all,
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
