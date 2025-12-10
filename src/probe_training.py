"""
Train numeric-decoding probes for each layer.

All probes are trained on the hidden state at the "=" position (h_output),
where the model has gathered information about both operands via attention.

Probes:
- probe_op1: h_output → x̂₁ (decode operand 1 from "=" position)
- probe_op2: h_output → x̂₂ (decode operand 2 from "=" position)
- probe_correct: h_output → ŷ (decode correct answer from "=" position)
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import pickle
import os
from config import (
    PROBE_HIDDEN_DIM, PROBE_LEARNING_RATE, PROBE_EPOCHS,
    PROBE_BATCH_SIZE, PROBE_VALIDATION_SPLIT, PROBES_DIR,
    FEATURES_PATH, DEVICE, set_seed
)


class NumericProbe(nn.Module):
    """
    Simple MLP probe: hidden_dim -> 64 -> 1
    """
    def __init__(self, input_dim, hidden_dim=PROBE_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            output: [batch_size, 1] - predicted number
            features: [batch_size, hidden_dim] - penultimate layer features
        """
        features = self.relu(self.fc1(x))
        output = self.fc2(features)
        return output, features


def train_probe(X_train, y_train, X_val, y_val, input_dim):
    """
    Train a single numeric probe.

    Returns:
        probe: Trained probe model
        val_r2: Validation R² score
        val_mae: Validation MAE
    """
    probe = NumericProbe(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LEARNING_RATE)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)

    # Training loop
    best_val_loss = float('inf')
    best_probe_state = None

    for epoch in range(PROBE_EPOCHS):
        probe.train()
        total_loss = 0
        num_batches = 0

        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), PROBE_BATCH_SIZE):
            batch_indices = indices[i:i+PROBE_BATCH_SIZE]
            X_batch = X_train_t[batch_indices]
            y_batch = y_train_t[batch_indices]

            optimizer.zero_grad()
            predictions, _ = probe(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Validation
        probe.eval()
        with torch.no_grad():
            val_predictions, _ = probe(X_val_t)
            val_loss = criterion(val_predictions, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_probe_state = probe.state_dict().copy()

    # Load best model
    probe.load_state_dict(best_probe_state)
    probe.eval()

    # Compute metrics
    with torch.no_grad():
        val_predictions, _ = probe(X_val_t)
        val_predictions_np = val_predictions.cpu().numpy().flatten()
        y_val_np = y_val

        val_r2 = r2_score(y_val_np, val_predictions_np)
        val_mae = mean_absolute_error(y_val_np, val_predictions_np)

    return probe, val_r2, val_mae


def extract_features_from_probe(probe, X):
    """
    Extract penultimate layer features from a trained probe.

    Args:
        probe: Trained NumericProbe
        X: Input hidden states [num_examples, hidden_dim]

    Returns:
        features: [num_examples, probe_hidden_dim]
    """
    probe.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        _, features = probe(X_t)
        return features.cpu().numpy()


def get_scalar_predictions(probe, X):
    """
    Get scalar predictions from a probe.

    Args:
        probe: Trained NumericProbe
        X: Input hidden states [num_examples, hidden_dim]

    Returns:
        predictions: [num_examples] scalar predictions
    """
    probe.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        predictions, _ = probe(X_t)
        return predictions.cpu().numpy().flatten()


def train_probes_all_layers(h_output_all, x1_values, x2_values, correct_answers):
    """
    Train numeric probes for all layers.

    ALL probes are trained on h_output (the "=" position), where the model
    has gathered information about both operands via attention.

    Args:
        h_output_all: [num_examples, num_layers, hidden_dim] - hidden states at "=" position
        x1_values: [num_examples] - ground truth operand 1
        x2_values: [num_examples] - ground truth operand 2
        correct_answers: [num_examples] - ground truth answers

    Returns:
        Dictionary with:
            - probes_op1: List of probes for operand 1 (from "=" position)
            - probes_op2: List of probes for operand 2 (from "=" position)
            - probes_correct: List of probes for correct answer (from "=" position)
            - F1_all: [num_examples, num_layers, probe_hidden_dim]
            - F2_all: [num_examples, num_layers, probe_hidden_dim]
            - F_correct_all: [num_examples, num_layers, probe_hidden_dim]
            - metrics: Dictionary of R² and MAE per layer
    """
    num_examples, num_layers, hidden_dim = h_output_all.shape

    probes_op1 = []
    probes_op2 = []
    probes_correct = []
    F1_all = np.zeros((num_examples, num_layers, PROBE_HIDDEN_DIM), dtype=np.float32)
    F2_all = np.zeros((num_examples, num_layers, PROBE_HIDDEN_DIM), dtype=np.float32)
    F_correct_all = np.zeros((num_examples, num_layers, PROBE_HIDDEN_DIM), dtype=np.float32)

    metrics = {
        'r2_op1': [],
        'mae_op1': [],
        'r2_op2': [],
        'mae_op2': [],
        'r2_correct': [],
        'mae_correct': []
    }

    print(f"Training probes for {num_layers} layers...")
    print(f"All probes trained on h_output ('=' position)")

    for layer_idx in tqdm(range(num_layers), desc="Training probes"):
        # Extract hidden states at "=" position for this layer
        h_output_layer = h_output_all[:, layer_idx, :]

        # Split data
        indices = np.arange(num_examples)
        train_idx, val_idx = train_test_split(
            indices, test_size=PROBE_VALIDATION_SPLIT, random_state=42
        )

        # Train probe for operand 1 (from "=" position)
        probe1, r2_1, mae_1 = train_probe(
            h_output_layer[train_idx], x1_values[train_idx],
            h_output_layer[val_idx], x1_values[val_idx],
            hidden_dim
        )
        probes_op1.append(probe1)
        metrics['r2_op1'].append(r2_1)
        metrics['mae_op1'].append(mae_1)

        # Train probe for operand 2 (from "=" position)
        probe2, r2_2, mae_2 = train_probe(
            h_output_layer[train_idx], x2_values[train_idx],
            h_output_layer[val_idx], x2_values[val_idx],
            hidden_dim
        )
        probes_op2.append(probe2)
        metrics['r2_op2'].append(r2_2)
        metrics['mae_op2'].append(mae_2)

        # Train probe for correct answer (from "=" position)
        probe_correct, r2_correct, mae_correct = train_probe(
            h_output_layer[train_idx], correct_answers[train_idx],
            h_output_layer[val_idx], correct_answers[val_idx],
            hidden_dim
        )
        probes_correct.append(probe_correct)
        metrics['r2_correct'].append(r2_correct)
        metrics['mae_correct'].append(mae_correct)

        # Extract features for all examples
        F1_all[:, layer_idx, :] = extract_features_from_probe(probe1, h_output_layer)
        F2_all[:, layer_idx, :] = extract_features_from_probe(probe2, h_output_layer)
        F_correct_all[:, layer_idx, :] = extract_features_from_probe(probe_correct, h_output_layer)

    print("\nProbe training complete!")
    print(f"Average R² (op1 from '='): {np.mean(metrics['r2_op1']):.3f}")
    print(f"Average R² (op2 from '='): {np.mean(metrics['r2_op2']):.3f}")
    print(f"Average R² (correct answer): {np.mean(metrics['r2_correct']):.3f}")

    return {
        'probes_op1': probes_op1,
        'probes_op2': probes_op2,
        'probes_correct': probes_correct,
        'F1_all': F1_all,
        'F2_all': F2_all,
        'F_correct_all': F_correct_all,
        'metrics': metrics
    }


def save_probe_results(probe_results, probes_dir=PROBES_DIR, features_path=FEATURES_PATH):
    """Save probe models and features."""
    # Save probe models
    for layer_idx, (probe1, probe2, probe_correct) in enumerate(zip(
        probe_results['probes_op1'], probe_results['probes_op2'], probe_results['probes_correct']
    )):
        torch.save(probe1.state_dict(), os.path.join(probes_dir, f'probe_op1_layer{layer_idx}.pt'))
        torch.save(probe2.state_dict(), os.path.join(probes_dir, f'probe_op2_layer{layer_idx}.pt'))
        torch.save(probe_correct.state_dict(), os.path.join(probes_dir, f'probe_correct_layer{layer_idx}.pt'))

    # Save features and metrics
    np.savez_compressed(
        features_path,
        F1_all=probe_results['F1_all'],
        F2_all=probe_results['F2_all'],
        F_correct_all=probe_results['F_correct_all'],
        **probe_results['metrics']
    )

    print(f"Probes saved to {probes_dir}")
    print(f"Features saved to {features_path}")


def load_probe_results(probes_dir=PROBES_DIR, features_path=FEATURES_PATH, hidden_dim=None):
    """Load probe models and features."""
    data = np.load(features_path)

    # Determine number of layers
    F1_all = data['F1_all']
    F2_all = data['F2_all']
    F_correct_all = data['F_correct_all']
    num_layers = F1_all.shape[1]

    probes_op1 = []
    probes_op2 = []
    probes_correct = []

    for layer_idx in range(num_layers):
        probe1 = NumericProbe(hidden_dim)
        probe1.load_state_dict(torch.load(
            os.path.join(probes_dir, f'probe_op1_layer{layer_idx}.pt')
        ))
        probe1 = probe1.to(DEVICE)
        probes_op1.append(probe1)

        probe2 = NumericProbe(hidden_dim)
        probe2.load_state_dict(torch.load(
            os.path.join(probes_dir, f'probe_op2_layer{layer_idx}.pt')
        ))
        probe2 = probe2.to(DEVICE)
        probes_op2.append(probe2)

        probe_correct = NumericProbe(hidden_dim)
        probe_correct.load_state_dict(torch.load(
            os.path.join(probes_dir, f'probe_correct_layer{layer_idx}.pt')
        ))
        probe_correct = probe_correct.to(DEVICE)
        probes_correct.append(probe_correct)

    metrics = {
        'r2_op1': data['r2_op1'],
        'mae_op1': data['mae_op1'],
        'r2_op2': data['r2_op2'],
        'mae_op2': data['mae_op2'],
        'r2_correct': data['r2_correct'],
        'mae_correct': data['mae_correct']
    }

    return {
        'probes_op1': probes_op1,
        'probes_op2': probes_op2,
        'probes_correct': probes_correct,
        'F1_all': F1_all,
        'F2_all': F2_all,
        'F_correct_all': F_correct_all,
        'metrics': metrics
    }


if __name__ == "__main__":
    from dataset_generator import load_dataset
    from model_inference import load_hidden_states

    set_seed()

    # Load data
    df = load_dataset()
    hidden_states = load_hidden_states()

    h_output_all = hidden_states['h_output_all']
    x1_values = df['x1'].values
    x2_values = df['x2'].values
    correct_answers = df['correct_answer'].values

    # Train probes (all from "=" position)
    probe_results = train_probes_all_layers(
        h_output_all, x1_values, x2_values, correct_answers
    )

    # Save results
    save_probe_results(probe_results)
