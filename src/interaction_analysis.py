"""
Compute interaction scores per layer and operation.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import (
    INTERACTION_PREDICTOR_HIDDEN_DIM, INTERACTION_LEARNING_RATE,
    INTERACTION_EPOCHS, INTERACTION_BATCH_SIZE, DEVICE,
    INTERACTION_SCORES_PATH, OPERATIONS, set_seed
)


class InteractionPredictor(nn.Module):
    """
    Simple MLP predictor: [F1, F2] -> Z
    """
    def __init__(self, input_dim, hidden_dim=INTERACTION_PREDICTOR_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)


def train_predictor(X_train, Z_train, X_val, Z_val):
    """
    Train interaction predictor.

    Returns:
        predictor: Trained model
        val_score: Validation score (negative MSE, so higher is better)
    """
    input_dim = X_train.shape[1]
    predictor = InteractionPredictor(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=INTERACTION_LEARNING_RATE)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    Z_train_t = torch.FloatTensor(Z_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    Z_val_t = torch.FloatTensor(Z_val).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(INTERACTION_EPOCHS):
        predictor.train()

        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), INTERACTION_BATCH_SIZE):
            batch_indices = indices[i:i+INTERACTION_BATCH_SIZE]
            X_batch = X_train_t[batch_indices]
            Z_batch = Z_train_t[batch_indices]

            optimizer.zero_grad()
            predictions = predictor(X_batch)
            loss = criterion(predictions, Z_batch)
            loss.backward()
            optimizer.step()

        # Validation
        predictor.eval()
        with torch.no_grad():
            val_predictions = predictor(X_val_t)
            val_loss = criterion(val_predictions, Z_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = predictor.state_dict().copy()

    # Load best model
    predictor.load_state_dict(best_state)
    predictor.eval()

    # Compute validation score (negative MSE, so higher is better)
    with torch.no_grad():
        val_predictions = predictor(X_val_t)
        val_mse = criterion(val_predictions, Z_val_t).item()
        val_score = -val_mse

    return predictor, val_score


def evaluate_predictor(predictor, X, Z):
    """Evaluate predictor and return score (negative MSE)."""
    predictor.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        Z_t = torch.FloatTensor(Z).to(DEVICE)
        predictions = predictor(X_t)
        mse = nn.MSELoss()(predictions, Z_t).item()
        return -mse


def compute_interaction_score(F1, F2, Z, verbose=False):
    """
    Compute interaction score using shuffling approach.

    Returns:
        interaction_score: float
        component_scores: dict with detailed breakdown
    """
    # Concatenate features
    X_all = np.concatenate([F1, F2], axis=1)

    # Split into train/val
    indices = np.arange(len(X_all))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train = X_all[train_idx]
    Z_train = Z[train_idx]
    X_val = X_all[val_idx]
    Z_val = Z[val_idx]

    F1_train = F1[train_idx]
    F2_train = F2[train_idx]
    F1_val = F1[val_idx]
    F2_val = F2[val_idx]

    # Train predictor
    predictor, S_all = train_predictor(X_train, Z_train, X_val, Z_val)

    # Compute shuffled scores
    # Shuffle F1
    np.random.seed(42)
    shuffle_idx_F1 = np.random.permutation(len(F1_val))
    F1_val_shuffled = F1_val[shuffle_idx_F1]
    X_shuf_F1 = np.concatenate([F1_val_shuffled, F2_val], axis=1)
    S_shuf_F1 = evaluate_predictor(predictor, X_shuf_F1, Z_val)

    # Shuffle F2
    np.random.seed(43)
    shuffle_idx_F2 = np.random.permutation(len(F2_val))
    F2_val_shuffled = F2_val[shuffle_idx_F2]
    X_shuf_F2 = np.concatenate([F1_val, F2_val_shuffled], axis=1)
    S_shuf_F2 = evaluate_predictor(predictor, X_shuf_F2, Z_val)

    # Shuffle both
    X_shuf_both = np.concatenate([F1_val_shuffled, F2_val_shuffled], axis=1)
    S_shuf_both = evaluate_predictor(predictor, X_shuf_both, Z_val)

    # Compute interaction components
    delta_1 = S_all - S_shuf_F1
    delta_2 = S_all - S_shuf_F2
    delta_12 = S_all - S_shuf_both

    interaction_score = max(0.0, delta_12 - 0.5 * (delta_1 + delta_2))

    component_scores = {
        'S_all': S_all,
        'S_shuf_F1': S_shuf_F1,
        'S_shuf_F2': S_shuf_F2,
        'S_shuf_both': S_shuf_both,
        'delta_1': delta_1,
        'delta_2': delta_2,
        'delta_12': delta_12,
        'interaction': interaction_score
    }

    if verbose:
        print(f"  S_all: {S_all:.4f}")
        print(f"  S_shuf_F1: {S_shuf_F1:.4f}, Δ1: {delta_1:.4f}")
        print(f"  S_shuf_F2: {S_shuf_F2:.4f}, Δ2: {delta_2:.4f}")
        print(f"  S_shuf_both: {S_shuf_both:.4f}, Δ12: {delta_12:.4f}")
        print(f"  Interaction: {interaction_score:.4f}")

    return interaction_score, component_scores


def compute_all_interactions(F1_all, F2_all, Z_all, df, filter_by=None):
    """
    Compute interaction scores for all layers and operations.

    Args:
        F1_all: [num_examples, num_layers, feature_dim]
        F2_all: [num_examples, num_layers, feature_dim]
        Z_all: [num_examples]
        df: DataFrame with metadata (op, difficulty, is_correct)
        filter_by: Optional dict to filter examples (e.g., {'is_correct': True})

    Returns:
        DataFrame with columns: layer, op, interaction_score
    """
    num_layers = F1_all.shape[1]

    results = []

    print("Computing interaction scores...")

    for op in tqdm(OPERATIONS, desc="Operations"):
        # Filter by operation
        op_mask = df['op'] == op

        # Apply additional filters
        if filter_by:
            for key, value in filter_by.items():
                op_mask &= (df[key] == value)

        op_indices = np.where(op_mask)[0]

        if len(op_indices) < 100:  # Skip if too few examples
            print(f"Skipping {op} (only {len(op_indices)} examples)")
            continue

        for layer_idx in range(num_layers):
            # Extract features for this layer and operation
            F1 = F1_all[op_indices, layer_idx, :]
            F2 = F2_all[op_indices, layer_idx, :]
            Z = Z_all[op_indices]

            # Compute interaction score
            interaction_score, _ = compute_interaction_score(F1, F2, Z)

            results.append({
                'layer': layer_idx,
                'op': op,
                'interaction_score': interaction_score,
                'num_examples': len(op_indices)
            })

    return pd.DataFrame(results)


def save_interaction_scores(df, path=INTERACTION_SCORES_PATH):
    """Save interaction scores to CSV."""
    df.to_csv(path, index=False)
    print(f"Interaction scores saved to {path}")


def load_interaction_scores(path=INTERACTION_SCORES_PATH):
    """Load interaction scores from CSV."""
    df = pd.read_csv(path)
    print(f"Interaction scores loaded from {path}")
    return df


if __name__ == "__main__":
    from dataset_generator import load_dataset
    from model_inference import load_hidden_states
    from probe_training import load_probe_results

    set_seed()

    # Load data
    df = load_dataset()
    hidden_states = load_hidden_states()
    probe_results = load_probe_results(hidden_dim=hidden_states['h1_all'].shape[2])

    F1_all = probe_results['F1_all']
    F2_all = probe_results['F2_all']
    Z_all = hidden_states['Z_all']

    # Compute interaction scores for all examples
    interaction_df_all = compute_all_interactions(F1_all, F2_all, Z_all, df)
    save_interaction_scores(interaction_df_all)

    # Compute for correct examples only
    interaction_df_correct = compute_all_interactions(
        F1_all, F2_all, Z_all, df, filter_by={'is_correct': True}
    )
    save_interaction_scores(
        interaction_df_correct,
        INTERACTION_SCORES_PATH.replace('.csv', '_correct.csv')
    )

    # Compute for incorrect examples only
    interaction_df_incorrect = compute_all_interactions(
        F1_all, F2_all, Z_all, df, filter_by={'is_correct': False}
    )
    save_interaction_scores(
        interaction_df_incorrect,
        INTERACTION_SCORES_PATH.replace('.csv', '_incorrect.csv')
    )
