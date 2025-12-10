"""
MI-based synergy analysis using KSG estimator.

Computes synergy as:
    Synergy = I(F1, F2; Z) - I(F1; Z) - I(F2; Z)

Where I(X; Y) is mutual information estimated via KSG.

Three feature representations are supported:
1. Scalar probe outputs (x̂₁, x̂₂) - 1D each
2. Log-probabilities of correct answer tokens - 1-5D
3. PCA-reduced probe features - configurable dimensionality
"""
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn

from config import DEVICE, OPERATIONS, RESULTS_DIR, RANDOM_SEED, set_seed
import os


# =============================================================================
# KSG Mutual Information Estimator
# =============================================================================

def ksg_mi(x, y, k=3):
    """
    Estimate mutual information I(X; Y) using the KSG estimator (Kraskov et al., 2004).

    Uses the first algorithm from the paper (estimator 1).

    Args:
        x: Array of shape [n_samples, dim_x] or [n_samples] for 1D
        y: Array of shape [n_samples, dim_y] or [n_samples] for 1D
        k: Number of nearest neighbors (default: 3)

    Returns:
        Estimated mutual information in nats
    """
    # Ensure 2D
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    y = np.atleast_2d(y.T).T if y.ndim == 1 else y

    n_samples = len(x)
    assert len(y) == n_samples, "x and y must have same number of samples"

    # Add tiny noise to break ties (important for discrete-ish data)
    x = x + np.random.randn(*x.shape) * 1e-10
    y = y + np.random.randn(*y.shape) * 1e-10

    # Joint space
    xy = np.hstack([x, y])

    # Build KD-trees
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # For each point, find distance to k-th neighbor in joint space
    # query returns (distances, indices), we want k+1 because point itself is included
    dists_xy, _ = tree_xy.query(xy, k=k+1, p=np.inf)  # Chebyshev (max) norm
    epsilon = dists_xy[:, -1]  # Distance to k-th neighbor (excluding self)

    # Count neighbors within epsilon in marginal spaces
    # We use epsilon as the radius and subtract 1 for the point itself
    n_x = np.array([len(tree_x.query_ball_point(x[i], epsilon[i], p=np.inf)) - 1
                    for i in range(n_samples)])
    n_y = np.array([len(tree_y.query_ball_point(y[i], epsilon[i], p=np.inf)) - 1
                    for i in range(n_samples)])

    # Handle edge cases where n_x or n_y is 0
    n_x = np.maximum(n_x, 1)
    n_y = np.maximum(n_y, 1)

    # KSG estimator 1
    mi = digamma(k) + digamma(n_samples) - np.mean(digamma(n_x) + digamma(n_y))

    return max(0, mi)  # MI is non-negative


def ksg_cmi(x, y, z, k=3):
    """
    Estimate conditional mutual information I(X; Y | Z) using KSG.

    Uses the chain rule: I(X; Y | Z) = I(X; Y, Z) - I(X; Z)

    Args:
        x: Array of shape [n_samples, dim_x]
        y: Array of shape [n_samples, dim_y]
        z: Array of shape [n_samples, dim_z]
        k: Number of nearest neighbors

    Returns:
        Estimated conditional MI in nats
    """
    # Ensure 2D
    y = np.atleast_2d(y.T).T if y.ndim == 1 else y
    z = np.atleast_2d(z.T).T if z.ndim == 1 else z

    yz = np.hstack([y, z])

    mi_x_yz = ksg_mi(x, yz, k=k)
    mi_x_z = ksg_mi(x, z, k=k)

    return max(0, mi_x_yz - mi_x_z)


def compute_synergy_mi(f1, f2, z, k=3):
    """
    Compute synergy using mutual information.

    Synergy = I(F1, F2; Z) - I(F1; Z) - I(F2; Z)

    Interpretation:
    - Positive synergy: F1 and F2 together provide more information than sum of parts
    - Zero synergy: F1 and F2 contribute independently
    - Negative synergy: Redundancy between F1 and F2

    Args:
        f1: Features for operand 1, shape [n_samples, dim_f1] or [n_samples]
        f2: Features for operand 2, shape [n_samples, dim_f2] or [n_samples]
        z: Target values, shape [n_samples] or [n_samples, dim_z]
        k: Number of neighbors for KSG

    Returns:
        synergy: float
        components: dict with I(F1,F2;Z), I(F1;Z), I(F2;Z)
    """
    # Ensure 2D
    f1 = np.atleast_2d(f1.T).T if f1.ndim == 1 else f1
    f2 = np.atleast_2d(f2.T).T if f2.ndim == 1 else f2
    z = np.atleast_2d(z.T).T if z.ndim == 1 else z

    # Joint features
    f12 = np.hstack([f1, f2])

    # Compute MI terms
    i_f12_z = ksg_mi(f12, z, k=k)
    i_f1_z = ksg_mi(f1, z, k=k)
    i_f2_z = ksg_mi(f2, z, k=k)

    synergy = i_f12_z - i_f1_z - i_f2_z

    components = {
        'I_F12_Z': i_f12_z,
        'I_F1_Z': i_f1_z,
        'I_F2_Z': i_f2_z,
        'synergy': synergy
    }

    return synergy, components


# =============================================================================
# Feature Extraction Methods
# =============================================================================

def get_scalar_probe_outputs(probe, hidden_states):
    """
    Get scalar probe predictions from hidden states.

    Args:
        probe: Trained NumericProbe
        hidden_states: [n_samples, hidden_dim]

    Returns:
        predictions: [n_samples] scalar predictions
    """
    probe.eval()
    with torch.no_grad():
        h_tensor = torch.FloatTensor(hidden_states).to(DEVICE)
        predictions, _ = probe(h_tensor)
        return predictions.cpu().numpy().flatten()


def get_log_probs_for_answer(model, tokenizer, input_ids_list, correct_answers, max_digits=5):
    """
    Get log-probabilities for correct answer tokens.

    For each example, we get the log-prob of each digit in the correct answer,
    producing a representation of dimension = number of digits (up to max_digits).

    Args:
        model: Language model
        tokenizer: Tokenizer
        input_ids_list: List of input_ids for each example
        correct_answers: Array of correct integer answers
        max_digits: Maximum number of digits to consider

    Returns:
        log_probs: [n_samples, max_digits] - padded with -inf for missing digits
    """
    model.eval()
    n_samples = len(input_ids_list)
    log_probs = np.full((n_samples, max_digits), -np.inf, dtype=np.float32)

    with torch.no_grad():
        for i, (input_ids, answer) in enumerate(tqdm(
            zip(input_ids_list, correct_answers),
            total=n_samples,
            desc="Extracting log-probs"
        )):
            # Get model output
            input_tensor = torch.tensor([input_ids], device=DEVICE)
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]  # Logits at last position
            log_softmax = torch.log_softmax(logits, dim=-1)

            # Convert answer to string and get digit tokens
            answer_str = str(int(answer))

            # Handle negative numbers
            if answer < 0:
                answer_str = answer_str[1:]  # Remove minus sign for digits

            for j, digit_char in enumerate(answer_str[:max_digits]):
                # Tokenize the digit (try with and without space)
                digit_tokens = tokenizer.encode(digit_char, add_special_tokens=False)
                if len(digit_tokens) == 1:
                    token_id = digit_tokens[0]
                    log_probs[i, j] = log_softmax[token_id].item()

    return log_probs


def get_first_token_log_prob(model, tokenizer, input_ids_list, correct_answers):
    """
    Get log-probability of just the first token of the correct answer.

    This is a simpler 1D representation.

    Args:
        model: Language model
        tokenizer: Tokenizer
        input_ids_list: List of input_ids
        correct_answers: Array of correct answers

    Returns:
        log_probs: [n_samples] - log-prob of first digit/token
    """
    model.eval()
    n_samples = len(input_ids_list)
    log_probs = np.zeros(n_samples, dtype=np.float32)

    with torch.no_grad():
        for i, (input_ids, answer) in enumerate(tqdm(
            zip(input_ids_list, correct_answers),
            total=n_samples,
            desc="Extracting first-token log-probs"
        )):
            input_tensor = torch.tensor([input_ids], device=DEVICE)
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]
            log_softmax = torch.log_softmax(logits, dim=-1)

            # Get first character of answer
            answer_str = str(int(answer))
            if answer < 0:
                first_char = '-'
            else:
                first_char = answer_str[0]

            digit_tokens = tokenizer.encode(first_char, add_special_tokens=False)
            if len(digit_tokens) >= 1:
                token_id = digit_tokens[0]
                log_probs[i] = log_softmax[token_id].item()

    return log_probs


def apply_pca_to_features(F_all, n_components=3):
    """
    Apply PCA to reduce feature dimensionality.

    Args:
        F_all: [n_samples, n_layers, feature_dim] or [n_samples, feature_dim]
        n_components: Number of PCA components

    Returns:
        F_pca: Reduced features with same shape except last dim is n_components
        pca_models: List of fitted PCA models (one per layer if 3D input)
    """
    if F_all.ndim == 2:
        pca = PCA(n_components=n_components)
        F_pca = pca.fit_transform(F_all)
        return F_pca, [pca]

    # 3D case: apply PCA per layer
    n_samples, n_layers, feature_dim = F_all.shape
    F_pca = np.zeros((n_samples, n_layers, n_components), dtype=np.float32)
    pca_models = []

    for layer_idx in range(n_layers):
        pca = PCA(n_components=n_components)
        F_pca[:, layer_idx, :] = pca.fit_transform(F_all[:, layer_idx, :])
        pca_models.append(pca)

    return F_pca, pca_models


# =============================================================================
# Main Synergy Computation Functions
# =============================================================================

def compute_synergy_scalar_probes(probes_op1, probes_op2, h1_all, h2_all, Z_all, df, k=3):
    """
    Compute MI-based synergy using scalar probe outputs.

    F1 = x̂₁ (scalar prediction of operand 1)
    F2 = x̂₂ (scalar prediction of operand 2)

    Args:
        probes_op1: List of probes for operand 1 (one per layer)
        probes_op2: List of probes for operand 2 (one per layer)
        h1_all: [n_samples, n_layers, hidden_dim]
        h2_all: [n_samples, n_layers, hidden_dim]
        Z_all: [n_samples] target values
        df: DataFrame with metadata
        k: KSG neighbors

    Returns:
        DataFrame with synergy scores per layer and operation
    """
    n_samples, n_layers, _ = h1_all.shape
    results = []

    print("Computing MI synergy using scalar probe outputs...")

    for op in tqdm(OPERATIONS, desc="Operations"):
        op_mask = df['op'] == op
        op_indices = np.where(op_mask)[0]

        if len(op_indices) < 100:
            print(f"Skipping {op} (only {len(op_indices)} examples)")
            continue

        Z_op = Z_all[op_indices]

        for layer_idx in range(n_layers):
            # Get scalar probe outputs for this layer
            h1_layer = h1_all[op_indices, layer_idx, :]
            h2_layer = h2_all[op_indices, layer_idx, :]

            x1_hat = get_scalar_probe_outputs(probes_op1[layer_idx], h1_layer)
            x2_hat = get_scalar_probe_outputs(probes_op2[layer_idx], h2_layer)

            # Compute synergy
            synergy, components = compute_synergy_mi(x1_hat, x2_hat, Z_op, k=k)

            results.append({
                'layer': layer_idx,
                'op': op,
                'synergy': synergy,
                'I_F12_Z': components['I_F12_Z'],
                'I_F1_Z': components['I_F1_Z'],
                'I_F2_Z': components['I_F2_Z'],
                'method': 'scalar_probe',
                'n_samples': len(op_indices)
            })

    return pd.DataFrame(results)


def compute_synergy_pca_features(F1_all, F2_all, Z_all, df, n_components=3, k=3):
    """
    Compute MI-based synergy using PCA-reduced features.

    Args:
        F1_all: [n_samples, n_layers, feature_dim]
        F2_all: [n_samples, n_layers, feature_dim]
        Z_all: [n_samples]
        df: DataFrame with metadata
        n_components: PCA dimensionality
        k: KSG neighbors

    Returns:
        DataFrame with synergy scores
    """
    n_samples, n_layers, feature_dim = F1_all.shape
    results = []

    print(f"Computing MI synergy using PCA features (n_components={n_components})...")

    # Apply PCA per layer
    F1_pca, _ = apply_pca_to_features(F1_all, n_components=n_components)
    F2_pca, _ = apply_pca_to_features(F2_all, n_components=n_components)

    for op in tqdm(OPERATIONS, desc="Operations"):
        op_mask = df['op'] == op
        op_indices = np.where(op_mask)[0]

        if len(op_indices) < 100:
            print(f"Skipping {op} (only {len(op_indices)} examples)")
            continue

        Z_op = Z_all[op_indices]

        for layer_idx in range(n_layers):
            f1 = F1_pca[op_indices, layer_idx, :]
            f2 = F2_pca[op_indices, layer_idx, :]

            synergy, components = compute_synergy_mi(f1, f2, Z_op, k=k)

            results.append({
                'layer': layer_idx,
                'op': op,
                'synergy': synergy,
                'I_F12_Z': components['I_F12_Z'],
                'I_F1_Z': components['I_F1_Z'],
                'I_F2_Z': components['I_F2_Z'],
                'method': f'pca_{n_components}',
                'n_samples': len(op_indices)
            })

    return pd.DataFrame(results)


def compute_synergy_log_probs(model, tokenizer, df, Z_all, k=3):
    """
    Compute MI-based synergy using log-probabilities of correct answer tokens.

    Note: This requires re-running inference, so it's slower.
    Also note: This uses the CORRECT answer's log-prob, which is a measure
    of how confident the model is about the right answer.

    For operand features, we use the log-prob of the first digit of each operand.

    Args:
        model: Language model
        tokenizer: Tokenizer
        df: DataFrame with input_ids and operand values
        Z_all: [n_samples] target values (model output or correct answer)
        k: KSG neighbors

    Returns:
        DataFrame with synergy scores (single row per operation, no layers)
    """
    results = []

    print("Computing MI synergy using log-probabilities...")

    # Get log-probs for the correct answer (first token)
    input_ids_list = df['input_ids'].tolist()
    correct_answers = df['correct_answer'].values

    # For operand features, we use the actual operand values (1D)
    # since log-probs of operands aren't directly available from the model
    x1_values = df['x1'].values.astype(np.float32)
    x2_values = df['x2'].values.astype(np.float32)

    # Get log-prob of first token of correct answer
    answer_log_probs = get_first_token_log_prob(model, tokenizer, input_ids_list, correct_answers)

    for op in tqdm(OPERATIONS, desc="Operations"):
        op_mask = df['op'] == op
        op_indices = np.where(op_mask)[0]

        if len(op_indices) < 100:
            continue

        # For this method, we measure synergy between operand VALUES
        # and the log-prob of the correct answer
        f1 = x1_values[op_indices]
        f2 = x2_values[op_indices]
        z = answer_log_probs[op_indices]

        # Also compute synergy with Z_all (model output logit)
        z_model = Z_all[op_indices]

        synergy_logprob, comp_logprob = compute_synergy_mi(f1, f2, z, k=k)
        synergy_model, comp_model = compute_synergy_mi(f1, f2, z_model, k=k)

        results.append({
            'op': op,
            'synergy_to_logprob': synergy_logprob,
            'synergy_to_model_output': synergy_model,
            'I_F12_Z_logprob': comp_logprob['I_F12_Z'],
            'I_F1_Z_logprob': comp_logprob['I_F1_Z'],
            'I_F2_Z_logprob': comp_logprob['I_F2_Z'],
            'method': 'log_prob',
            'n_samples': len(op_indices)
        })

    return pd.DataFrame(results)


def compute_synergy_raw_operands(x1_all, x2_all, Z_all, df, k=3):
    """
    Compute baseline MI synergy using raw operand values.

    This serves as a baseline: how much synergy exists between the
    raw numbers x1, x2 and the target Z?

    Args:
        x1_all: [n_samples] raw operand 1 values
        x2_all: [n_samples] raw operand 2 values
        Z_all: [n_samples] target values
        df: DataFrame with metadata
        k: KSG neighbors

    Returns:
        DataFrame with synergy scores (one per operation)
    """
    results = []

    print("Computing baseline MI synergy (raw operands)...")

    for op in tqdm(OPERATIONS, desc="Operations"):
        op_mask = df['op'] == op
        op_indices = np.where(op_mask)[0]

        if len(op_indices) < 100:
            continue

        f1 = x1_all[op_indices].astype(np.float32)
        f2 = x2_all[op_indices].astype(np.float32)
        z = Z_all[op_indices]

        synergy, components = compute_synergy_mi(f1, f2, z, k=k)

        results.append({
            'op': op,
            'synergy': synergy,
            'I_F12_Z': components['I_F12_Z'],
            'I_F1_Z': components['I_F1_Z'],
            'I_F2_Z': components['I_F2_Z'],
            'method': 'raw_operands',
            'n_samples': len(op_indices)
        })

    return pd.DataFrame(results)


# =============================================================================
# Orchestration
# =============================================================================

def compute_all_mi_synergy(
    probes_op1, probes_op2,
    h1_all, h2_all,
    F1_all, F2_all,
    Z_all, df,
    model=None, tokenizer=None,
    pca_components=[3, 5],
    k=3
):
    """
    Compute MI-based synergy using all methods.

    Args:
        probes_op1, probes_op2: Lists of trained probes
        h1_all, h2_all: Hidden states [n_samples, n_layers, hidden_dim]
        F1_all, F2_all: Probe features [n_samples, n_layers, feature_dim]
        Z_all: Target values [n_samples]
        df: DataFrame with metadata
        model, tokenizer: For log-prob method (optional)
        pca_components: List of PCA dimensions to try
        k: KSG neighbors

    Returns:
        Dictionary of DataFrames with results for each method
    """
    results = {}

    # Method 1: Scalar probe outputs
    results['scalar_probe'] = compute_synergy_scalar_probes(
        probes_op1, probes_op2, h1_all, h2_all, Z_all, df, k=k
    )

    # Method 2: PCA-reduced features
    for n_comp in pca_components:
        key = f'pca_{n_comp}'
        results[key] = compute_synergy_pca_features(
            F1_all, F2_all, Z_all, df, n_components=n_comp, k=k
        )

    # Method 3: Raw operands baseline
    results['raw_baseline'] = compute_synergy_raw_operands(
        df['x1'].values, df['x2'].values, Z_all, df, k=k
    )

    # Method 4: Log-probs (if model provided)
    if model is not None and tokenizer is not None:
        results['log_prob'] = compute_synergy_log_probs(
            model, tokenizer, df, Z_all, k=k
        )

    return results


def save_mi_synergy_results(results_dict, output_dir=RESULTS_DIR):
    """Save MI synergy results to CSV files."""
    for method, df in results_dict.items():
        path = os.path.join(output_dir, f'mi_synergy_{method}.csv')
        df.to_csv(path, index=False)
        print(f"Saved {method} results to {path}")


def load_mi_synergy_results(output_dir=RESULTS_DIR):
    """Load MI synergy results from CSV files."""
    import glob
    results = {}
    pattern = os.path.join(output_dir, 'mi_synergy_*.csv')
    for path in glob.glob(pattern):
        method = os.path.basename(path).replace('mi_synergy_', '').replace('.csv', '')
        results[method] = pd.read_csv(path)
        print(f"Loaded {method} from {path}")
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_mi_synergy_by_layer(synergy_df, save_path=None, title="MI Synergy by Layer"):
    """
    Plot MI synergy scores vs layer for each operation.

    Args:
        synergy_df: DataFrame with columns [layer, op, synergy]
        save_path: Path to save figure
        title: Plot title
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for op in OPERATIONS:
        op_data = synergy_df[synergy_df['op'] == op]
        if len(op_data) > 0:
            plt.plot(op_data['layer'], op_data['synergy'],
                    marker='o', label=op, linewidth=2)

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('MI Synergy (nats)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_mi_components_by_layer(synergy_df, operation='mul', save_path=None):
    """
    Plot MI components (I(F1,F2;Z), I(F1;Z), I(F2;Z)) for a specific operation.

    Args:
        synergy_df: DataFrame with columns [layer, op, I_F12_Z, I_F1_Z, I_F2_Z, synergy]
        operation: Which operation to plot
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    op_data = synergy_df[synergy_df['op'] == operation]
    if len(op_data) == 0:
        print(f"No data for operation {operation}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot MI components
    ax1.plot(op_data['layer'], op_data['I_F12_Z'], marker='o', label='I(F1,F2; Z)', linewidth=2)
    ax1.plot(op_data['layer'], op_data['I_F1_Z'], marker='s', label='I(F1; Z)', linewidth=2)
    ax1.plot(op_data['layer'], op_data['I_F2_Z'], marker='^', label='I(F2; Z)', linewidth=2)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Mutual Information (nats)', fontsize=12)
    ax1.set_title(f'MI Components: {operation.upper()}', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot synergy
    ax2.plot(op_data['layer'], op_data['synergy'], marker='o', linewidth=2, color='purple')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Synergy (nats)', fontsize=12)
    ax2.set_title(f'Synergy = I(F1,F2;Z) - I(F1;Z) - I(F2;Z): {operation.upper()}', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_mi_synergy_comparison(results_dict, save_path=None):
    """
    Compare MI synergy across different methods (scalar probe, PCA, etc.).

    Args:
        results_dict: Dictionary of DataFrames from compute_all_mi_synergy
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    # Filter to methods that have layer information
    layered_methods = {k: v for k, v in results_dict.items()
                       if 'layer' in v.columns}

    if not layered_methods:
        print("No layered methods to compare")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, op in enumerate(OPERATIONS):
        if idx >= len(axes):
            break

        ax = axes[idx]

        for method_name, df in layered_methods.items():
            op_data = df[df['op'] == op]
            if len(op_data) > 0:
                ax.plot(op_data['layer'], op_data['synergy'],
                       marker='o', label=method_name, linewidth=2, alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('MI Synergy (nats)', fontsize=10)
        ax.set_title(f'{op.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Remove unused subplots
    for idx in range(len(OPERATIONS), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('MI Synergy Comparison Across Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_mi_synergy_heatmap(synergy_df, save_path=None, title="MI Synergy Heatmap"):
    """
    Plot heatmap of MI synergy: operations vs layers.

    Args:
        synergy_df: DataFrame with columns [layer, op, synergy]
        save_path: Path to save figure
        title: Plot title
    """
    import matplotlib.pyplot as plt

    # Pivot data
    pivot_data = synergy_df.pivot(index='op', columns='layer', values='synergy')

    plt.figure(figsize=(14, 6))

    # Use diverging colormap centered at 0
    vmax = max(abs(pivot_data.values.min()), abs(pivot_data.values.max()))
    im = plt.imshow(pivot_data.values, aspect='auto', cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax, interpolation='nearest')

    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
    plt.yticks(range(len(pivot_data.index)), pivot_data.index)

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Operation', fontsize=12)
    plt.title(title, fontsize=14)

    cbar = plt.colorbar(im)
    cbar.set_label('MI Synergy (nats)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_baseline_synergy_comparison(results_dict, save_path=None):
    """
    Compare synergy across operations for baseline (raw operands) method.

    Args:
        results_dict: Dictionary containing 'raw_baseline' DataFrame
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    if 'raw_baseline' not in results_dict:
        print("No baseline results found")
        return

    baseline_df = results_dict['raw_baseline']

    plt.figure(figsize=(10, 6))

    ops = baseline_df['op'].values
    synergies = baseline_df['synergy'].values

    colors = plt.cm.Set2(np.linspace(0, 1, len(ops)))
    bars = plt.bar(ops, synergies, color=colors)

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Operation', fontsize=12)
    plt.ylabel('MI Synergy (nats)', fontsize=12)
    plt.title('Baseline MI Synergy: Raw Operands (x1, x2) → Z', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, syn in zip(bars, synergies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{syn:.3f}', ha='center',
                va='bottom' if height >= 0 else 'top', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def generate_mi_synergy_plots(results_dict=None, plots_dir=None):
    """
    Generate all MI synergy plots.

    Args:
        results_dict: Dictionary of results (if None, loads from disk)
        plots_dir: Directory to save plots (if None, uses config)
    """
    from config import PLOTS_DIR

    if plots_dir is None:
        plots_dir = PLOTS_DIR

    if results_dict is None:
        results_dict = load_mi_synergy_results()

    print("Generating MI synergy plots...")

    # 1. Synergy by layer for scalar probe method
    if 'scalar_probe' in results_dict:
        plot_mi_synergy_by_layer(
            results_dict['scalar_probe'],
            save_path=os.path.join(plots_dir, 'mi_synergy_scalar_probe.png'),
            title='MI Synergy by Layer (Scalar Probe Outputs)'
        )

        # Components for multiplication
        plot_mi_components_by_layer(
            results_dict['scalar_probe'],
            operation='mul',
            save_path=os.path.join(plots_dir, 'mi_components_mul.png')
        )

        # Heatmap
        plot_mi_synergy_heatmap(
            results_dict['scalar_probe'],
            save_path=os.path.join(plots_dir, 'mi_synergy_heatmap.png'),
            title='MI Synergy Heatmap (Scalar Probe)'
        )

    # 2. Comparison across methods
    plot_mi_synergy_comparison(
        results_dict,
        save_path=os.path.join(plots_dir, 'mi_synergy_method_comparison.png')
    )

    # 3. Baseline comparison
    plot_baseline_synergy_comparison(
        results_dict,
        save_path=os.path.join(plots_dir, 'mi_synergy_baseline.png')
    )

    # 4. PCA methods
    for key in results_dict:
        if key.startswith('pca_'):
            plot_mi_synergy_by_layer(
                results_dict[key],
                save_path=os.path.join(plots_dir, f'mi_synergy_{key}.png'),
                title=f'MI Synergy by Layer ({key.upper()} Features)'
            )

    print("MI synergy plots generated!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from dataset_generator import load_dataset
    from model_inference import load_hidden_states, load_model_and_tokenizer
    from probe_training import load_probe_results

    set_seed()

    print("Loading data...")
    df = load_dataset()
    hidden_states = load_hidden_states()
    probe_results = load_probe_results(hidden_dim=hidden_states['h1_all'].shape[2])

    h1_all = hidden_states['h1_all']
    h2_all = hidden_states['h2_all']
    Z_all = hidden_states['Z_all']

    F1_all = probe_results['F1_all']
    F2_all = probe_results['F2_all']
    probes_op1 = probe_results['probes_op1']
    probes_op2 = probe_results['probes_op2']

    # Optionally load model for log-prob method
    # model, tokenizer = load_model_and_tokenizer()
    model, tokenizer = None, None  # Skip log-prob method for now

    print("\nComputing MI-based synergy...")
    results = compute_all_mi_synergy(
        probes_op1, probes_op2,
        h1_all, h2_all,
        F1_all, F2_all,
        Z_all, df,
        model=model, tokenizer=tokenizer,
        pca_components=[3, 5],
        k=3
    )

    # Save results
    save_mi_synergy_results(results)

    # Print summary
    print("\n" + "="*60)
    print("MI Synergy Summary")
    print("="*60)

    for method, result_df in results.items():
        print(f"\n{method}:")
        if 'layer' in result_df.columns:
            # Average across layers
            summary = result_df.groupby('op')['synergy'].mean()
            for op, syn in summary.items():
                print(f"  {op}: {syn:.4f}")
        else:
            for _, row in result_df.iterrows():
                syn_col = 'synergy' if 'synergy' in row else 'synergy_to_model_output'
                print(f"  {row['op']}: {row[syn_col]:.4f}")
