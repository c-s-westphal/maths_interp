"""
Compute operation-specific probe metrics and save them.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import torch
from probe_training import NumericProbe
from config import OPERATIONS, DEVICE
import os


def compute_and_save_op_specific_metrics(
    h_output_all, correct_answers, ops, probes_correct, results_dir
):
    """
    Compute operation-specific probe R² and MAE scores.

    Args:
        h_output_all: [num_examples, num_layers, hidden_dim]
        correct_answers: [num_examples]
        ops: [num_examples] - operation labels
        probes_correct: List of trained probes (one per layer)
        results_dir: Directory to save results
    """
    num_examples, num_layers, hidden_dim = h_output_all.shape

    # Initialize storage
    results = {
        'operations': [],
        'layer': [],
        'r2': [],
        'mae': [],
        'num_examples': []
    }

    print("\nComputing operation-specific probe metrics...")

    for op in OPERATIONS:
        op_mask = ops == op
        op_indices = np.where(op_mask)[0]

        if len(op_indices) == 0:
            continue

        print(f"  {op}: {len(op_indices)} examples")

        for layer_idx in range(num_layers):
            # Get hidden states and targets for this operation
            h_output_layer = h_output_all[op_indices, layer_idx, :]
            targets = correct_answers[op_indices]

            # Get predictions from probe
            probe = probes_correct[layer_idx]
            probe.eval()

            with torch.no_grad():
                h_tensor = torch.FloatTensor(h_output_layer).to(DEVICE)
                predictions, _ = probe(h_tensor)
                predictions = predictions.cpu().numpy().flatten()

            # Compute metrics
            r2 = r2_score(targets, predictions)
            mae = mean_absolute_error(targets, predictions)

            results['operations'].append(op)
            results['layer'].append(layer_idx)
            results['r2'].append(r2)
            results['mae'].append(mae)
            results['num_examples'].append(len(op_indices))

    # Save to CSV
    df_results = pd.DataFrame(results)
    save_path = os.path.join(results_dir, 'probe_metrics_by_operation.csv')
    df_results.to_csv(save_path, index=False)
    print(f"\nOperation-specific probe metrics saved to {save_path}")

    return df_results


if __name__ == "__main__":
    from dataset_generator import load_dataset
    from model_inference import load_hidden_states
    from probe_training import load_probe_results
    from config import set_seed, RESULTS_DIR

    set_seed()

    # Load data
    print("Loading data...")
    df = load_dataset()
    hidden_states = load_hidden_states()
    probe_results = load_probe_results(hidden_dim=hidden_states['h1_all'].shape[2])

    h_output_all = hidden_states['h_output_all']
    correct_answers = df['correct_answer'].values
    ops = df['op'].values
    probes_correct = probe_results['probes_correct']

    # Compute and save
    compute_and_save_op_specific_metrics(
        h_output_all, correct_answers, ops, probes_correct, RESULTS_DIR
    )
