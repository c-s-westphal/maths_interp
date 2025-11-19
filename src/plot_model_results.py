"""
Generate plots for any model results.
Usage: python plot_model_results.py --model pythia2.8b
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
from config import OPERATIONS


def compute_operation_specific_probe_r2(results_dir):
    """
    Compute operation-specific probe R² scores by loading probes and evaluating per operation.
    """
    import pickle
    from sklearn.metrics import r2_score
    import torch
    import sys

    # Try to import from src
    try:
        sys.path.insert(0, 'src')
        from probe_training import NumericProbe
        from config import DEVICE
    except:
        return None

    # Load features and dataset
    features_path = os.path.join(results_dir, 'features.npz')
    hidden_states_path = os.path.join(results_dir, 'hidden_states.npz')
    probes_dir = os.path.join(results_dir, 'probes')

    # Try multiple possible dataset locations
    dataset_locations = [
        'data/arithmetic_dataset.pkl',
        'src/data/arithmetic_dataset.pkl',
        os.path.join(os.path.dirname(os.path.dirname(results_dir)), 'data', 'arithmetic_dataset.pkl')
    ]

    dataset_loaded = False
    for loc in dataset_locations:
        if os.path.exists(loc):
            try:
                with open(loc, 'rb') as f:
                    df = pickle.load(f)
                dataset_loaded = True
                break
            except:
                continue

    if not dataset_loaded or not os.path.exists(hidden_states_path):
        print(f"  Warning: Could not load dataset or hidden states for operation-specific R²")
        return None

    try:
        hidden_states = np.load(hidden_states_path, allow_pickle=True)
        h_output_all = hidden_states['h_output_all']  # [num_examples, num_layers, hidden_dim]
    except:
        print(f"  Warning: Could not load h_output_all from hidden_states.npz")
        return None

    correct_answers = df['correct_answer'].values
    ops = df['op'].values

    num_examples, num_layers, hidden_dim = h_output_all.shape

    # Try to load probes
    if not os.path.exists(probes_dir):
        print(f"  Warning: Probes directory not found")
        return None

    # Compute R² for each operation
    op_r2_scores = {}
    for op in OPERATIONS:
        op_mask = ops == op
        op_indices = np.where(op_mask)[0]

        if len(op_indices) == 0:
            continue

        r2_per_layer = []
        for layer_idx in range(num_layers):
            probe_path = os.path.join(probes_dir, f'probe_correct_layer{layer_idx}.pt')

            if not os.path.exists(probe_path):
                r2_per_layer.append(np.nan)
                continue

            try:
                # Load probe
                probe = NumericProbe(hidden_dim)
                probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
                probe.eval()

                # Get predictions for this operation
                h_output_layer = h_output_all[op_indices, layer_idx, :]
                answers_op = correct_answers[op_indices]

                with torch.no_grad():
                    h_tensor = torch.FloatTensor(h_output_layer)
                    predictions, _ = probe(h_tensor)
                    predictions = predictions.numpy().flatten()

                # Compute R²
                r2 = r2_score(answers_op, predictions)
                r2_per_layer.append(r2)
            except Exception as e:
                r2_per_layer.append(np.nan)

        op_r2_scores[op] = np.array(r2_per_layer)

    return op_r2_scores if op_r2_scores else None


def load_model_results(results_dir):
    """Load all result files for a given model."""
    results = {}

    # Load interaction scores
    results['interaction_model'] = pd.read_csv(os.path.join(results_dir, 'interaction_scores.csv'))
    results['interaction_gt'] = pd.read_csv(os.path.join(results_dir, 'interaction_scores_gt.csv'))
    results['interaction_baseline'] = pd.read_csv(os.path.join(results_dir, 'interaction_scores_baseline.csv'))
    results['interaction_correct'] = pd.read_csv(os.path.join(results_dir, 'interaction_scores_correct.csv'))
    results['interaction_incorrect'] = pd.read_csv(os.path.join(results_dir, 'interaction_scores_incorrect.csv'))

    # Load operation-specific probe R² scores from CSV (if available)
    probe_metrics_path = os.path.join(results_dir, 'probe_metrics_by_operation.csv')
    if os.path.exists(probe_metrics_path):
        print("\nLoading operation-specific probe metrics from CSV...")
        probe_metrics_df = pd.read_csv(probe_metrics_path)

        # Convert to dict: op -> [r2_layer0, r2_layer1, ...]
        op_r2_scores = {}
        for op in OPERATIONS:
            op_data = probe_metrics_df[probe_metrics_df['operations'] == op].sort_values('layer')
            if len(op_data) > 0:
                op_r2_scores[op] = op_data['r2'].values

        results['op_probe_r2'] = op_r2_scores
    else:
        # Fall back to computing from hidden states
        print("\nComputing operation-specific probe R² scores...")
        op_r2_scores = compute_operation_specific_probe_r2(results_dir)
        results['op_probe_r2'] = op_r2_scores

    return results


def plot_interaction_comparison_fixed(
    interaction_model_output, interaction_gt, interaction_baseline,
    op_probe_r2=None,
    save_path=None
):
    """
    Plot interaction comparison with FIXED dual y-axes:
    - Left axis (blue): Model output interaction (F1, F2 → model output)
    - Right axis (green/red): Ground truth interaction and baseline (F1, F2 → correct answer)
    - Purple line: Correct answer probe R² score (operation-specific, if available)
    """
    # Create subplots for each operation (wider to accommodate 3 y-axes)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    # Compute max R² across all operations for consistent scaling
    max_r2 = 0.2  # Default
    if op_probe_r2 is not None:
        all_r2_values = []
        for op in OPERATIONS:
            if op in op_probe_r2 and op_probe_r2[op] is not None:
                valid_r2 = op_probe_r2[op][~np.isnan(op_probe_r2[op])]
                if len(valid_r2) > 0:
                    all_r2_values.extend(valid_r2)
        if all_r2_values:
            max_r2 = max(0.2, np.max(all_r2_values) * 1.1)  # Add 10% headroom

    for idx, op in enumerate(OPERATIONS):
        ax = axes[idx]

        # Filter data for this operation
        model_data = interaction_model_output[interaction_model_output['op'] == op]
        gt_data = interaction_gt[interaction_gt['op'] == op]
        baseline_data = interaction_baseline[interaction_baseline['op'] == op]

        # Create twin axes for different scales
        ax2 = ax.twinx()  # Right y-axis for ground truth interaction

        # Plot model output interaction on LEFT y-axis (blue)
        if len(model_data) > 0:
            line1 = ax.plot(model_data['layer'], model_data['interaction_score'],
                   marker='o', label='F1, F2 → model output', linewidth=2, color='blue')
            ax.set_ylabel('Model Output Interaction', fontsize=9, color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

        # Plot ground truth interaction on RIGHT y-axis (green)
        if len(gt_data) > 0:
            line2 = ax2.plot(gt_data['layer'], gt_data['interaction_score'],
                   marker='s', label='F1, F2 → correct answer', linewidth=2, color='green')
            ax2.set_ylabel('Correct Answer Interaction', fontsize=9, color='green')
            ax2.tick_params(axis='y', labelcolor='green')

        # Plot baseline on RIGHT y-axis (red dashed)
        if len(baseline_data) > 0:
            baseline_score = baseline_data['interaction_score'].values[0]
            if len(model_data) > 0 or len(gt_data) > 0:
                line3 = ax2.axhline(y=baseline_score, color='red', linestyle='--',
                          linewidth=2, label='x1, x2 → correct (baseline)')

        # Plot OPERATION-SPECIFIC correct answer probe R² on THIRD y-axis (purple)
        line4 = None
        if op_probe_r2 is not None and op in op_probe_r2 and op_probe_r2[op] is not None:
            # Create a third y-axis for probe R² (offset to the right)
            ax3 = ax.twinx()
            # Offset the third axis to the right
            ax3.spines['right'].set_position(('axes', 1.15))

            r2_correct = op_probe_r2[op]
            layers = np.arange(len(r2_correct))
            line4 = ax3.plot(layers, r2_correct,
                   marker='^', label=f'Probe R² ({op})', linewidth=2,
                   color='purple', linestyle=':', alpha=0.8)
            ax3.set_ylabel('Probe R²', fontsize=9, color='purple')
            ax3.tick_params(axis='y', labelcolor='purple')
            ax3.set_ylim([0, max_r2])  # Dynamically scaled based on actual max

        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_title(f'{op.upper()}', fontsize=12, fontweight='bold')

        # Combine legends from both axes
        lines = []
        labels = []
        if len(model_data) > 0:
            lines.extend(line1)
            labels.append('F1, F2 → model output')
        if len(gt_data) > 0:
            lines.extend(line2)
            labels.append('F1, F2 → correct answer')
        if len(baseline_data) > 0:
            lines.append(line3)
            labels.append('x1, x2 → correct (baseline)')
        if line4 is not None:
            lines.extend(line4)
            labels.append('Correct answer probe R²')

        ax.legend(lines, labels, fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    # Adjust layout to make room for the third y-axis
    plt.subplots_adjust(right=0.85, wspace=0.4)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_interaction_correct_vs_incorrect_by_operation(
    interaction_correct, interaction_incorrect,
    save_path=None
):
    """
    Plot interaction scores for correct vs incorrect examples for ALL operations.
    Creates a 2x3 subplot grid showing each operation separately.
    Only plots lines for operations that have data.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, op in enumerate(OPERATIONS):
        ax = axes[idx]

        # Filter by operation
        correct_data = interaction_correct[interaction_correct['op'] == op]
        incorrect_data = interaction_incorrect[interaction_incorrect['op'] == op]

        has_data = False
        if len(correct_data) > 0:
            ax.plot(correct_data['layer'], correct_data['interaction_score'],
                    marker='o', label='Correct', linewidth=2, color='green')
            has_data = True

        if len(incorrect_data) > 0:
            ax.plot(incorrect_data['layer'], incorrect_data['interaction_score'],
                    marker='s', label='Incorrect', linewidth=2, color='red')
            has_data = True

        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('Interaction Score', fontsize=10)
        ax.set_title(f'{op.upper()}', fontsize=12, fontweight='bold')

        if has_data:
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')

        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle('Interaction Score: Correct vs Incorrect (All Operations)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Generate plots for a specific model."""
    parser = argparse.ArgumentParser(description='Generate plots for model results')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., pythia2.8b, phi2)')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Custom results directory (default: results_{model}/results)')
    parser.add_argument('--plots-dir', type=str, default=None,
                       help='Custom plots directory (default: plots_{model})')

    args = parser.parse_args()

    # Set up directories
    if args.results_dir is None:
        results_dir = f'results_{args.model}/results'
    else:
        results_dir = args.results_dir

    if args.plots_dir is None:
        plots_dir = f'plots_{args.model}/plots'
    else:
        plots_dir = args.plots_dir

    print(f"Loading {args.model} results from {results_dir}...")
    results = load_model_results(results_dir)

    # Print data summary
    print("\nData summary:")
    print(f"  Operations with correct data: {results['interaction_correct']['op'].unique()}")
    print(f"  Operations with incorrect data: {results['interaction_incorrect']['op'].unique()}")

    # Create output directory
    os.makedirs(plots_dir, exist_ok=True)

    print("\nGenerating fixed interaction comparison plot...")
    plot_interaction_comparison_fixed(
        results['interaction_model'],
        results['interaction_gt'],
        results['interaction_baseline'],
        op_probe_r2=results.get('op_probe_r2'),
        save_path=os.path.join(plots_dir, 'interaction_comparison_fixed.png')
    )

    print("\nGenerating correct vs incorrect by operation plot...")
    plot_interaction_correct_vs_incorrect_by_operation(
        results['interaction_correct'],
        results['interaction_incorrect'],
        save_path=os.path.join(plots_dir, 'interaction_correct_vs_incorrect_all_ops.png')
    )

    print("\nAll plots generated successfully!")
    print(f"Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
