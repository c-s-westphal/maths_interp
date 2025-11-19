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


def load_model_results(results_dir):
    """Load all result files for a given model."""
    results = {}

    # Load interaction scores
    results['interaction_model'] = pd.read_csv(os.path.join(results_dir, 'interaction_scores.csv'))
    results['interaction_gt'] = pd.read_csv(os.path.join(results_dir, 'interaction_scores_gt.csv'))
    results['interaction_baseline'] = pd.read_csv(os.path.join(results_dir, 'interaction_scores_baseline.csv'))
    results['interaction_correct'] = pd.read_csv(os.path.join(results_dir, 'interaction_scores_correct.csv'))
    results['interaction_incorrect'] = pd.read_csv(os.path.join(results_dir, 'interaction_scores_incorrect.csv'))

    return results


def plot_interaction_comparison_fixed(
    interaction_model_output, interaction_gt, interaction_baseline,
    save_path=None
):
    """
    Plot interaction comparison with FIXED dual y-axes:
    - Left axis (blue): Model output interaction (F1, F2 → model output)
    - Right axis (green/red): Ground truth interaction and baseline (F1, F2 → correct answer)
    """
    # Create subplots for each operation
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, op in enumerate(OPERATIONS):
        ax = axes[idx]

        # Filter data for this operation
        model_data = interaction_model_output[interaction_model_output['op'] == op]
        gt_data = interaction_gt[interaction_gt['op'] == op]
        baseline_data = interaction_baseline[interaction_baseline['op'] == op]

        # Create twin axis for ground truth (right y-axis)
        ax2 = ax.twinx()

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

        ax.legend(lines, labels, fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()

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
