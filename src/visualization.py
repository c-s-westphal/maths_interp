"""
Generate visualizations and analysis plots.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from config import PLOTS_DIR, OPERATIONS


def plot_interaction_by_layer(interaction_df, save_path=None, title="Interaction Score by Layer"):
    """
    Plot interaction scores vs layer for each operation.

    Args:
        interaction_df: DataFrame with columns [layer, op, interaction_score]
        save_path: Path to save figure (if None, will display)
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    for op in OPERATIONS:
        op_data = interaction_df[interaction_df['op'] == op]
        if len(op_data) > 0:
            plt.plot(op_data['layer'], op_data['interaction_score'],
                    marker='o', label=op, linewidth=2)

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Interaction Score', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_interaction_comparison(
    interaction_model_output, interaction_gt, interaction_baseline,
    save_path=None
):
    """
    Plot all three types of interactions on the same graph:
    1. F1, F2 → model output (Z)
    2. F1, F2 → correct answer
    3. x1, x2 → correct answer (baseline)

    Args:
        interaction_model_output: DataFrame with [layer, op, interaction_score] for model output
        interaction_gt: DataFrame with [layer, op, interaction_score] for ground truth
        interaction_baseline: DataFrame with [op, interaction_score] for baseline (no layers)
        save_path: Path to save figure
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


def plot_interaction_correct_vs_incorrect(
    interaction_correct, interaction_incorrect, operation='mul',
    save_path=None
):
    """
    Plot interaction scores for correct vs incorrect examples.

    Args:
        interaction_correct: DataFrame with correct examples
        interaction_incorrect: DataFrame with incorrect examples
        operation: Operation to plot
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    # Filter by operation
    correct_data = interaction_correct[interaction_correct['op'] == operation]
    incorrect_data = interaction_incorrect[interaction_incorrect['op'] == operation]

    if len(correct_data) > 0:
        plt.plot(correct_data['layer'], correct_data['interaction_score'],
                marker='o', label='Correct', linewidth=2, color='green')

    if len(incorrect_data) > 0:
        plt.plot(incorrect_data['layer'], incorrect_data['interaction_score'],
                marker='s', label='Incorrect', linewidth=2, color='red')

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Interaction Score', fontsize=12)
    plt.title(f'Interaction Score: Correct vs Incorrect ({operation})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_probe_quality(metrics, save_path=None):
    """
    Plot probe quality (R² scores) vs layer.

    Args:
        metrics: Dictionary with r2_op1, r2_op2, r2_correct and MAE lists
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    layers = np.arange(len(metrics['r2_op1']))

    # R² scores
    ax1.plot(layers, metrics['r2_op1'], marker='o', label='Operand 1', linewidth=2)
    ax1.plot(layers, metrics['r2_op2'], marker='s', label='Operand 2', linewidth=2)
    ax1.plot(layers, metrics['r2_correct'], marker='^', label='Correct Answer', linewidth=2)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Probe Quality: R² Score by Layer', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # MAE scores
    ax2.plot(layers, metrics['mae_op1'], marker='o', label='Operand 1', linewidth=2)
    ax2.plot(layers, metrics['mae_op2'], marker='s', label='Operand 2', linewidth=2)
    ax2.plot(layers, metrics['mae_correct'], marker='^', label='Correct Answer', linewidth=2)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('Probe Quality: MAE by Layer', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_model_accuracy_by_operation(df, save_path=None):
    """
    Plot model accuracy by operation type.

    Args:
        df: DataFrame with columns [op, is_correct]
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    accuracies = []
    operations = []

    for op in OPERATIONS:
        op_data = df[df['op'] == op]
        if len(op_data) > 0:
            accuracy = op_data['is_correct'].mean()
            accuracies.append(accuracy)
            operations.append(op)

    bars = plt.bar(operations, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple'])

    plt.xlabel('Operation', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy by Operation', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_interaction_heatmap(interaction_df, save_path=None):
    """
    Plot heatmap of interaction scores: operations vs layers.

    Args:
        interaction_df: DataFrame with columns [layer, op, interaction_score]
        save_path: Path to save figure
    """
    # Pivot data to create matrix
    pivot_data = interaction_df.pivot(index='op', columns='layer', values='interaction_score')

    plt.figure(figsize=(14, 6))
    im = plt.imshow(pivot_data.values, aspect='auto', cmap='viridis', interpolation='nearest')

    # Set ticks
    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
    plt.yticks(range(len(pivot_data.index)), pivot_data.index)

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Operation', fontsize=12)
    plt.title('Interaction Score Heatmap', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Interaction Score', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def generate_all_plots():
    """Generate all plots for the analysis."""
    from interaction_analysis import load_interaction_scores
    from probe_training import load_probe_results
    from dataset_generator import load_dataset
    from model_inference import load_hidden_states

    print("Loading data for visualization...")

    # Load data
    df = load_dataset()
    hidden_states = load_hidden_states()
    probe_results = load_probe_results(hidden_dim=hidden_states['h1_all'].shape[2])

    # Load interaction scores
    from config import INTERACTION_SCORES_PATH
    interaction_all = load_interaction_scores()

    try:
        interaction_correct = load_interaction_scores(
            INTERACTION_SCORES_PATH.replace('.csv', '_correct.csv')
        )
    except:
        interaction_correct = None

    try:
        interaction_incorrect = load_interaction_scores(
            INTERACTION_SCORES_PATH.replace('.csv', '_incorrect.csv')
        )
    except:
        interaction_incorrect = None

    try:
        interaction_gt = load_interaction_scores(
            INTERACTION_SCORES_PATH.replace('.csv', '_gt.csv')
        )
    except:
        interaction_gt = None

    try:
        interaction_baseline = load_interaction_scores(
            INTERACTION_SCORES_PATH.replace('.csv', '_baseline.csv')
        )
    except:
        interaction_baseline = None

    print("\nGenerating plots...")

    # 1. Interaction comparison: model output vs ground truth vs baseline
    if interaction_gt is not None and interaction_baseline is not None:
        plot_interaction_comparison(
            interaction_all,
            interaction_gt,
            interaction_baseline,
            save_path=os.path.join(PLOTS_DIR, 'interaction_comparison.png')
        )

    # 2. Interaction by layer (all operations) - model output only
    plot_interaction_by_layer(
        interaction_all,
        save_path=os.path.join(PLOTS_DIR, 'interaction_by_layer.png'),
        title='Interaction Score by Layer: F1, F2 → Model Output'
    )

    # 3. Interaction: correct vs incorrect (for multiplication)
    if interaction_correct is not None and interaction_incorrect is not None:
        plot_interaction_correct_vs_incorrect(
            interaction_correct,
            interaction_incorrect,
            operation='mul',
            save_path=os.path.join(PLOTS_DIR, 'interaction_correct_vs_incorrect_mul.png')
        )

    # 4. Probe quality
    plot_probe_quality(
        probe_results['metrics'],
        save_path=os.path.join(PLOTS_DIR, 'probe_quality.png')
    )

    # 5. Model accuracy by operation
    plot_model_accuracy_by_operation(
        df,
        save_path=os.path.join(PLOTS_DIR, 'accuracy_by_operation.png')
    )

    # 6. Interaction heatmap
    plot_interaction_heatmap(
        interaction_all,
        save_path=os.path.join(PLOTS_DIR, 'interaction_heatmap.png')
    )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()
