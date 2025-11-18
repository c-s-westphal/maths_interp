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
        metrics: Dictionary with r2_op1 and r2_op2 lists
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    layers = np.arange(len(metrics['r2_op1']))

    # R² scores
    ax1.plot(layers, metrics['r2_op1'], marker='o', label='Operand 1', linewidth=2)
    ax1.plot(layers, metrics['r2_op2'], marker='s', label='Operand 2', linewidth=2)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Probe Quality: R² Score by Layer', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # MAE scores
    ax2.plot(layers, metrics['mae_op1'], marker='o', label='Operand 1', linewidth=2)
    ax2.plot(layers, metrics['mae_op2'], marker='s', label='Operand 2', linewidth=2)
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

    print("\nGenerating plots...")

    # 1. Interaction by layer (all operations)
    plot_interaction_by_layer(
        interaction_all,
        save_path=os.path.join(PLOTS_DIR, 'interaction_by_layer.png'),
        title='Interaction Score by Layer (All Examples)'
    )

    # 2. Interaction: correct vs incorrect (for multiplication)
    if interaction_correct is not None and interaction_incorrect is not None:
        plot_interaction_correct_vs_incorrect(
            interaction_correct,
            interaction_incorrect,
            operation='mul',
            save_path=os.path.join(PLOTS_DIR, 'interaction_correct_vs_incorrect_mul.png')
        )

    # 3. Probe quality
    plot_probe_quality(
        probe_results['metrics'],
        save_path=os.path.join(PLOTS_DIR, 'probe_quality.png')
    )

    # 4. Model accuracy by operation
    plot_model_accuracy_by_operation(
        df,
        save_path=os.path.join(PLOTS_DIR, 'accuracy_by_operation.png')
    )

    # 5. Interaction heatmap
    plot_interaction_heatmap(
        interaction_all,
        save_path=os.path.join(PLOTS_DIR, 'interaction_heatmap.png')
    )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()
