"""
Generate visualizations for MI-based synergy analysis.

All plots are generated in both scalar probe and PCA versions.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from config import PLOTS_DIR, OPERATIONS, DIFFICULTY_LEVELS


def plot_synergy_by_layer(synergy_df, save_path=None, title="MI Synergy by Layer"):
    """
    Plot MI synergy scores vs layer for each operation.

    Args:
        synergy_df: DataFrame with columns [layer, op, synergy]
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 7))

    colors = plt.cm.Set1(np.linspace(0, 1, len(OPERATIONS)))

    for op, color in zip(OPERATIONS, colors):
        op_data = synergy_df[synergy_df['op'] == op]
        if len(op_data) > 0:
            plt.plot(op_data['layer'], op_data['synergy'],
                    marker='o', label=op, linewidth=2, color=color, markersize=5)

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('MI Synergy (nats)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_mi_components(synergy_df, operation='mul', save_path=None):
    """
    Plot MI components (I(F1,F2;Z), I(F1;Z), I(F2;Z)) and synergy for a specific operation.

    Args:
        synergy_df: DataFrame with columns [layer, op, I_F12_Z, I_F1_Z, I_F2_Z, synergy]
        operation: Which operation to plot
        save_path: Path to save figure
    """
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
    ax2.fill_between(op_data['layer'], 0, op_data['synergy'], alpha=0.3, color='purple')
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


def plot_synergy_heatmap(synergy_df, save_path=None, title="MI Synergy Heatmap"):
    """
    Plot heatmap of MI synergy: operations vs layers.

    Args:
        synergy_df: DataFrame with columns [layer, op, synergy]
        save_path: Path to save figure
        title: Plot title
    """
    # Pivot data
    pivot_data = synergy_df.pivot(index='op', columns='layer', values='synergy')

    plt.figure(figsize=(16, 6))

    # Use diverging colormap centered at 0
    vmax = max(abs(pivot_data.values.min()), abs(pivot_data.values.max()))
    im = plt.imshow(pivot_data.values, aspect='auto', cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax, interpolation='nearest')

    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
    plt.yticks(range(len(pivot_data.index)), pivot_data.index)

    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Operation', fontsize=12)
    plt.title(title, fontsize=14)

    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('MI Synergy (nats)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_synergy_by_operation(synergy_df, layer=None, save_path=None, title="MI Synergy by Operation"):
    """
    Bar plot of synergy by operation (averaged across layers or for specific layer).

    Args:
        synergy_df: DataFrame with columns [layer, op, synergy]
        layer: If None, average across all layers. Otherwise, use specific layer.
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    if layer is not None:
        data = synergy_df[synergy_df['layer'] == layer]
        title = f"{title} (Layer {layer})"
    else:
        data = synergy_df.groupby('op')['synergy'].mean().reset_index()

    ops = []
    synergies = []
    for op in OPERATIONS:
        op_data = data[data['op'] == op]
        if len(op_data) > 0:
            ops.append(op)
            if 'synergy' in op_data.columns:
                synergies.append(op_data['synergy'].values[0] if layer is not None
                               else op_data['synergy'].values[0])

    colors = plt.cm.Set2(np.linspace(0, 1, len(ops)))
    bars = plt.bar(ops, synergies, color=colors, edgecolor='black', linewidth=0.5)

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Operation', fontsize=12)
    plt.ylabel('MI Synergy (nats)', fontsize=12)
    plt.title(title, fontsize=14)
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


def plot_probe_quality(metrics, save_path=None):
    """
    Plot probe quality (R² scores and MAE) vs layer.

    Args:
        metrics: Dictionary with r2_op1, r2_op2, r2_correct and MAE lists
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    layers = np.arange(len(metrics['r2_op1']))

    # R² scores
    ax1.plot(layers, metrics['r2_op1'], marker='o', label='Operand 1 (x̂₁)', linewidth=2)
    ax1.plot(layers, metrics['r2_op2'], marker='s', label='Operand 2 (x̂₂)', linewidth=2)
    ax1.plot(layers, metrics['r2_correct'], marker='^', label='Correct Answer (ŷ)', linewidth=2)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Probe Quality: R² Score by Layer', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # MAE scores
    ax2.plot(layers, metrics['mae_op1'], marker='o', label='Operand 1 (x̂₁)', linewidth=2)
    ax2.plot(layers, metrics['mae_op2'], marker='s', label='Operand 2 (x̂₂)', linewidth=2)
    ax2.plot(layers, metrics['mae_correct'], marker='^', label='Correct Answer (ŷ)', linewidth=2)
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


def plot_model_accuracy(df, save_path=None):
    """
    Plot model accuracy by operation and difficulty.

    Args:
        df: DataFrame with columns [op, difficulty, is_correct]
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # By operation
    op_acc = df.groupby('op')['is_correct'].mean()
    colors = plt.cm.Set2(np.linspace(0, 1, len(OPERATIONS)))
    bars1 = ax1.bar([op for op in OPERATIONS if op in op_acc.index],
                    [op_acc[op] for op in OPERATIONS if op in op_acc.index],
                    color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Operation', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy by Operation', fontsize=14)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)

    # By difficulty
    diff_acc = df.groupby('difficulty')['is_correct'].mean()
    diff_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(DIFFICULTY_LEVELS)))
    bars2 = ax2.bar([d for d in DIFFICULTY_LEVELS if d in diff_acc.index],
                    [diff_acc[d] for d in DIFFICULTY_LEVELS if d in diff_acc.index],
                    color=diff_colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Difficulty', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy by Difficulty', fontsize=14)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_synergy_correct_vs_incorrect(synergy_correct, synergy_incorrect, save_path=None):
    """
    Compare synergy for correct vs incorrect predictions.

    Args:
        synergy_correct: DataFrame with synergy for correct examples
        synergy_incorrect: DataFrame with synergy for incorrect examples
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, op in enumerate(OPERATIONS):
        if idx >= len(axes):
            break

        ax = axes[idx]

        correct_data = synergy_correct[synergy_correct['op'] == op]
        incorrect_data = synergy_incorrect[synergy_incorrect['op'] == op]

        if len(correct_data) > 0:
            ax.plot(correct_data['layer'], correct_data['synergy'],
                   marker='o', label='Correct', linewidth=2, color='green')

        if len(incorrect_data) > 0:
            ax.plot(incorrect_data['layer'], incorrect_data['synergy'],
                   marker='s', label='Incorrect', linewidth=2, color='red')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('MI Synergy (nats)', fontsize=10)
        ax.set_title(f'{op.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Remove unused subplot
    if len(OPERATIONS) < len(axes):
        for idx in range(len(OPERATIONS), len(axes)):
            fig.delaxes(axes[idx])

    plt.suptitle('MI Synergy: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_method_comparison(results_dict, save_path=None):
    """
    Compare synergy across different methods (scalar probe vs PCA).

    Args:
        results_dict: Dictionary of DataFrames from compute_all_mi_synergy
        save_path: Path to save figure
    """
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
                label = method_name.replace('_', ' ').title()
                ax.plot(op_data['layer'], op_data['synergy'],
                       marker='o', label=label, linewidth=2, alpha=0.8, markersize=4)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('MI Synergy (nats)', fontsize=10)
        ax.set_title(f'{op.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Remove unused subplots
    for idx in range(len(OPERATIONS), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('MI Synergy: Scalar Probe vs PCA Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_baseline_comparison(baseline_df, save_path=None):
    """
    Plot baseline synergy (raw operands) by operation.

    Args:
        baseline_df: DataFrame with columns [op, synergy]
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    ops = [op for op in OPERATIONS if op in baseline_df['op'].values]
    synergies = [baseline_df[baseline_df['op'] == op]['synergy'].values[0] for op in ops]

    colors = plt.cm.Set2(np.linspace(0, 1, len(ops)))
    bars = plt.bar(ops, synergies, color=colors, edgecolor='black', linewidth=0.5)

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Operation', fontsize=12)
    plt.ylabel('MI Synergy (nats)', fontsize=12)
    plt.title('Baseline MI Synergy: Raw Operands (x₁, x₂) → Z', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

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


def generate_all_plots(mi_results=None):
    """
    Generate all visualization plots.

    Args:
        mi_results: Dictionary of MI synergy results. If None, loads from disk.
    """
    from mi_synergy_analysis import load_mi_synergy_results
    from probe_training import load_probe_results
    from dataset_generator import load_dataset
    from model_inference import load_hidden_states

    print("Loading data for visualization...")

    # Load data
    df = load_dataset()
    hidden_states = load_hidden_states()
    probe_results = load_probe_results(hidden_dim=hidden_states['h1_all'].shape[2])

    # Load MI synergy results
    if mi_results is None:
        mi_results = load_mi_synergy_results()

    print("\nGenerating plots...")

    # === SCALAR PROBE PLOTS ===
    if 'scalar_probe' in mi_results:
        scalar_df = mi_results['scalar_probe']

        # 1. Synergy by layer
        plot_synergy_by_layer(
            scalar_df,
            save_path=os.path.join(PLOTS_DIR, 'synergy_by_layer_scalar.png'),
            title='MI Synergy by Layer (Scalar Probe: x̂₁, x̂₂)'
        )

        # 2. Synergy heatmap
        plot_synergy_heatmap(
            scalar_df,
            save_path=os.path.join(PLOTS_DIR, 'synergy_heatmap_scalar.png'),
            title='MI Synergy Heatmap (Scalar Probe)'
        )

        # 3. MI components for each operation
        for op in OPERATIONS:
            plot_mi_components(
                scalar_df,
                operation=op,
                save_path=os.path.join(PLOTS_DIR, f'mi_components_{op}_scalar.png')
            )

        # 4. Synergy by operation (average)
        plot_synergy_by_operation(
            scalar_df,
            save_path=os.path.join(PLOTS_DIR, 'synergy_by_operation_scalar.png'),
            title='Average MI Synergy by Operation (Scalar Probe)'
        )

    # === PCA PLOTS ===
    for pca_key in ['pca_3', 'pca_5']:
        if pca_key in mi_results:
            pca_df = mi_results[pca_key]
            n_comp = pca_key.split('_')[1]

            # 1. Synergy by layer
            plot_synergy_by_layer(
                pca_df,
                save_path=os.path.join(PLOTS_DIR, f'synergy_by_layer_{pca_key}.png'),
                title=f'MI Synergy by Layer (PCA-{n_comp})'
            )

            # 2. Synergy heatmap
            plot_synergy_heatmap(
                pca_df,
                save_path=os.path.join(PLOTS_DIR, f'synergy_heatmap_{pca_key}.png'),
                title=f'MI Synergy Heatmap (PCA-{n_comp})'
            )

            # 3. MI components for multiplication
            plot_mi_components(
                pca_df,
                operation='mul',
                save_path=os.path.join(PLOTS_DIR, f'mi_components_mul_{pca_key}.png')
            )

            # 4. Synergy by operation
            plot_synergy_by_operation(
                pca_df,
                save_path=os.path.join(PLOTS_DIR, f'synergy_by_operation_{pca_key}.png'),
                title=f'Average MI Synergy by Operation (PCA-{n_comp})'
            )

    # === METHOD COMPARISON ===
    plot_method_comparison(
        mi_results,
        save_path=os.path.join(PLOTS_DIR, 'synergy_method_comparison.png')
    )

    # === BASELINE ===
    if 'raw_baseline' in mi_results:
        plot_baseline_comparison(
            mi_results['raw_baseline'],
            save_path=os.path.join(PLOTS_DIR, 'synergy_baseline.png')
        )

    # === PROBE QUALITY ===
    plot_probe_quality(
        probe_results['metrics'],
        save_path=os.path.join(PLOTS_DIR, 'probe_quality.png')
    )

    # === MODEL ACCURACY ===
    plot_model_accuracy(
        df,
        save_path=os.path.join(PLOTS_DIR, 'model_accuracy.png')
    )

    print("\nAll plots generated successfully!")
    print(f"Plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    generate_all_plots()
