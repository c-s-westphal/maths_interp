"""
Analyze whether high interaction information examples lead to correct answers.

This script identifies examples with:
1. Highest average interaction across layers
2. Highest final-layer interaction
And checks their correctness rates.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import OPERATIONS, PLOTS_DIR, set_seed
import os


def compute_per_example_interactions(F1_all, F2_all, Z_all, df, interaction_predictor):
    """
    Compute TRUE per-example synergy scores using trained models.

    For each example, we measure synergy as:
    synergy = 0.5*(err_F1 + err_F2) - err_joint

    Where:
    - err_joint: squared error of joint model f(F1, F2) → Z
    - err_F1: squared error of marginal model g(F1) → Z
    - err_F2: squared error of marginal model h(F2) → Z

    Positive synergy → joint model is better (true interaction!)
    Negative synergy → joint model is worse (anti-synergy)

    Args:
        F1_all: [num_examples, num_layers, feature_dim]
        F2_all: [num_examples, num_layers, feature_dim]
        Z_all: [num_examples]
        df: DataFrame with metadata
        interaction_predictor: Not used (kept for backwards compatibility)

    Returns:
        DataFrame with per-example synergy scores
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    num_examples, num_layers, feature_dim = F1_all.shape

    print("Training models for per-example synergy computation...")

    # Train models for each layer (on ALL data)
    joint_models = []
    marginal_F1_models = []
    marginal_F2_models = []

    for layer_idx in range(num_layers):
        F1_layer = F1_all[:, layer_idx, :]
        F2_layer = F2_all[:, layer_idx, :]
        X_joint = np.concatenate([F1_layer, F2_layer], axis=1)

        # Train joint model
        joint_model = Ridge(alpha=1.0)
        joint_model.fit(X_joint, Z_all)
        joint_models.append(joint_model)

        # Train marginal models
        marginal_F1 = Ridge(alpha=1.0)
        marginal_F1.fit(F1_layer, Z_all)
        marginal_F1_models.append(marginal_F1)

        marginal_F2 = Ridge(alpha=1.0)
        marginal_F2.fit(F2_layer, Z_all)
        marginal_F2_models.append(marginal_F2)

    print("Computing per-example synergy scores...")

    results = []

    for idx in range(num_examples):
        example_data = {
            'example_idx': idx,
            'op': df.iloc[idx]['op'],
            'is_correct': df.iloc[idx]['is_correct'],
            'x1': df.iloc[idx]['x1'],
            'x2': df.iloc[idx]['x2'],
            'correct_answer': df.iloc[idx]['correct_answer']
        }

        Z_true = Z_all[idx]
        synergy_scores = []

        for layer_idx in range(num_layers):
            F1 = F1_all[idx, layer_idx, :].reshape(1, -1)
            F2 = F2_all[idx, layer_idx, :].reshape(1, -1)
            X_joint = np.concatenate([F1, F2], axis=1)

            # Predict with each model
            pred_joint = joint_models[layer_idx].predict(X_joint)[0]
            pred_F1 = marginal_F1_models[layer_idx].predict(F1)[0]
            pred_F2 = marginal_F2_models[layer_idx].predict(F2)[0]

            # Compute squared errors
            err_joint = (Z_true - pred_joint) ** 2
            err_F1 = (Z_true - pred_F1) ** 2
            err_F2 = (Z_true - pred_F2) ** 2

            # Synergy: how much better is joint than average of marginals?
            # Higher synergy = better joint performance
            synergy = 0.5 * (err_F1 + err_F2) - err_joint
            synergy_scores.append(synergy)

        example_data['interaction_avg'] = np.mean(synergy_scores)
        example_data['interaction_final'] = synergy_scores[-1]
        example_data['interaction_trajectory'] = synergy_scores

        results.append(example_data)

    return pd.DataFrame(results)


def analyze_high_vs_low_interaction(per_example_df, metric='interaction_avg', top_k=100):
    """
    Compare correctness rates for high vs low interaction examples.

    Args:
        per_example_df: DataFrame with per-example interaction scores
        metric: 'interaction_avg' or 'interaction_final'
        top_k: Number of top/bottom examples to compare

    Returns:
        Dictionary with analysis results
    """
    results = {}

    for op in OPERATIONS:
        op_data = per_example_df[per_example_df['op'] == op].copy()

        if len(op_data) == 0:
            continue

        # Sort by interaction metric
        op_data_sorted = op_data.sort_values(metric, ascending=False)

        # Get top-k and bottom-k
        actual_k = min(top_k, len(op_data) // 4)  # Use at most 25% of data

        top_k_examples = op_data_sorted.head(actual_k)
        bottom_k_examples = op_data_sorted.tail(actual_k)

        # Compute accuracy
        top_k_accuracy = top_k_examples['is_correct'].mean()
        bottom_k_accuracy = bottom_k_examples['is_correct'].mean()
        overall_accuracy = op_data['is_correct'].mean()

        results[op] = {
            'top_k': actual_k,
            'top_k_accuracy': top_k_accuracy,
            'bottom_k_accuracy': bottom_k_accuracy,
            'overall_accuracy': overall_accuracy,
            'diff': top_k_accuracy - bottom_k_accuracy,
            'total_examples': len(op_data)
        }

    return results


def plot_interaction_vs_correctness(per_example_df, save_dir):
    """
    Plot per-example synergy trajectories for correct vs incorrect examples.

    Higher synergy = joint model (F1, F2) performs better than marginal models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, op in enumerate(OPERATIONS):
        ax = axes[idx]

        op_data = per_example_df[per_example_df['op'] == op]

        if len(op_data) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(f'{op.upper()}', fontsize=12, fontweight='bold')
            continue

        correct_data = op_data[op_data['is_correct'] == True]
        incorrect_data = op_data[op_data['is_correct'] == False]

        # Compute mean trajectories
        if len(correct_data) > 0:
            correct_trajectories = np.array([t for t in correct_data['interaction_trajectory']])
            correct_mean = correct_trajectories.mean(axis=0)
            correct_std = correct_trajectories.std(axis=0)
            layers = np.arange(len(correct_mean))

            ax.plot(layers, correct_mean, color='green', linewidth=2, label='Correct')
            ax.fill_between(layers, correct_mean - correct_std, correct_mean + correct_std,
                           color='green', alpha=0.2)

        if len(incorrect_data) > 0:
            incorrect_trajectories = np.array([t for t in incorrect_data['interaction_trajectory']])
            incorrect_mean = incorrect_trajectories.mean(axis=0)
            incorrect_std = incorrect_trajectories.std(axis=0)
            layers = np.arange(len(incorrect_mean))

            ax.plot(layers, incorrect_mean, color='red', linewidth=2, label='Incorrect')
            ax.fill_between(layers, incorrect_mean - incorrect_std, incorrect_mean + incorrect_std,
                           color='red', alpha=0.2)

        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Per-Example Synergy', fontsize=10)
        ax.set_title(f'{op.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)  # Zero line

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle('Per-Example Synergy: Correct vs Incorrect\n(Higher = better joint model performance)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'interaction_trajectories_correct_vs_incorrect.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def main():
    """Run interaction trajectory analysis."""
    set_seed()

    print("Loading data...")
    from dataset_generator import load_dataset
    from model_inference import load_hidden_states
    from probe_training import load_probe_results

    df = load_dataset()
    hidden_states = load_hidden_states()
    probe_results = load_probe_results(hidden_dim=hidden_states['h1_all'].shape[2])

    F1_all = probe_results['F1_all']
    F2_all = probe_results['F2_all']
    Z_all = hidden_states['Z_all']

    print("\nComputing per-example interaction scores...")
    per_example_df = compute_per_example_interactions(F1_all, F2_all, Z_all, df, None)

    print("\n" + "="*80)
    print("Analysis: High vs Low Interaction (Average across layers)")
    print("="*80)
    results_avg = analyze_high_vs_low_interaction(per_example_df, metric='interaction_avg', top_k=100)

    for op, stats in results_avg.items():
        print(f"\n{op.upper()}:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Overall accuracy: {stats['overall_accuracy']:.1%}")
        print(f"  Top {stats['top_k']} (high avg interaction): {stats['top_k_accuracy']:.1%}")
        print(f"  Bottom {stats['top_k']} (low avg interaction): {stats['bottom_k_accuracy']:.1%}")
        print(f"  Difference: {stats['diff']:+.1%}")

    print("\n" + "="*80)
    print("Analysis: High vs Low Interaction (Final layer)")
    print("="*80)
    results_final = analyze_high_vs_low_interaction(per_example_df, metric='interaction_final', top_k=100)

    for op, stats in results_final.items():
        print(f"\n{op.upper()}:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Overall accuracy: {stats['overall_accuracy']:.1%}")
        print(f"  Top {stats['top_k']} (high final interaction): {stats['top_k_accuracy']:.1%}")
        print(f"  Bottom {stats['top_k']} (low final interaction): {stats['bottom_k_accuracy']:.1%}")
        print(f"  Difference: {stats['diff']:+.1%}")

    print("\nGenerating visualization...")
    plot_interaction_vs_correctness(per_example_df, PLOTS_DIR)

    # Save results
    from config import RESULTS_DIR
    results_df = pd.DataFrame([
        {
            'op': op,
            'metric': 'avg',
            'top_k': stats['top_k'],
            'top_k_accuracy': stats['top_k_accuracy'],
            'bottom_k_accuracy': stats['bottom_k_accuracy'],
            'overall_accuracy': stats['overall_accuracy'],
            'difference': stats['diff']
        }
        for op, stats in results_avg.items()
    ] + [
        {
            'op': op,
            'metric': 'final',
            'top_k': stats['top_k'],
            'top_k_accuracy': stats['top_k_accuracy'],
            'bottom_k_accuracy': stats['bottom_k_accuracy'],
            'overall_accuracy': stats['overall_accuracy'],
            'difference': stats['diff']
        }
        for op, stats in results_final.items()
    ])

    save_path = os.path.join(RESULTS_DIR, 'interaction_trajectory_analysis.csv')
    results_df.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
