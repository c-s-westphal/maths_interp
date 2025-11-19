"""
Main pipeline for arithmetic reasoning interpretability experiment.

This script orchestrates the entire experiment:
1. Generate dataset
2. Tokenize and find operand positions
3. Run model inference and extract hidden states
4. Train numeric-decoding probes
5. Compute interaction scores
6. Generate visualizations
"""
import os
import sys
import argparse
from config import set_seed, DEVICE
import time


def run_full_pipeline(skip_existing=False, model_override=None):
    """
    Run the complete experimental pipeline.

    Args:
        skip_existing: If True, skip steps where output files already exist
        model_override: If provided, use this model name instead of config default
    """
    # Override model name if provided
    if model_override:
        import config
        config.MODEL_NAME = model_override
        print(f"Using model override: {model_override}")

    print("="*80)
    print("Arithmetic Reasoning Interpretability Experiment")
    print("="*80)
    print(f"Device: {DEVICE}")
    from config import MODEL_NAME
    print(f"Model: {MODEL_NAME}")
    print()

    # Set random seed
    set_seed()

    # Step 1: Generate dataset
    print("\n" + "="*80)
    print("STEP 1: Generate Arithmetic Dataset")
    print("="*80)

    from config import DATASET_PATH
    if skip_existing and os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at {DATASET_PATH}, skipping generation...")
        from dataset_generator import load_dataset
        df = load_dataset()
    else:
        from dataset_generator import generate_dataset, save_dataset
        df = generate_dataset()
        save_dataset(df)

    # Step 2: Tokenization
    print("\n" + "="*80)
    print("STEP 2: Tokenize and Find Operand Positions")
    print("="*80)

    from model_inference import load_model_and_tokenizer
    from tokenizer_utils import process_dataset_tokenization
    from dataset_generator import save_dataset

    model, tokenizer = load_model_and_tokenizer()

    if 'input_ids' not in df.columns or df['input_ids'].isna().any():
        df = process_dataset_tokenization(df, tokenizer)
        save_dataset(df)
    else:
        print("Dataset already tokenized, skipping...")

    # Step 3: Model inference
    print("\n" + "="*80)
    print("STEP 3: Run Model Inference and Extract Hidden States")
    print("="*80)

    from config import HIDDEN_STATES_PATH
    if skip_existing and os.path.exists(HIDDEN_STATES_PATH):
        print(f"Hidden states already exist at {HIDDEN_STATES_PATH}, skipping inference...")
        from model_inference import load_hidden_states
        hidden_states = load_hidden_states()
    else:
        from model_inference import process_dataset_inference, save_hidden_states
        hidden_states = process_dataset_inference(df, model, tokenizer)
        save_hidden_states(hidden_states)

        # Update dataframe with predictions
        df['predicted_answer'] = hidden_states['predicted_answers']
        df['is_correct'] = hidden_states['is_correct']
        save_dataset(df)

    # Reload dataset with predictions
    from dataset_generator import load_dataset
    df = load_dataset()

    # Step 4: Train probes
    print("\n" + "="*80)
    print("STEP 4: Train Numeric-Decoding Probes")
    print("="*80)

    from config import FEATURES_PATH
    if skip_existing and os.path.exists(FEATURES_PATH):
        print(f"Probe features already exist at {FEATURES_PATH}, skipping training...")
        from probe_training import load_probe_results
        probe_results = load_probe_results(hidden_dim=hidden_states['h1_all'].shape[2])
    else:
        from probe_training import train_probes_all_layers, save_probe_results

        h1_all = hidden_states['h1_all']
        h2_all = hidden_states['h2_all']
        h_output_all = hidden_states['h_output_all']
        x1_values = df['x1'].values
        x2_values = df['x2'].values
        correct_answers = df['correct_answer'].values

        probe_results = train_probes_all_layers(
            h1_all, h2_all, h_output_all, x1_values, x2_values, correct_answers
        )
        save_probe_results(probe_results)

    # Step 5: Compute interaction scores
    print("\n" + "="*80)
    print("STEP 5: Compute Interaction Scores")
    print("="*80)

    from interaction_analysis import compute_all_interactions, compute_baseline_interaction, save_interaction_scores
    from config import INTERACTION_SCORES_PATH

    F1_all = probe_results['F1_all']
    F2_all = probe_results['F2_all']
    Z_all = hidden_states['Z_all']
    correct_answers = df['correct_answer'].values
    x1_values = df['x1'].values
    x2_values = df['x2'].values

    # Type 1: Interaction(F1, F2 → Z) - predicting model output
    if skip_existing and os.path.exists(INTERACTION_SCORES_PATH):
        print(f"Interaction scores (→ Z) already exist, skipping...")
    else:
        print("\n[1/3] Computing interaction: F1, F2 → model output (Z)...")
        interaction_df_all = compute_all_interactions(F1_all, F2_all, Z_all, df, target_name="model_output")
        save_interaction_scores(interaction_df_all)

    # Type 2: Interaction(F1, F2 → correct_answer) - predicting ground truth
    gt_path = INTERACTION_SCORES_PATH.replace('.csv', '_gt.csv')
    if not (skip_existing and os.path.exists(gt_path)):
        print("\n[2/3] Computing interaction: F1, F2 → correct answer...")
        interaction_df_gt = compute_all_interactions(
            F1_all, F2_all, correct_answers, df, target_name="correct_answer"
        )
        save_interaction_scores(interaction_df_gt, gt_path)

    # Type 3: Baseline interaction(x1, x2 → correct_answer) - raw numbers
    baseline_path = INTERACTION_SCORES_PATH.replace('.csv', '_baseline.csv')
    if not (skip_existing and os.path.exists(baseline_path)):
        print("\n[3/3] Computing baseline interaction: x1, x2 → correct answer (no layers)...")
        interaction_df_baseline = compute_baseline_interaction(
            x1_values, x2_values, correct_answers, df
        )
        save_interaction_scores(interaction_df_baseline, baseline_path)

    # Also compute for correct/incorrect (using model output Z)
    correct_path = INTERACTION_SCORES_PATH.replace('.csv', '_correct.csv')
    if not (skip_existing and os.path.exists(correct_path)):
        print("\nComputing interaction scores for correct examples...")
        interaction_df_correct = compute_all_interactions(
            F1_all, F2_all, Z_all, df, filter_by={'is_correct': True}, target_name="model_output"
        )
        save_interaction_scores(interaction_df_correct, correct_path)

    incorrect_path = INTERACTION_SCORES_PATH.replace('.csv', '_incorrect.csv')
    if not (skip_existing and os.path.exists(incorrect_path)):
        print("\nComputing interaction scores for incorrect examples...")
        interaction_df_incorrect = compute_all_interactions(
            F1_all, F2_all, Z_all, df, filter_by={'is_correct': False}, target_name="model_output"
        )
        save_interaction_scores(interaction_df_incorrect, incorrect_path)

    # Step 6: Generate visualizations
    print("\n" + "="*80)
    print("STEP 6: Generate Visualizations")
    print("="*80)

    from visualization import generate_all_plots
    generate_all_plots()

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nResults saved in:")
    print(f"  - Dataset: {DATASET_PATH}")
    print(f"  - Hidden states: {HIDDEN_STATES_PATH}")
    print(f"  - Probe features: {FEATURES_PATH}")
    print(f"  - Interaction scores: {INTERACTION_SCORES_PATH}")
    from config import PLOTS_DIR
    print(f"  - Plots: {PLOTS_DIR}/")
    print("\nSee README.md for details on interpreting results.")


def main():
    parser = argparse.ArgumentParser(
        description="Run arithmetic reasoning interpretability experiment"
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip steps where output files already exist'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Override model name (e.g., "microsoft/phi-2", "EleutherAI/pythia-2.8b")'
    )

    args = parser.parse_args()

    start_time = time.time()
    run_full_pipeline(skip_existing=args.skip_existing, model_override=args.model)
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
