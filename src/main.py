"""
Main pipeline for arithmetic reasoning interpretability experiment.

This script orchestrates the entire experiment:
1. Generate dataset
2. Tokenize and find operand positions
3. Run model inference and extract hidden states
4. Train numeric-decoding probes
5. Compute MI-based synergy scores
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

    from config import DATASET_PATH, DIFFICULTY_LEVELS, EXAMPLES_PER_OP_DIFFICULTY, OPERATIONS
    print(f"Operations: {OPERATIONS}")
    print(f"Difficulty levels: {DIFFICULTY_LEVELS}")
    print(f"Examples per (op, difficulty): {EXAMPLES_PER_OP_DIFFICULTY}")
    print(f"Total examples: {len(OPERATIONS) * len(DIFFICULTY_LEVELS) * EXAMPLES_PER_OP_DIFFICULTY}")

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
        probe_results = load_probe_results(hidden_dim=hidden_states['h_output_all'].shape[2])
    else:
        from probe_training import train_probes_all_layers, save_probe_results

        # All probes are trained on h_output (the "=" position)
        h_output_all = hidden_states['h_output_all']
        x1_values = df['x1'].values
        x2_values = df['x2'].values
        correct_answers = df['correct_answer'].values

        probe_results = train_probes_all_layers(
            h_output_all, x1_values, x2_values, correct_answers
        )
        save_probe_results(probe_results)

    # Step 5: Compute MI-based synergy
    print("\n" + "="*80)
    print("STEP 5: Compute MI-Based Synergy (KSG Estimator)")
    print("="*80)

    from mi_synergy_analysis import (
        compute_all_mi_synergy, save_mi_synergy_results
    )
    from config import RESULTS_DIR, MI_KSG_NEIGHBORS, MI_PCA_COMPONENTS, MI_NEXT_LAYER_PCA_COMPONENTS

    mi_synergy_path = os.path.join(RESULTS_DIR, 'mi_synergy_scalar_probe.csv')
    if skip_existing and os.path.exists(mi_synergy_path):
        print(f"MI synergy results already exist at {mi_synergy_path}, skipping...")
        from mi_synergy_analysis import load_mi_synergy_results
        mi_results = load_mi_synergy_results()
    else:
        print("Computing MI-based synergy using low-dimensional representations...")
        print(f"  - All features extracted from '=' position")
        print(f"  - Scalar probe outputs (1D)")
        print(f"  - PCA reduced features: {MI_PCA_COMPONENTS}")
        print(f"  - Next-layer target: PCA-{MI_NEXT_LAYER_PCA_COMPONENTS}")
        print(f"  - KSG neighbors (k): {MI_KSG_NEIGHBORS}")

        mi_results = compute_all_mi_synergy(
            probes_op1=probe_results['probes_op1'],
            probes_op2=probe_results['probes_op2'],
            h_output_all=hidden_states['h_output_all'],
            F1_all=probe_results['F1_all'],
            F2_all=probe_results['F2_all'],
            Z_all=hidden_states['Z_all'],
            df=df,
            model=None,  # Skip log-prob method (requires re-inference)
            tokenizer=None,
            pca_components=MI_PCA_COMPONENTS,
            next_layer_pca_components=MI_NEXT_LAYER_PCA_COMPONENTS,
            k=MI_KSG_NEIGHBORS
        )
        save_mi_synergy_results(mi_results)

    # Step 6: Generate visualizations
    print("\n" + "="*80)
    print("STEP 6: Generate Visualizations")
    print("="*80)

    from visualization import generate_all_plots
    generate_all_plots(mi_results)

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nResults saved in:")
    print(f"  - Dataset: {DATASET_PATH}")
    print(f"  - Hidden states: {HIDDEN_STATES_PATH}")
    print(f"  - Probe features: {FEATURES_PATH}")
    print(f"  - MI synergy (scalar probe): {os.path.join(RESULTS_DIR, 'mi_synergy_scalar_probe.csv')}")
    print(f"  - MI synergy (PCA-3): {os.path.join(RESULTS_DIR, 'mi_synergy_pca_3.csv')}")
    print(f"  - MI synergy (PCA-5): {os.path.join(RESULTS_DIR, 'mi_synergy_pca_5.csv')}")
    print(f"  - MI synergy (baseline): {os.path.join(RESULTS_DIR, 'mi_synergy_raw_baseline.csv')}")
    from config import PLOTS_DIR
    print(f"  - Plots: {PLOTS_DIR}/")


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
