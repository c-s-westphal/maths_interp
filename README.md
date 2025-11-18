# Operand Feature Interaction in Arithmetic Reasoning

An interpretability experiment studying how small language models internally represent numbers and combine them during arithmetic operations.

## Overview

This project investigates how the Pythia-160M language model:
- Represents operands (numbers) across different layers
- Combines operand features to produce arithmetic answers
- Shows different interaction patterns for different operations (add, sub, mul, max, min)
- Differs in representation between correct vs incorrect predictions

## Hypothesis

**Multiplication should show higher feature interaction than max/min operations** because:
- Multiplication requires computing relationships between digit positions
- Max/min can be solved by comparing numbers independently
- Addition shows moderate interaction (carry operations)

## Project Structure

```
maths_interp/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── config.py            # Configuration and constants
│   ├── dataset_generator.py # Generate arithmetic dataset
│   ├── tokenizer_utils.py   # Tokenization and position finding
│   ├── model_inference.py   # Model inference and hidden states
│   ├── probe_training.py    # Train numeric-decoding probes
│   ├── interaction_analysis.py  # Compute interaction scores
│   ├── visualization.py     # Generate plots
│   └── main.py             # Main pipeline orchestrator
├── data/                     # Generated datasets
├── results/                  # Saved models and features
│   └── probes/              # Per-layer probe models
└── plots/                    # Generated visualizations
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start: Run Full Pipeline

```bash
cd src
python main.py
```

This will:
1. Generate ~3,000 arithmetic examples (5 operations × 3 difficulties × 200 examples)
2. Tokenize and locate operand positions
3. Run Pythia-160M inference and extract hidden states
4. Train numeric-decoding probes for each layer
5. Compute interaction scores for all operations and layers
6. Generate visualization plots

**Expected runtime:** 20-40 minutes on CPU, 5-10 minutes on GPU

### Skip Already-Computed Steps

```bash
python main.py --skip-existing
```

### Run Individual Steps

```python
# Generate dataset only
python dataset_generator.py

# Test tokenization
python tokenizer_utils.py

# Run model inference
python model_inference.py

# Train probes
python probe_training.py

# Compute interactions
python interaction_analysis.py

# Generate plots
python visualization.py
```

## Methodology

### 1. Dataset Generation

**Operations:** add, sub, mul, max, min

**Difficulty levels:**
- Easy: 1-2 digit numbers (1-99)
- Medium: 3 digit numbers (100-999)
- Hard: 4 digit numbers (1000-9999)

**Prompt formats:**
```
add: "12 + 7 ="
sub: "34 - 19 ="
mul: "5 * 17 ="
max: "max(5, 17) ="
min: "min(12, 3) ="
```

### 2. Operand Position Finding

For each prompt, we:
1. Tokenize the full prompt
2. Separately tokenize operand strings (x1 and x2)
3. Find token subsequences in the prompt
4. Record the position of the **last token** of each operand

### 3. Hidden State Extraction

Run model with `output_hidden_states=True` and extract:
- `h1_ℓ`: Hidden state at operand 1 position, layer ℓ
- `h2_ℓ`: Hidden state at operand 2 position, layer ℓ
- `Z`: Logit of the predicted answer token

### 4. Numeric-Decoding Probes

For each layer ℓ, train two probes:
- **Probe 1:** `h1_ℓ → x̂1` (predict operand 1 value)
- **Probe 2:** `h2_ℓ → x̂2` (predict operand 2 value)

**Architecture:** MLP with one hidden layer (d → 64 → 1)

**Features:** Extract penultimate layer activations as `F1_ℓ` and `F2_ℓ`

### 5. Interaction Score Computation

For each layer ℓ and operation type:

1. Train predictor: `g([F1, F2]) → Z`
2. Compute performance scores:
   - `S_all`: Performance with true F1 and F2
   - `S_shuf_F1`: Performance with shuffled F1
   - `S_shuf_F2`: Performance with shuffled F2
   - `S_shuf_both`: Performance with both shuffled

3. Compute interaction:
```
Δ1 = S_all - S_shuf_F1
Δ2 = S_all - S_shuf_F2
Δ12 = S_all - S_shuf_both

Interaction = max(0, Δ12 - 0.5 × (Δ1 + Δ2))
```

**Interpretation:** Higher interaction = features must be combined to predict output

## Expected Results

### Probe Quality
- **Early layers:** Lower R² scores (operands not yet well-represented)
- **Middle/late layers:** Higher R² scores (clear numeric representations)
- **Final layers:** May decrease if model focuses on answer generation

### Interaction Patterns

**Multiplication vs Max/Min:**
- Multiplication should show **higher interaction scores** in middle/late layers
- Max/min should show **lower interaction** (more independent processing)
- Addition should be **intermediate**

**Correct vs Incorrect:**
- Correct predictions may show:
  - Higher interaction (better feature integration)
  - More consistent patterns across layers
- Incorrect predictions may show:
  - Lower interaction (failed to combine features)
  - Irregular patterns

## Generated Plots

1. **`interaction_by_layer.png`**
   - Interaction score vs layer for all operations
   - Compare mul (high) vs max/min (low)

2. **`interaction_correct_vs_incorrect_mul.png`**
   - Multiplication: correct vs incorrect examples
   - Check if correct shows higher interaction

3. **`probe_quality.png`**
   - R² and MAE for numeric probes across layers
   - Verify probes successfully decode numbers

4. **`accuracy_by_operation.png`**
   - Model accuracy for each operation type
   - Context for interaction results

5. **`interaction_heatmap.png`**
   - Heatmap: operations × layers
   - Visual summary of all patterns

## Configuration

Edit `src/config.py` to modify:
- Model name (`MODEL_NAME`)
- Dataset size (`EXAMPLES_PER_OP_DIFFICULTY`)
- Probe architecture (`PROBE_HIDDEN_DIM`)
- Training hyperparameters
- File paths

## Output Files

**Data:**
- `data/arithmetic_dataset.pkl`: Generated dataset with predictions

**Results:**
- `results/hidden_states.npz`: Extracted hidden states (h1, h2, Z)
- `results/features.npz`: Probe features (F1, F2) and metrics
- `results/probes/`: Per-layer probe model weights
- `results/interaction_scores.csv`: All interaction scores
- `results/interaction_scores_correct.csv`: Correct examples only
- `results/interaction_scores_incorrect.csv`: Incorrect examples only

**Plots:**
- `plots/*.png`: All generated visualizations

## Success Criteria

✅ **Experiment is successful if:**

1. Probes decode operands with R² > 0.5 in middle/late layers
2. Multiplication shows noticeably higher interaction than max/min
3. Visible differences between operations across layers
4. Some differences between correct vs incorrect predictions

❌ **Troubleshooting:**

- **Low probe R²:** Model may not represent numbers well (try different difficulty levels)
- **No interaction differences:** Try larger dataset or different operations
- **Model accuracy too low:** Use easier examples or smaller number ranges

## Future Extensions

1. **PID (Partial Information Decomposition):** More rigorous synergy quantification
2. **SAEs (Sparse Autoencoders):** Discover interpretable features automatically
3. **Causal interventions:** Directly manipulate operand features
4. **Attention analysis:** How do attention patterns differ by operation?
5. **Scaling laws:** How do patterns change with model size?

## References

- **Pythia Models:** Biderman et al. (2023) - "Pythia: A Suite for Analyzing Large Language Models"
- **Mechanistic Interpretability:** Elhage et al. (2021) - "A Mathematical Framework for Transformer Circuits"
- **Numeric Reasoning:** Razeghi et al. (2022) - "Impact of Pretraining Term Frequencies on Few-Shot Reasoning"

## License

MIT License - Feel free to use for research and education.

## Citation

If you use this code for research, please cite:

```bibtex
@misc{arithmetic_interaction_2024,
  title={Operand Feature Interaction in Arithmetic Reasoning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/maths_interp}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

**Happy experimenting! 🔬**
