# Analyst Choice 2 — Class-Specific Patch Evidence

## Objective

Identify influential image patches and determine whether each patch favors Class 0 (benign) or Class 1 (combined non-benign categories).

## Method

Implemented **class-specific logit decomposition** as an alternative to Grad-CAM and iterative patch removal.

For each patch, the evidence contribution is:

\[
E_p = a_p(W_1-W_0)^T h_p
\]

Where:

- \(E_p\): Signed class-specific evidence for patch \(p\)
- \(a_p\): Effective attention weight (stain × slice × patch attention)
- \(W_1-W_0\): Difference between the two classifier weight vectors
- \(h_p\): Projected 512-dimensional patch representation

Positive scores favor Class 1; negative scores favor benign. Patches are ranked by the absolute magnitude of their evidence scores.

## Model and Data

Used the existing trained KimiaNet-based hierarchical attention MIL model. No retraining, architecture modifications, or hyperparameter tuning were performed.

**Checkpoint:**

`/projects/e32998/MIL_training/final_runs/run_20261105_110826_kimianet_split01/checkpoints/best.pth`

**Embeddings:**

`/projects/e32998/patches_varsize_pooled4096`

The scripts reuse the existing `models.py`, `dataset.py`, and `data_utils.py` from the professor-provided Presentation 2 baseline code.

## Files

- `class_specific_evidence.py` — Initial implementation, synthetic validation, and single-case testing.
- `class_specific_evidence_all17.py` — Analysis of all 17 saved test cases, validation, CSV summaries, and evidence visualizations.

## Running the Analysis

Run on QUEST with the required Python dependencies installed. The scripts should be run alongside the professor's baseline modules, or with that directory available on the Python import path.

```bash
python class_specific_evidence_all17.py
```

Results are generated in `choice2_results/`.

## Results

Analyzed all 17 cases in the existing saved test evaluation.

- 13/17 cases correctly classified (76.47%).
- 17/17 predicted class labels reproduced.
- 14/17 saved Class 1 probabilities reproduced within 0.001.
- Maximum logit reconstruction error: approximately 4.57 × 10⁻⁷.
- The strongest five patches all favored the predicted class in 10 cases, opposed it in 5 cases, and showed mixed directions in 2 cases.
- All four incorrectly classified cases had their strongest five patches favoring the opposite class.

## Limitations

The method decomposes the model's existing class-score difference. It does not measure the causal effect of removing patches.

Patch indices have not been verified against original image coordinates. Three cases showed probability differences relative to the saved evaluation.

The model is binary: original label 1 maps to Class 0, while original labels 3 and 4 are combined into Class 1. The method cannot distinguish those two non-benign categories.

These results are exploratory and do not establish clinical reliability.