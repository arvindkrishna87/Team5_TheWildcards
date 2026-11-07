# Coder Option 1 Findings

This folder compares frozen UNI2 and KimiaNet patch features for benign vs.
high-grade prediction, using the Presentation 2 base code.

## Main result

Results combine all 93 test cases across 5 splits (seed 436).

| Encoder | Accuracy | Benign recall | High-grade recall |
|---|---|---|---|
| KimiaNet | 0.817 | 0.857 | 0.793 |
| UNI2 | 0.957 | 0.971 | 0.948 |

UNI2 alone was right on 14 cases and KimiaNet alone on 1 (exact McNemar p = 0.001).

This is a fair comparison on the same splits, but it is not a leak-free estimate.
The model settings came from the earlier tuning, and they were not retuned for UNI2.

## Folder contents

| Item | Contents |
|---|---|
| `Code1_UNI2vsKimiaNet_6Nov_Lucas/` | Pipeline, job scripts, and the 5 split files |
| `Code2_Metrics_6Nov_Lucas.py` | Metrics, required plots, and McNemar test |
| `Code3_Ideas_6Nov_Lucas/` | Code for the exploratory checks |
| `Results_6Nov_Lucas/` | Result tables, plots, exploratory results, and job logs |

## Changes from the base code

- `precompute_pooled_features.py`: adds frozen UNI2 embeddings (1,536 features).
  Each whole patch is resized to 224×224 with no center crop, then normalized.
  The job can be split across several GPUs.
- `main.py`: adds `--embeddings_dir`, `--pooled_dim`, and `--val_every_updates`.
- `trainer.py`: logs train and validation loss per gradient update and per epoch.
  Interim validation is for logging only and does not affect training.
- `utils.py`: adds the SLURM job ID to run folder names, so jobs starting in the
  same second do not share a folder.
- `config.py`: saves results to a personal folder.

## How the runs were done

1. Created the 5 splits with `make_splits.py --seed 436`.
2. Created UNI2 embeddings with `embed_uni2.sbatch`.
3. Trained each split with `train_split.sbatch <split> <kimianet|uni2>`.
4. Created the metrics and plots with `Code2_Metrics_6Nov_Lucas.py`.

Paths point to my Quest folders, so they need updating to rerun.
`KimiaNet.pth`, embeddings, UNI2 weights, and checkpoints are not included.

## Exploratory checks

These do not change the final model, and no run was chosen by test score.

- Repeated splits: UNI2 averaged 0.933 and KimiaNet 0.817 across 4 split seeds.
- Same input: UNI2 stayed ahead at 224×224 (0.957 vs. 0.774) and at original size
  (0.914 vs. 0.817). Original-size UNI2 patches were edge-padded.
- Also included: a learning curve, a logistic regression baseline, and a stain
  knockout test.

The scripts in `Code3_Ideas_6Nov_Lucas/` show what was run, but their paths must be
set up before they can be rerun.
