# Robustness checks (exploratory, not part of the main Presentation 2 results)

Planned before running (2026-11-06). All runs are reported; no run is chosen by test score. None of this changes the final model.

1. Repeated cross-validation: 3 new split seeds (101, 202, 303), KimiaNet and UNI2, unchanged settings.
2. Learning curve: 25% and 50% of training cases, per class, on the seed 436 splits, both encoders. Validation and test cases unchanged.
3. Baseline: per-stain mean features plus logistic regression. Regularization chosen on validation cases only.
4. Stain knockout: main best.pth checkpoints scored with stains removed. No retraining.

Differences from the main runs: no attention analysis, and no every-10-updates validation logging (logging does not affect training). Fewer CPUs requested.
Results: results/ and RESULTS_SUMMARY.md in this folder.
