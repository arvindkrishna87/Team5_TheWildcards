from pathlib import Path
import pandas as pd

base = Path.home() / "STAT390_Presentation3"
runs = base / "training_runs"

experiments = {
    "Baseline (512)": "run_20261120_044034_987361_3075354",
    "Embedding 256": "run_20261120_044034_987460_3075353",
    "Embedding 128": "run_20261120_044034_979358_3075352",
    "Embedding 64": "run_20261120_045659_838684_2332852",
    "Stain dropout (30%)": "run_20261120_045659_835168_2332853",
    "Slice dropout (30%)": "run_20261120_045724_801088_3144269",
}

# Label convention: 0 = benign, 1 = high-grade
results = []
all_predictions = []

for name, folder in experiments.items():
    path = runs / folder / "predictions_best.csv"
    df = pd.read_csv(path)

    y_true = df["true_label"].astype(int)
    y_pred = df["predicted_label"].astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    accuracy = (tn + tp) / len(df)
    benign_recall = tn / (tn + fp) if tn + fp else float("nan")
    high_grade_recall = tp / (tp + fn) if tp + fn else float("nan")

    results.append({
        "experiment": name,
        "n_test": len(df),
        "accuracy": accuracy,
        "benign_recall": benign_recall,
        "high_grade_recall": high_grade_recall,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    })

    df["experiment"] = name
    all_predictions.append(df)

summary = pd.DataFrame(results)
predictions = pd.concat(all_predictions, ignore_index=True)

output_dir = base / "presentation3_analysis"
output_dir.mkdir(exist_ok=True)

summary.to_csv(output_dir / "experiment_summary.csv", index=False)
predictions.to_csv(output_dir / "all_predictions.csv", index=False)

print("\nPRESENTATION 3 EXPERIMENT SUMMARY")
print("=" * 95)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nFILES SAVED:")
print(output_dir / "experiment_summary.csv")
print(output_dir / "all_predictions.csv")
