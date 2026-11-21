from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path.home() / "STAT390_P3_Analyst2"
BASE = Path("/projects/e32998/MIL_training/final_runs")

RUNS = [
    "run_20261119_115429_786671_2230960",
    "run_20261119_121657_761730_3558450",
]

test_sets = []
split_sets = {}

# Load our 50 previously validated test splits
for rep in range(1, 11):
    for fold in range(1, 6):
        path = (
            ROOT / "analyst2_splits"
            / f"rep{rep:02d}"
            / f"data_splits_new_{fold:02d}.npz"
        )
        with np.load(path) as data:
            split_sets[(rep, fold)] = set(
                map(int, data["test_cases"])
            )

for name in RUNS:
    folder = BASE / name
    csv_path = folder / "predictions_best.csv"

    df = pd.read_csv(csv_path)
    df["case_id"] = df["case_id"].astype(int)

    test_cases = set(df["case_id"])
    test_sets.append(test_cases)

    incorrect = df[
        df["true_label"] != df["predicted_label"]
    ]

    print("\n" + "=" * 55)
    print("RUN:", name)
    print("Test cases:", sorted(test_cases))
    print("Correct:", len(df) - len(incorrect))
    print("Incorrect:", len(incorrect))

    print("\nMISCLASSIFIED CASES:")
    if incorrect.empty:
        print("None")
    else:
        print(
            incorrect[
                [
                    "case_id",
                    "true_label",
                    "predicted_label",
                    "prob_class0",
                    "prob_class1",
                ]
            ].to_string(index=False)
        )

    matching_splits = [
        f"Repetition {rep}, Fold {fold}"
        for (rep, fold), cases in split_sets.items()
        if cases == test_cases
    ]

    print("\nMatches our validated splits:")
    print(matching_splits if matching_splits else "No exact match")

    checkpoint = torch.load(
        folder / "checkpoints" / "best.pth",
        map_location="cpu",
        weights_only=True,
    )

    print("\nCheckpoint architecture:")
    print(checkpoint.get("arch", "Not recorded"))

    print("\nCheckpoint configuration:")
    print(str(checkpoint.get("config", "Not recorded"))[:2000])

    attention_dir = folder / "attention_analysis"
    if attention_dir.exists():
        print("\nAttention output files:")
        for path in sorted(attention_dir.iterdir())[:10]:
            print(" ", path.name)

print("\n" + "=" * 55)
print("COMPARING THE TWO RUNS")
print("Shared test cases:", sorted(test_sets[0] & test_sets[1]))
print("Total unique cases tested:", len(test_sets[0] | test_sets[1]))