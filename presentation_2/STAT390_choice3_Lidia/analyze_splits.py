import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(
    "/projects/e32998/STAT390_Fall2026/"
    "Team2_TheConsultancy/Presentation_1/"
    "Alyssa_Presentation1_base_code/"
    "no_train_val_leakage_splits"
)

rows = []
assignments = {}

for i in range(1, 6):
    path = base / f"data_splits_new_{i:02d}.npz"

    with np.load(path, allow_pickle=True) as data:
        train = set(map(int, data["train_cases"]))
        val = set(map(int, data["val_cases"]))
        test = set(map(int, data["test_cases"]))

    assignments[i] = {
        "train": train,
        "val": val,
        "test": test
    }

    rows.append({
        "split": i,
        "train_cases": len(train),
        "val_cases": len(val),
        "test_cases": len(test),
        "total_cases": len(train | val | test),
        "overlap_train_val": len(train & val),
        "overlap_train_test": len(train & test),
        "overlap_val_test": len(val & test)
    })

summary = pd.DataFrame(rows)

print("\nCASE COUNTS BY SPLIT")
print(summary.to_string(index=False))

summary.to_csv(
    "team2_split_counts.csv", index=False
)

# Compare validation case membership with Split 5
reference = assignments[5]["val"]

comparison = []

for i in range(1, 6):
    current = assignments[i]["val"]

    comparison.append({
        "split": i,
        "shared_with_split5": len(current & reference),
        "different_from_split5": len(current - reference),
        "validation_case_ids": sorted(current)
    })

comparison_df = pd.DataFrame(comparison)

print("\nVALIDATION CASE COMPARISON")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    "team2_validation_comparison.csv", index=False
)

# Visualization
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(summary))
width = 0.25

ax.bar(
    x - width, summary["train_cases"],
    width, label="Training"
)
ax.bar(
    x, summary["val_cases"],
    width, label="Validation"
)
ax.bar(
    x + width, summary["test_cases"],
    width, label="Testing"
)

ax.set_xticks(x)
ax.set_xticklabels(summary["split"])
ax.set_xlabel("Split")
ax.set_ylabel("Number of cases")
ax.set_title("Team 2: Dataset Composition by Split")
ax.legend()

plt.tight_layout()
plt.savefig(
    "team2_split_composition.png", dpi=200
)

print("\nSaved team2_split_composition.png")
