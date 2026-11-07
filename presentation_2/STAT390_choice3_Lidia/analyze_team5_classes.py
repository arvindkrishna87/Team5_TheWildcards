import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(
    "/projects/e32998/STAT390_Fall2026/"
    "Team5_TheWildcards/presentation_1/"
    "Code1_ModelTrainingFixed_23Oct_Lucas/splits_seed436"
)

labels = pd.read_csv(
    "/projects/e32998/STAT390_Krish/Code/Code4_reduce_runtime/extra_benign_case_grade_match.csv"
)
labels["Case"] = labels["Case"].astype(int)

print("Original class frequencies:")
print(labels["Class"].value_counts().sort_index())

rows = []

for split in range(1, 6):
    path = base / f"data_splits_new_{split:02d}.npz"

    with np.load(path, allow_pickle=True) as data:
        groups = {
            "Train": data["train_cases"],
            "Validation": data["val_cases"],
            "Test": data["test_cases"]
        }

    for group, case_ids in groups.items():
        subset = labels[labels["Case"].isin(case_ids)]

        benign = (subset["Class"] == 1).sum()
        high_grade = subset["Class"].isin([3, 4]).sum()
        class_2 = (subset["Class"] == 2).sum()

        rows.append({
            "split": split,
            "group": group,
            "benign": benign,
            "high_grade": high_grade,
            "class_2": class_2,
            "total_cases": len(case_ids),
            "matched_cases": len(subset)
        })

summary = pd.DataFrame(rows)

print("\nCLASS DISTRIBUTION BY SPLIT")
print(summary.to_string(index=False))

summary.to_csv(
    "team5_class_distribution.csv", index=False
)

# Validation class composition
val = summary[summary["group"] == "Validation"]

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(5)

ax.bar(x, val["benign"].to_numpy(), label="Class 1 (benign)")
ax.bar(
    x, 
    val["high_grade"].to_numpy(),
    bottom=val["benign"].to_numpy(),
    label="Classes 3/4 (high-grade)"
)
ax.bar(
    x, val["class_2"].to_numpy(),
    bottom=val["benign"].to_numpy() + val["high_grade"].to_numpy(),
    label="Class 2 (check eligibility)"
)

ax.set_xticks(x)
ax.set_xticklabels(val["split"])
ax.set_xlabel("Split")
ax.set_ylabel("Number of validation cases")
ax.set_title("Team 5: Validation Class Composition")
ax.legend()

plt.tight_layout()
plt.savefig(
    "team5_validation_classes.png", dpi=200
)

print("\nSaved team5_validation_classes.png")
