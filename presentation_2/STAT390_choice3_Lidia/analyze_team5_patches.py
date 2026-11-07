import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

patch_dir = Path("/projects/e32998/patches_benign_split")

split_dir = (
    "/projects/e32998/STAT390_Fall2026/"
    "Team5_TheWildcards/presentation_1/"
    "Code1_ModelTrainingFixed_23Oct_Lucas/splits_seed436"
)

# Count patches for each case and slice/stain combination.
# The filename format is:
# case_038_match_10_h&e_patch100.png

pattern = re.compile(
    r"^case_(\d+)_(match|unmatched)_?(\d+)_(h&e-labels|h&e|melan|sox10)\.?_patch(\d+)\.png$",
    re.IGNORECASE
)

patch_counts = defaultdict(int)
unmatched = 0

print("Scanning patch filenames...")

for filename in os.listdir(patch_dir):
    match = pattern.match(filename)

    if not match:
        unmatched += 1
        continue

    case_id = int(match.group(1))
    match_type = match.group(2).lower()
    slice_id = int(match.group(3))
    stain = match.group(4).lower()

    patch_counts[(case_id, match_type, slice_id)] += 1

print("Matched slice/stain groups:", len(patch_counts))
print("Unmatched filenames:", unmatched)

# Total available patches and capped patches per case
case_available = defaultdict(int)
case_capped = defaultdict(int)
case_slices = defaultdict(int)

for (case_id, match_type, slice_id), count in patch_counts.items():
    case_available[case_id] += count
    case_capped[case_id] += min(count, 800)
    case_slices[case_id] += 1

rows = []

for split in range(1, 6):
    path = Path(split_dir) / f"data_splits_new_{split:02d}.npz"

    with np.load(path, allow_pickle=True) as data:
        validation_cases = [int(x) for x in data["val_cases"]]

    for case_id in validation_cases:
        rows.append({
            "split": split,
            "case": case_id,
            "available_patches": case_available[case_id],
            "capped_patches": case_capped[case_id],
            "slice_stain_groups": case_slices[case_id]
        })

df = pd.DataFrame(rows)

df.to_csv("team5_patch_counts_by_case.csv", index=False)

summary = df.groupby("split").agg(
    validation_cases=("case", "count"),
    total_available_patches=("available_patches", "sum"),
    median_patches_per_case=("available_patches", "median"),
    total_capped_patches=("capped_patches", "sum"),
    median_capped_per_case=("capped_patches", "median"),
    total_slice_stain_groups=("slice_stain_groups", "sum")
).reset_index()

summary.to_csv("team5_patch_summary.csv", index=False)

print("\nPATCH SUMMARY BY VALIDATION SPLIT")
print(summary.round(1).to_string(index=False))

print("\nCASES WITH NO MATCHED PATCHES")
print(df[df["available_patches"] == 0].to_string(index=False))

# Plot distribution of capped patch counts
fig, ax = plt.subplots(figsize=(9, 5))

groups = [
    df.loc[df["split"] == i, "capped_patches"].values
    for i in range(1, 6)
]

ax.boxplot(groups, labels=["1", "2", "3", "4", "5"])

ax.set_xlabel("Validation split")
ax.set_ylabel("Capped patch count per case")
ax.set_title("Team 5: Validation Patch Count Distribution")

plt.tight_layout()
plt.savefig("team5_patch_distribution.png", dpi=200)
plt.close()

print("\nSaved team5_patch_distribution.png")
