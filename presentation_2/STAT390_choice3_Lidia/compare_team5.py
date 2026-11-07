import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

split_path = (
    "/projects/e32998/STAT390_Fall2026/"
    "Team5_TheWildcards/presentation_1/"
    "Code1_ModelTrainingFixed_23Oct_Lucas/"
    "splits_seed436/data_splits_new_04.npz"
)

labels_path = (
    "/projects/e32998/STAT390_Krish/Code/"
    "Code4_reduce_runtime/extra_benign_case_grade_match.csv"
)

patch_dir = "/projects/e32998/patches_benign_split"

data = np.load(split_path)
train = set(map(int, data["train_cases"]))
val = set(map(int, data["val_cases"]))
test = set(map(int, data["test_cases"]))

print("TEAM 5 — SPLIT 4")
print("Training cases:", len(train))
print("Validation cases:", len(val))
print("Test cases:", len(test))
print("Train/validation overlap:", len(train & val))

labels = pd.read_csv(labels_path)
print("\nLabel CSV columns:", list(labels.columns))

# Match the model's filename grouping convention.
pattern = re.compile(
    r"^case_(\d+)_(match|unmatched)_?(\d+)_.*\.png$",
    re.IGNORECASE
)

slice_counts = defaultdict(int)
unmatched = 0

for filename in os.listdir(patch_dir):
    m = pattern.match(filename)
    if m is None:
        unmatched += 1
        continue

    case_id = int(m.group(1))
    slice_id = m.group(2).lower() + "_" + m.group(3)
    slice_counts[(case_id, slice_id)] += 1

case_stats = []

for case_id in sorted(val):
    groups = [
        count for (cid, sid), count in slice_counts.items()
        if cid == case_id
    ]

    case_stats.append({
        "case": case_id,
        "slice_groups": len(groups),
        "available_patches": sum(groups),
        "capped_patches": sum(min(n, 800) for n in groups)
    })

df = pd.DataFrame(case_stats)

print("\nVALIDATION PATCH SUMMARY")
print("Unmatched filenames:", unmatched)
print("Validation slice groups:", df["slice_groups"].sum())
print("Total available patches:", df["available_patches"].sum())
print("Total capped patches:", df["capped_patches"].sum())
print("Median capped patches/case:", df["capped_patches"].median())
print("\nIndividual validation cases:")
print(df.to_string(index=False))

df.to_csv("team5_split4_patch_details.csv", index=False)
