"""
STAT 390 Presentation 3
Analyst Choice 2: Repeated cross-validation validation.

Reads the actual saved split files and produces
a printed results summary, CSV, and figure.
"""

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "analyst2_splits"
OUTPUT = ROOT / "analyst2_results"
OUTPUT.mkdir(exist_ok=True)

GROUPED_CASES = [
    (22, 107, 108),
    (24, 118),
    (25, 119),
    (26, 109, 110, 111),
]

rows = []
partitions = []
case_universe = None

for rep in range(1, 11):
    test_counts = Counter()
    fold_groups = []

    for fold in range(1, 6):
        path = (
            SPLITS / f"rep{rep:02d}"
            / f"data_splits_new_{fold:02d}.npz"
        )

        with np.load(path) as data:
            train = set(map(int, data["train_cases"]))
            val = set(map(int, data["val_cases"]))
            test = set(map(int, data["test_cases"]))

        assert not (train & val)
        assert not (train & test)
        assert not (val & test)

        all_cases = train | val | test

        if case_universe is None:
            case_universe = all_cases

        assert all_cases == case_universe
        assert len(all_cases) == 99

        for group in GROUPED_CASES:
            locations = sum(
                all(c in subset for c in group)
                for subset in (train, val, test)
            )
            assert locations == 1, (
                f"Patient group split: {group}"
            )

        rows.append({
            "repetition": rep,
            "fold": fold,
            "train_cases": len(train),
            "validation_cases": len(val),
            "test_cases": len(test),
        })

        test_counts.update(test)
        fold_groups.append(frozenset(test))

    assert len(test_counts) == 99
    assert all(n == 1 for n in test_counts.values())

    partitions.append(frozenset(fold_groups))

assert len(set(partitions)) == 10

df = pd.DataFrame(rows)

print("\nSTAT 390 — ANALYST CHOICE 2")
print("=" * 45)
print("Eligible cases:", len(case_universe))
print("Repetitions:", 10)
print("Folds per repetition:", 5)
print("Total validated splits:", len(df))
print("Unique five-fold partitions:", len(set(partitions)))
print("Planned held-out predictions:", len(case_universe) * 10)
print("Patient grouping preserved: YES")
print("Each case tested once per repetition: YES")

print("\nTEST FOLD SIZES")
print("-" * 45)

matrix = df.pivot(
    index="repetition",
    columns="fold",
    values="test_cases"
)

print(matrix.to_string())

# Save the actual numerical results
csv_path = OUTPUT / "split_validation.csv"
df.to_csv(csv_path, index=False)

# Plot actual test-set sizes
fig, ax = plt.subplots(figsize=(9, 5))

values = matrix.to_numpy()

ax.imshow(
    values,
    cmap="Blues",
    vmin=18,
    vmax=20,
    aspect="auto"
)

for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        ax.text(
            j, i, str(values[i, j]),
            ha="center", va="center",
            fontsize=11, color="black"
        )

ax.set_xticks(range(5))
ax.set_xticklabels([f"Fold {i}" for i in range(1, 6)])

ax.set_yticks(range(10))
ax.set_yticklabels(range(1, 11))

ax.set_ylabel("Repetition")
ax.set_title(
    "Held-Out Test Cases Across 10 Repetitions",
    fontsize=14
)

fig.tight_layout()

figure_path = OUTPUT / "split_validation.png"
fig.savefig(figure_path, dpi=200)
plt.close(fig)

print("\nFiles saved:")
print(csv_path)
print(figure_path)
print("\nALL 50 SPLITS VALIDATED")
