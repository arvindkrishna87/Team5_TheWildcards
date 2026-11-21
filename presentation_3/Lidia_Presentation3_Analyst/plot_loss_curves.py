import os
import re
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "presentation3_figures"
os.makedirs(OUT, exist_ok=True)

NAMES = [
    "Baseline (512)",
    "Embedding 256",
    "Embedding 128",
    "Embedding 64",
    "Stain dropout",
    "Slice dropout"
]

COLORS = [
    "#334155", "#0d9488", "#2563eb",
    "#9333ea", "#f59e0b", "#e11d48"
]

records = []

# Tasks 0-23 correspond to splits 2-5,
# with six configurations per split.
for task in range(24):
    split = task // 6 + 2
    model = NAMES[task % 6]

    path = f"logs/cv_9515161_{task}.out"

    if not os.path.exists(path):
        print("Missing:", path)
        continue

    with open(path, "r", errors="replace") as f:
        content = f.read()

    # Each epoch header is followed by its training/validation
    # loss summary. The non-greedy search avoids crossing epochs.
    pattern = (
        r"Epoch\s+(\d+)/\d+.*?"
        r"Train Loss:\s*([0-9.eE+-]+).*?"
        r"Val Loss:\s*([0-9.eE+-]+)"
    )

    matches = re.findall(pattern, content, flags=re.S)

    for epoch, train_loss, val_loss in matches:
        records.append({
            "Split": split,
            "Model": model,
            "Epoch": int(epoch),
            "Train Loss": float(train_loss),
            "Val Loss": float(val_loss)
        })

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(
        "No epoch loss values found. Check log format."
    )

df = df.drop_duplicates(
    subset=["Split", "Model", "Epoch"],
    keep="last"
).sort_values(["Split", "Model", "Epoch"])

df.to_csv(OUT + "/loss_curves.csv", index=False)

print("Epoch records extracted:", len(df))
print(df.groupby(["Split", "Model"]).size().to_string())

# ---------------------------------------------------
# FIGURE 1: Baseline vs dropout, splits 2-5
# ---------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
selected = [
    "Baseline (512)",
    "Stain dropout",
    "Slice dropout"
]

selected_colors = {
    "Baseline (512)": "#334155",
    "Stain dropout": "#f59e0b",
    "Slice dropout": "#0d9488"
}

for ax, split in zip(axes.flat, range(2, 6)):
    for model in selected:
        subset = df[
            (df["Split"] == split) &
            (df["Model"] == model)
        ]

        if subset.empty:
            continue

        color = selected_colors[model]

        ax.plot(
            subset["Epoch"],
            subset["Train Loss"],
            color=color,
            linestyle="-",
            linewidth=2,
            label=model + " train"
        )

        ax.plot(
            subset["Epoch"],
            subset["Val Loss"],
            color=color,
            linestyle="--",
            linewidth=2,
            label=model + " val"
        )

    ax.set_title(f"Split {split}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)

handles, labels = axes.flat[0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="lower center",
    ncol=3,
    fontsize=10
)

fig.suptitle(
    "Training vs. Validation Loss: Dropout Experiments",
    fontsize=16
)
fig.tight_layout(rect=[0, 0.10, 1, 0.95])

fig.savefig(
    OUT + "/05_dropout_loss_curves.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# ---------------------------------------------------
# FIGURE 2: Baseline vs embedding sizes, splits 2-5
# ---------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
selected = NAMES[:4]

for ax, split in zip(axes.flat, range(2, 6)):
    for model in selected:
        subset = df[
            (df["Split"] == split) &
            (df["Model"] == model)
        ]

        if subset.empty:
            continue

        color = COLORS[NAMES.index(model)]

        ax.plot(
            subset["Epoch"],
            subset["Train Loss"],
            color=color,
            linestyle="-",
            linewidth=2,
            label=model + " train"
        )

        ax.plot(
            subset["Epoch"],
            subset["Val Loss"],
            color=color,
            linestyle="--",
            linewidth=2,
            label=model + " val"
        )

    ax.set_title(f"Split {split}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)

handles, labels = axes.flat[0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="lower center",
    ncol=4,
    fontsize=9
)

fig.suptitle(
    "Training vs. Validation Loss: Embedding Size",
    fontsize=16
)
fig.tight_layout(rect=[0, 0.10, 1, 0.95])

fig.savefig(
    OUT + "/06_embedding_loss_curves.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("\nSUCCESS! Saved loss curves in:", OUT)
