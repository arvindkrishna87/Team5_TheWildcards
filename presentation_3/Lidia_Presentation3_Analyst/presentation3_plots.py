import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Team 5 - Presentation 3
# Preliminary dataset: 93 cases, five splits

OUT = "presentation3_figures"
os.makedirs(OUT, exist_ok=True)

# Accuracy results from completed Quest experiments
results = pd.DataFrame({
    "Split": [1, 2, 3, 4, 5],
    "Baseline (512)": [1, 1, .7895, .9474, 1],
    "Embedding 256": [.9444, 1, .8947, .9474, 1],
    "Embedding 128": [.9444, 1, .8947, .9474, .8889],
    "Embedding 64": [.8889, 1, .7368, .9474, 1],
    "Stain dropout": [1, .9474, .8421, .9474, .8889],
    "Slice dropout": [1, 1, .8947, .9474, .9444]
})

models = list(results.columns[1:])
accuracy = results[models] * 100

summary = pd.DataFrame({
    "Model": models,
    "Mean Accuracy (%)": accuracy.mean().values,
    "SD (pp)": accuracy.std(ddof=1).values
})

params = {
    "Baseline (512)": 1182341,
    "Embedding 256": 591749,
    "Embedding 128": 296453,
    "Embedding 64": np.nan,
    "Stain dropout": 1182341,
    "Slice dropout": 1182341
}

summary["Parameters"] = summary["Model"].map(params)
summary.to_csv(OUT + "/model_summary.csv", index=False)
results.to_csv(OUT + "/accuracy_by_split.csv", index=False)

print("\nFIVE-SPLIT RESULTS")
print(summary.round(2).to_string(index=False))

# Plot 1: Embedding dimension comparison
fig, ax = plt.subplots(figsize=(9, 5))
names = models[:4]
means = summary["Mean Accuracy (%)"].iloc[:4]
sds = summary["SD (pp)"].iloc[:4]

bars = ax.bar(
    ["512", "256", "128", "64"],
    means,
    yerr=sds,
    capsize=5,
    color=["#64748b", "#0d9488", "#60a5fa", "#94a3b8"]
)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
            f"{val:.1f}%", ha="center", fontsize=11)

ax.set_title("Embedding Dimension vs. Test Accuracy")
ax.set_xlabel("Embedding Dimension")
ax.set_ylabel("Mean Test Accuracy (%)")
ax.set_ylim(0, 115)
ax.grid(axis="y", alpha=.2)
plt.tight_layout()
plt.savefig(OUT + "/01_embedding_accuracy.png", dpi=300)
plt.close()

# Plot 2: Dropout comparison
indices = [0, 4, 5]
means = summary["Mean Accuracy (%)"].iloc[indices]
sds = summary["SD (pp)"].iloc[indices]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    ["No dropout", "Stain dropout", "Slice dropout"],
    means,
    yerr=sds,
    capsize=5,
    color=["#64748b", "#f59e0b", "#0d9488"]
)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
            f"{val:.1f}%", ha="center", fontsize=11)

ax.set_title("Effect of Training-Time Dropout")
ax.set_ylabel("Mean Test Accuracy (%)")
ax.set_ylim(0, 115)
ax.grid(axis="y", alpha=.2)
plt.tight_layout()
plt.savefig(OUT + "/02_dropout_accuracy.png", dpi=300)
plt.close()

# Plot 3: Accuracy across five splits
fig, ax = plt.subplots(figsize=(11, 6))
colors = [
    "#334155", "#0d9488", "#2563eb",
    "#9333ea", "#f59e0b", "#e11d48"
]

for model, color in zip(models, colors):
    ax.plot(
        results["Split"],
        results[model] * 100,
        marker="o",
        linewidth=2,
        label=model,
        color=color
    )

ax.set_title("Test Accuracy Across Five Splits")
ax.set_xlabel("Split")
ax.set_ylabel("Test Accuracy (%)")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_ylim(65, 105)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(alpha=.25)
plt.tight_layout()
plt.savefig(
    OUT + "/03_accuracy_across_splits.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Plot 4: Heatmap (matplotlib only)
fig, ax = plt.subplots(figsize=(10, 5))
matrix = accuracy.T.values
image = ax.imshow(
    matrix, cmap="YlGnBu",
    vmin=70, vmax=100, aspect="auto"
)

ax.set_xticks(range(5))
ax.set_xticklabels(["1", "2", "3", "4", "5"])
ax.set_yticks(range(6))
ax.set_yticklabels(models)

for i in range(6):
    for j in range(5):
        val = matrix[i, j]
        ax.text(
            j, i, f"{val:.1f}%",
            ha="center", va="center",
            color="white" if val < 85 else "black",
            fontsize=10
        )

ax.set_title("Test Accuracy by Model and Split")
ax.set_xlabel("Split")
fig.colorbar(image, ax=ax, label="Accuracy (%)")
plt.tight_layout()
plt.savefig(OUT + "/04_accuracy_heatmap.png", dpi=300)
plt.close()

print("\nSUCCESS! Files saved in:", OUT)
