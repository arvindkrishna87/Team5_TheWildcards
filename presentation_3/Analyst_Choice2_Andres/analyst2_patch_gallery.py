from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

RUN = Path(
    "/projects/e32998/MIL_training/final_runs/"
    "run_20261119_115429_786671_2230960"
)

SOURCE = (
    RUN / "attention_analysis"
    / "top_effective_patches_per_case_5.0pct.csv"
)

OUTPUT = (
    Path.home()
    / "STAT390_P3_Analyst2"
    / "analyst2_results"
    / "case46_vs22_patch_gallery.png"
)

df = pd.read_csv(SOURCE)

fig, axes = plt.subplots(
    2, 3, figsize=(14, 8),
    layout="constrained"
)

cases = [
    (46, "Misclassified: Benign → High-grade"),
    (22, "Correct: Benign → Benign"),
]

for row, (case_id, description) in enumerate(cases):
    selected = (
        df[df["case_id"] == case_id]
        .sort_values("effective_weight", ascending=False)
        .head(3)
    )

    for col, (_, patch) in enumerate(selected.iterrows()):
        ax = axes[row, col]
        path = Path(patch["patch_path"])

        if path.exists():
            with Image.open(path) as img:
                ax.imshow(img.convert("RGB"))
        else:
            ax.text(
                0.5, 0.5, "IMAGE NOT FOUND",
                ha="center", va="center",
                transform=ax.transAxes
            )
            print("Missing:", path)

        patch_number = (
            path.stem.split("patch")[-1]
        )

        attention = patch["effective_weight"] * 100

        ax.set_title(
            f"Case {case_id} | {patch['stain'].upper()} "
            f"Patch {patch_number}\n"
            f"Effective attention: {attention:.2f}%",
            fontsize=11
        )
        ax.axis("off")

fig.suptitle(
    "Highest-attention tissue patches\n"
    "Case 46 (incorrect) vs. Case 22 (correct)",
    fontsize=17,
    fontweight="bold"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
plt.close(fig)

print("Created:", OUTPUT)
print("Source:", SOURCE)
print("Note: One trained checkpoint, not repeated-error results.")