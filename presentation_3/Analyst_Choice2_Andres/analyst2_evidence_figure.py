"""STAT 390 P3 Analyst Choice 2 — plot measured signed evidence.

Run in the same QUEST folder as analyst2_case_evidence.py, after that script
has created analyst2_results/case46_vs22_top_attention_evidence.csv.

This produces one model-checkpoint comparison, not repeat-level error rates.
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "analyst2_results"
SOURCE = RESULTS / "case46_vs22_top_attention_evidence.csv"
OUTPUT = RESULTS / "case46_vs22_signed_evidence.png"

if not SOURCE.is_file():
    raise FileNotFoundError(f"Run analyst2_case_evidence.py first: {SOURCE}")

df = pd.read_csv(SOURCE)
needed = {"case_id_for_report", "rank", "patch_file", "stain", "effective_weight", "signed_evidence"}
if not needed.issubset(df.columns):
    raise ValueError(f"Missing CSV columns: {sorted(needed - set(df.columns))}")

case_specs = [
    (46, "Case 46 · Benign → High-grade (incorrect)", "+3.1425 overall margin"),
    (22, "Case 22 · Benign → Benign (correct)", "−5.8182 overall margin"),
]
colors = {"high-grade": "#BF5A42", "benign": "#33658A"}
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7))
for ax, (case_id, title, margin_note) in zip(axes, case_specs):
    part = df.loc[df["case_id_for_report"].astype(int) == case_id].sort_values("rank")
    if len(part) != 5:
        raise ValueError(f"Expected exactly five rows for Case {case_id}; found {len(part)}")
    evidence = part["signed_evidence"].astype(float).tolist()
    attention = part["effective_weight"].astype(float).tolist()
    labels = []
    barcolors = []
    for _, r in part.iterrows():
        m = re.search(r"_patch(\d+)\.", str(r["patch_file"]))
        patchnum = m.group(1) if m else str(r["patch_file"])
        labels.append(f"{str(r['stain']).upper()}  ·  patch {patchnum}")
        barcolors.append(colors["high-grade"] if float(r["signed_evidence"]) > 0 else colors["benign"])
    yy = range(5)
    ax.barh(yy, evidence, color=barcolors, height=0.63)
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_yticks(list(yy), labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=15)
    ax.text(0, 1.01, margin_note, ha="left", va="bottom", transform=ax.transAxes, fontsize=9, color="#555555")
    ax.set_xlabel("Signed evidence  (positive → high-grade; negative → benign)", fontsize=9)
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    bound = max(max(abs(v) for v in evidence), 0.01)
    ax.set_xlim(-bound*1.43 if min(evidence)<0 else -bound*0.15, bound*1.43)
    for j, (e, a) in enumerate(zip(evidence, attention)):
        ax.annotate(f"{e:+.3f}   |   {a*100:.2f}% attention", xy=(e, j),
                    xytext=(5 if e>=0 else -5,0), textcoords="offset points",
                    ha="left" if e>=0 else "right", va="center", fontsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

fig.suptitle("Highest-attention patches: direction of evidence", fontsize=16, weight="bold", y=1.02)
fig.legend(handles=[Patch(color=colors["high-grade"], label="Supports high-grade"),
                    Patch(color=colors["benign"], label="Supports benign")],
           loc="lower center", bbox_to_anchor=(0.5, -0.065), ncol=2, frameon=False)
fig.text(0.5, -0.02, "One checkpoint · top five selected by effective attention · case-level comparisons, not repeated-error frequencies",
         ha="center", va="center", fontsize=9, color="#555555")
fig.subplots_adjust(left=0.14, right=0.97, top=0.83, bottom=0.23, wspace=0.36)
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Created: {OUTPUT}")
print("Source: case46_vs22_top_attention_evidence.csv")
print("Caution: one saved checkpoint; not 10-repeat analysis")