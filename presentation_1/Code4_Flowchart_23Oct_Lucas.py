#!/usr/bin/env python3
"""Draw the Stage 3 + Stage 4 algorithm flowchart, color-coded by script.

Usage:  python make_flowchart.py   ->  results_summary/flowchart.png and .svg
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.expanduser("~/STAT390_coder/results_summary")

# fill, border per script
COLORS = {
    "config.py":             ("#E8E8E8", "#6B6B6B"),
    "data_utils.py":         ("#D6E6F7", "#2F6DB5"),
    "make_splits.py":        ("#E6DDF4", "#6A4BA8"),
    "utils.py":              ("#D5EFEC", "#24857A"),
    "main.py":               ("#FBE2CC", "#C4661F"),
    "dataset.py":            ("#DCEFD6", "#3D8A2E"),
    "models.py":             ("#F7D9D9", "#B53A3A"),
    "trainer.py":            ("#FAF0C8", "#A88411"),
    "attention_analysis.py": ("#F6DDEC", "#A8417C"),
    "analyze_results.py":    ("#EADFD3", "#7A5636"),
}

fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

W = 4.95        # box width
WRAP = 68       # characters per line of body text
LINE = 0.152    # height of one body line
GAP = 0.17      # vertical gap between boxes


def box(x, y, title, detail, script, script_label=None):
    """Box centred at x with its top edge at y; height fits the wrapped text. Returns (top, bottom)."""
    fill, edge = COLORS[script]
    lines = textwrap.wrap(detail, WRAP)
    h = 0.36 + LINE * len(lines) + 0.06
    ax.add_patch(FancyBboxPatch((x - W / 2, y - h), W, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.6))
    ax.text(x - W / 2 + 0.12, y - 0.08, title, ha="left", va="top", fontsize=9.3, weight="bold", color="#1a1a1a")
    ax.text(x + W / 2 - 0.1, y - 0.09, script_label or script, ha="right", va="top", fontsize=7.4,
            family="monospace", color=edge, weight="bold")
    for i, ln in enumerate(lines):
        ax.text(x - W / 2 + 0.12, y - 0.33 - i * LINE, ln, ha="left", va="top", fontsize=7.6, color="#333333")
    return y, y - h


def arrow(x1, y1, x2, y2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.3,
                                 color="#444444", ls=ls))


def elbow(x1, y1, xmid, y2, x2, ls="-"):
    """Right-angle connector routed through the gap between columns."""
    ax.plot([x1, xmid, xmid], [y1, y1, y2], color="#444444", lw=1.3, ls=ls, solid_capstyle="round")
    arrow(xmid, y2, x2, y2, ls=ls)


def column(x, top, items):
    spans = []
    y = top
    for it in items:
        t, b = box(x, y, **it)
        if spans:
            arrow(x, spans[-1][1], x, t)
        spans.append((t, b))
        y = b - GAP
    return spans


# ---- Title and upstream strip ----
ax.text(0.25, 8.72, "Coder Flowchart: Stages 3 & 4", fontsize=16, weight="bold", va="center")
ax.add_patch(FancyBboxPatch((0.2, 7.98), 15.6, 0.44, boxstyle="round,pad=0,rounding_size=0.06",
                            fc="#F4F4F4", ec="#AAAAAA", lw=1, ls="--"))
ax.text(8.0, 8.2, "Done before this:  Stage 1 cut each tissue slice into patch images, and Stage 2 turned each "
                  "patch into a 4096-number feature vector with KimiaNet (saved as one .pt file per patch).",
        fontsize=8.6, va="center", ha="center", color="#555")

xA, xB, xC = 2.7, 8.0, 13.3
for x, head, sub in [(xA, "STAGE 3 · make splits", "make_splits.py  (run once)"),
                     (xB, "STAGE 4 · set up", "main.py  (one SLURM job per split)"),
                     (xC, "STAGE 4 · train & evaluate", "main.py  (continued)")]:
    ax.text(x, 7.72, head, ha="center", fontsize=11.5, weight="bold", color="#222")
    ax.text(x, 7.47, sub, ha="center", fontsize=8.5, color="#666", family="monospace")

TOP = 7.27

# ---- Column A: Stage 3 ----
spansA = column(xA, TOP, [
    dict(title="1  load_labels · get_all_patch_files", script="data_utils.py",
         detail="Reads the label file, which gives every case its diagnosis class, and lists every patch image in "
                "the patch folder. The list is sorted so the order is identical on every run."),
    dict(title="2  group_patches_by_slice", script="data_utils.py",
         detail="Reads each filename to find its case, slice and stain, and groups the patches by slice. Each case "
                "then gets a label: class 1 becomes benign (0), and classes 3 and 4 become high-grade (1)."),
    dict(title="3  split_by_case_with_constraints", script="make_splits.py",
         detail="Divides the 93 cases into 5 folds of 18–19 cases, keeping the benign / high-grade mix about the "
                "same in every fold. Related pseudo-cases from one patient (e.g. 22, 107 and 108) always stay in the "
                "same fold. The seed (436) controls how cases are shuffled."),
    dict(title="4  rotate folds × 5", script="make_splits.py", script_label="make_splits.py + data_utils.py",
         detail="Builds 5 splits by rotating the folds: 3 folds train, 1 validates and 1 tests, so every case is "
                "tested exactly once. A leak check confirms that no case or patch appears in two sets."),
    dict(title="5  save_data_splits", script="utils.py",
         detail="Saves the case IDs of each split to its own file (data_splits_new_01 to 05.npz). Each Stage 4 job "
                "reads one of these files."),
])

# ---- Column B: Stage 4 setup ----
spansB = column(xB, TOP, [
    dict(title="1  parse_args", script="config.py", script_label="main.py + config.py",
         detail="Reads the run settings. Defaults come from config.py, and the job script adds the split file, the "
                "seed, and caps of 500 patches per slice and 5 slices per stain."),
    dict(title="2  set_seed · create_run_directory", script="utils.py",
         detail="Fixes the random seed so the run can be repeated exactly. Creates a results folder named by the "
                "start time and the SLURM job ID."),
    dict(title="3  prepare_data", script="data_utils.py", script_label="main.py → data_utils.py, utils.py",
         detail="Repeats the filename-to-label steps from Stage 3 and loads the chosen split file. Builds the "
                "train, validation and test sets of cases and checks again that they do not overlap."),
    dict(title="4  create_data_loaders", script="dataset.py", script_label="main.py → dataset.py",
         detail="Each item given to the model is one whole case, with its patches grouped by stain and slice. "
                "Each patch's feature file is read from disk, up to 500 patches per slice and 5 slices per stain. "
                "Training cases get a new random set of patches every epoch; validation and test cases always use "
                "the same ones."),
    dict(title="5  create_model  (HierarchicalAttnMIL)", script="models.py",
         detail="Each patch's 4096 numbers are compressed to 512. Attention then combines patches into one summary "
                "per slice, slices into one per stain, and stains into one per case, giving the most weight to the "
                "most informative parts. A final layer turns the case summary into two scores: benign and "
                "high-grade."),
])

# ---- Column C: Stage 4 training & evaluation ----
spansC = column(xC, TOP, [
    dict(title="6  MILTrainer.train  ↻ repeats every epoch", script="trainer.py",
         detail="The model trains on every training case once, using a loss that counts benign mistakes 2.53 times "
                "more and adds a small term that keeps attention from fixating on one patch. It then measures "
                "validation loss, lowers the learning rate if progress stalls, and saves a checkpoint (best.pth "
                "whenever validation loss improves). Training stops after 8 epochs with no improvement, but never "
                "before epoch 10."),
    dict(title="7  load best.pth", script="main.py",
         detail="Reloads the weights from the epoch with the lowest validation loss, so the test uses that version "
                "of the model rather than the last epoch's."),
    dict(title="8  evaluate(test_loader)", script="trainer.py",
         detail="Runs every test case through the model and turns its scores into probabilities of benign and "
                "high-grade. Saves them to predictions.csv and draws the confusion matrix."),
    dict(title="9  analyze_attention_weights", script="attention_analysis.py",
         detail="Runs the test cases again while recording attention weights. Multiplies the stain, slice and "
                "patch weights to rank every patch, and saves images of the most and least important ones."),
    dict(title="10  write results.txt", script="main.py",
         detail="Writes the test loss, accuracy and number of test cases to a summary file."),
    dict(title="After all 5 jobs:  analyze_results.py", script="analyze_results.py", script_label="my script",
         detail="Combines the 5 runs into one table of accuracy, recall per class and runtime, and plots the loss "
                "curves and confusion matrices."),
])

# ---- Cross-column connectors ----
gapAB = (xA + xB) / 2
gapBC = (xB + xC) / 2
tA5, bA5 = spansA[-1]
tB3, bB3 = spansB[2]
elbow(xA + W / 2, (tA5 + bA5) / 2, gapAB, (tB3 + bB3) / 2, xB - W / 2, ls="--")   # split files -> prepare_data
tB5, bB5 = spansB[-1]
tC1, bC1 = spansC[0]
elbow(xB + W / 2, (tB5 + bB5) / 2, gapBC, (tC1 + bC1) / 2, xC - W / 2)            # model -> training loop

lowest = min(s[-1][1] for s in (spansA, spansB, spansC))
if lowest < 0.05:
    print(f"WARNING: content runs off the bottom (lowest box edge at y={lowest:.2f})")

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, "flowchart.png"), dpi=200)
fig.savefig(os.path.join(OUT_DIR, "flowchart.svg"))
print(f"Saved flowchart.png and flowchart.svg to {OUT_DIR}/  (lowest box edge y={lowest:.2f})")
