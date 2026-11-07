#!/usr/bin/env python3
"""
Compare KimiaNet and UNI2 across the 5 splits for Presentation 2.

Finds each finished run from its job log in logs/, then writes to results_summary/:
  summary_table.csv         accuracy, per-class recall and confusion counts per split, plus all 93 cases pooled
  all_test_predictions.csv  predicted probability for every test case, for each encoder
  paired_comparison.txt     cases where exactly one encoder was right, with an exact McNemar test
  confusion_matrices.png    pooled confusion matrix per encoder
  metrics_by_split.png      accuracy and both recalls per split
  loss_per_epoch.png        train / validation loss per epoch, one panel per split
  loss_per_update.png       train / validation loss per gradient update, one panel per split (train: 20-update rolling mean; validation: every 10 updates)
  deviation_plots.png       true label vs predicted probability for all 93 test cases

Usage:  python compare_encoders.py
"""
import os
import re
import glob

import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/STAT390_coder/presentation2")
LOGS_DIR = os.path.join(BASE, "logs")
OUT_DIR = os.path.join(BASE, "results_summary")

ENCODERS = {"kimianet": "KimiaNet", "uni2": "UNI2"}
ENC_STYLE = {"kimianet": ("#2a78d6", "o"), "uni2": ("#eb6834", "s")}
TRAIN, VAL = "#1baf7a", "#7a5bd1"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SEQ = ["#eef4fc", "#c7dcf5", "#8fb8eb", "#4f8fdc", "#2a78d6", "#1d5aa6"]
CLASS_NAMES = ["Benign", "High-grade"]
LOSS_YMAX = 3  # Early spikes above this are cut off so the later curves stay readable

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 13, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def find_runs():
    """Return {encoder: {split: run_dir}} for finished runs; later jobs replace earlier ones."""
    runs = {enc: {} for enc in ENCODERS}
    for path in sorted(glob.glob(os.path.join(LOGS_DIR, "P2_*_s?_*.out"))):
        match = re.search(r"P2_(kimianet|uni2)_s(\d)_", os.path.basename(path))
        text = open(path).read()
        run_dir = re.search(r"Created run directory: (\S+)", text)
        if match and run_dir and "Training completed successfully" in text:
            runs[match.group(1)][int(match.group(2))] = run_dir.group(1)
    return runs


def metrics(df):
    y, p = df["true_label"], df["predicted_label"]
    tn, fp = int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tp = int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum())
    return {"n": len(df), "accuracy": (tn + tp) / len(df), "benign_recall": tn / max(tn + fp, 1),
            "high_grade_recall": tp / max(tp + fn, 1), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def load(runs):
    preds, rows, epochs, updates = [], [], {}, {}
    for enc, by_split in runs.items():
        for split, run_dir in sorted(by_split.items()):
            df = pd.read_csv(os.path.join(run_dir, "predictions_best.csv"))
            df["encoder"], df["split"] = enc, split
            preds.append(df)
            ep = pd.read_csv(os.path.join(run_dir, "checkpoints", "epoch_losses.csv"))
            epochs[enc, split] = ep
            updates[enc, split] = pd.read_csv(os.path.join(run_dir, "checkpoints", "update_losses.csv"))
            rows.append({"encoder": enc, "split": split, **metrics(df),
                         "best_epoch": int(ep.loc[ep["val_loss"].idxmin(), "epoch"]),
                         "epochs_run": len(ep), "best_val_loss": ep["val_loss"].min()})
    preds = pd.concat(preds, ignore_index=True)
    for enc, df in preds.groupby("encoder"):
        rows.append({"encoder": enc, "split": "all", **metrics(df)})
    return preds, pd.DataFrame(rows), epochs, updates


def plot_confusion(table, encs):
    fig, axes = plt.subplots(1, len(encs), figsize=(5.6 * len(encs), 5), squeeze=False)
    for ax, enc in zip(axes[0], encs):
        row = table[(table["encoder"] == enc) & (table["split"] == "all")].iloc[0]
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=cm.sum() * 0.6)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=26, fontweight="bold",
                        color="white" if cm[i, j] > cm.sum() * 0.3 else INK)
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{ENCODERS[enc]}: accuracy {row['accuracy']:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=200)
    plt.close(fig)


def plot_metrics(table, encs):
    names = [("accuracy", "Accuracy"), ("benign_recall", "Benign recall"), ("high_grade_recall", "High-grade recall")]
    per_split = table[table["split"] != "all"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (col, title) in zip(axes, names):
        for k, enc in enumerate(encs):
            color, marker = ENC_STYLE[enc]
            df = per_split[per_split["encoder"] == enc]
            x = df["split"].astype(int) + (k - (len(encs) - 1) / 2) * 0.16
            ax.scatter(x, df[col], s=90, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5,
                       label=ENCODERS[enc], zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Split")
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 1.05)
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "metrics_by_split.png"), dpi=200)
    plt.close(fig)


def loss_panels(curves, encs, draw, xlabel, name):
    splits = sorted({s for _, s in curves})
    fig, axes = plt.subplots(len(encs), len(splits), figsize=(3.2 * len(splits), 3 * len(encs)), sharex=True, sharey=True, squeeze=False)
    for row, enc in zip(axes, encs):
        for ax, split in zip(row, splits):
            if (enc, split) in curves:
                draw(ax, curves[(enc, split)])
            ax.set_title(f"{ENCODERS[enc]}, split {split}", fontsize=11)
    axes[0][0].set_ylim(0, LOSS_YMAX)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for row in axes:
        row[0].set_ylabel("Weighted loss")
    handles = [plt.Line2D([], [], color=TRAIN, lw=2, label="Train"), plt.Line2D([], [], color=VAL, lw=2, label="Validation")]
    axes[0][-1].legend(handles=handles, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=200)
    plt.close(fig)


def plot_losses(epochs, updates, encs):
    def by_epoch(ax, df):
        ax.plot(df["epoch"], df["train_ce"], color=TRAIN, lw=1.6)
        ax.plot(df["epoch"], df["val_loss"], color=VAL, lw=1.6)

    def by_update(ax, df):
        # One case per update, so raw train loss is very noisy: show a 20-update rolling mean
        ax.plot(df["update"], df["train_ce"].rolling(20, min_periods=1).mean(), color=TRAIN, lw=1.2)
        val = df.dropna(subset=["val_loss"])
        ax.plot(val["update"], val["val_loss"], color=VAL, lw=1.2)

    loss_panels(epochs, encs, by_epoch, "Epoch", "loss_per_epoch.png")
    loss_panels(updates, encs, by_update, "Gradient update", "loss_per_update.png")


def plot_deviation(preds, encs):
    order = preds.drop_duplicates("case_id").sort_values(["true_label", "case_id"])["case_id"].tolist()
    ypos = {c: i for i, c in enumerate(order)}
    fig, axes = plt.subplots(1, len(encs), figsize=(7.5 * len(encs), 14), sharey=True, squeeze=False)
    for ax, enc in zip(axes[0], encs):
        df = preds[preds["encoder"] == enc]
        for _, r in df.iterrows():
            y = ypos[r["case_id"]]
            ax.plot([r["true_label"], r["prob_class1"]], [y, y], color=MUTED, lw=0.8, zorder=1)
        y = df["case_id"].map(ypos)
        ax.scatter(df["true_label"], y, s=42, color=TRAIN, zorder=2, label="True label")
        ax.scatter(df["prob_class1"], y, s=38, color=INK, marker="x", linewidth=1.2, zorder=3, label="Predicted probability")
        ax.axvline(0.5, color=MUTED, ls="--", lw=1.2, label="0.5 cutoff")
        ax.set_yticks(range(len(order)), order, fontsize=8)
        ax.set_ylim(-1, len(order))
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("True label (0 or 1) vs predicted probability of high-grade")
        ax.set_title(ENCODERS[enc])
    axes[0][0].set_ylabel("Case (sorted by true label)")
    axes[0][0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "deviation_plots.png"), dpi=170)
    plt.close(fig)


def paired_comparison(preds):
    wide = preds.pivot(index="case_id", columns="encoder", values="correct")
    k_only, u_only = int((wide["kimianet"] & ~wide["uni2"]).sum()), int((~wide["kimianet"] & wide["uni2"]).sum())
    p = binomtest(k_only, k_only + u_only, 0.5).pvalue if k_only + u_only else 1.0
    return (f"Cases compared: {len(wide)}\nOnly KimiaNet correct: {k_only}\nOnly UNI2 correct: {u_only}\n"
            f"Both correct: {int((wide['kimianet'] & wide['uni2']).sum())}\n"
            f"Both wrong: {int((~wide['kimianet'] & ~wide['uni2']).sum())}\nExact McNemar p-value: {p:.3f}\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = find_runs()
    for enc, by_split in runs.items():
        print(f"{ENCODERS[enc]}: finished splits {sorted(by_split)}")
    encs = [enc for enc, by_split in runs.items() if by_split]

    preds, table, epochs, updates = load(runs)
    table.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
    preds.to_csv(os.path.join(OUT_DIR, "all_test_predictions.csv"), index=False)
    print(table.round(3).to_string(index=False))

    per_split = table[table["split"] != "all"]
    print(per_split.groupby("encoder")[["accuracy", "benign_recall", "high_grade_recall"]].agg(["mean", "std"]).round(3).to_string())

    plot_confusion(table, encs)
    plot_metrics(table, encs)
    plot_losses(epochs, updates, encs)
    plot_deviation(preds, encs)
    if len(encs) == 2 and all(len(runs[e]) == 5 for e in encs):
        text = paired_comparison(preds)
        open(os.path.join(OUT_DIR, "paired_comparison.txt"), "w").write(text)
        print(text)


if __name__ == "__main__":
    main()
