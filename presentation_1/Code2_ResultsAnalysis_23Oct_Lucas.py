#!/usr/bin/env python3
"""
Collect results from the 5 training runs for the Coders presentation.

Reads each job log in logs/ (to find its split, run folder, losses and runtime)
and each run's predictions.csv, then writes to results_summary/:
  summary_table.csv       accuracy, per-class recall, confusion counts, runtime per split
  all_test_predictions.csv predicted probability for every test case of every split
  loss_curves.png         train / val loss per split (best epoch marked)
  confusion_matrices.png  one confusion matrix per split

Usage:  python analyze_results.py
"""
import os
import re
import glob

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/STAT390_coder")
LOGS_DIR = os.path.join(BASE, "logs")
OUT_DIR = os.path.join(BASE, "results_summary")

NUM = r"(-?[\d.]+)"   # train loss can go negative because the entropy term is subtracted
EPOCH_RE = re.compile(rf"Train Loss: {NUM} \| Train CE: {NUM} \| Train Patch Entropy: {NUM} \| "
                      rf"Val Loss: {NUM} \| Val Acc: {NUM}")


def parse_log(path):
    text = open(path).read()
    split = re.search(r"^Split: (\d)", text, re.M)
    run_dir = re.search(r"Created run directory: (\S+)", text)
    if not split or not run_dir:
        return None
    runtime = re.search(r"Total execution time: ([\d.]+)s", text)
    best = re.search(r"best\.pth \(epoch=(\d+)\)", text)
    epochs = [tuple(map(float, m)) for m in EPOCH_RE.findall(text)]
    return {
        "split": int(split.group(1)),
        "job_id": re.search(r"MIL_split_(\d+)\.out", path).group(1),
        "run_dir": run_dir.group(1),
        "runtime_min": float(runtime.group(1)) / 60 if runtime else None,
        "best_epoch": int(best.group(1)) if best else None,
        "losses": pd.DataFrame(epochs, columns=["train_loss", "train_ce", "train_entropy", "val_loss", "val_acc"],
                               index=range(1, len(epochs) + 1)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    runs = {}
    for path in sorted(glob.glob(os.path.join(LOGS_DIR, "MIL_split_*.out"))):
        info = parse_log(path)
        if info is None or info["runtime_min"] is None:
            print(f"Skipping {os.path.basename(path)} (not finished or not a training log)")
            continue
        if info["split"] in runs:
            print(f"Split {info['split']} appears twice; using the later job {info['job_id']}")
        runs[info["split"]] = info

    if not runs:
        print("No finished runs found.")
        return

    rows, all_preds = [], []
    for split in sorted(runs):
        r = runs[split]
        preds = pd.read_csv(os.path.join(r["run_dir"], "predictions.csv"))
        preds.insert(0, "split", split)
        all_preds.append(preds)

        t, p = preds["true_label"], preds["predicted_label"]
        tn = int(((t == 0) & (p == 0)).sum())   # benign predicted benign
        fp = int(((t == 0) & (p == 1)).sum())   # benign predicted high-grade
        fn = int(((t == 1) & (p == 0)).sum())   # high-grade predicted benign
        tp = int(((t == 1) & (p == 1)).sum())   # high-grade predicted high-grade
        rows.append({
            "split": split,
            "job_id": r["job_id"],
            "n_test": len(preds),
            "accuracy": (tn + tp) / len(preds),
            "recall_benign": tn / (tn + fp) if tn + fp else float("nan"),
            "recall_high_grade": tp / (tp + fn) if tp + fn else float("nan"),
            "TN_benign_ok": tn, "FP_benign_as_HG": fp, "FN_HG_as_benign": fn, "TP_HG_ok": tp,
            "epochs_run": len(r["losses"]),
            "best_epoch": r["best_epoch"],
            "runtime_min": round(r["runtime_min"], 1),
            "run_dir": r["run_dir"],
        })

    summary = pd.DataFrame(rows)
    mean_row = summary[["accuracy", "recall_benign", "recall_high_grade", "runtime_min"]].mean()
    std_row = summary[["accuracy", "recall_benign", "recall_high_grade", "runtime_min"]].std()
    summary.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
    pd.concat(all_preds).to_csv(os.path.join(OUT_DIR, "all_test_predictions.csv"), index=False)

    # Loss curves
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
    for ax, split in zip(axes[0], sorted(runs)):
        L = runs[split]["losses"]
        ax.plot(L.index, L["train_loss"], marker="o", ms=3, label="train")
        ax.plot(L.index, L["val_loss"], marker="o", ms=3, label="val")
        if runs[split]["best_epoch"]:
            ax.axvline(runs[split]["best_epoch"], color="gray", ls="--", lw=1, label="best epoch")
        ax.set_title(f"Split {split}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Loss")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=150)
    plt.close(fig)

    # Confusion matrices
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    for ax, row in zip(axes[0], rows):
        m = [[row["TN_benign_ok"], row["FP_benign_as_HG"]], [row["FN_HG_as_benign"], row["TP_HG_ok"]]]
        ax.imshow(m, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, m[i][j], ha="center", va="center", fontsize=14,
                        color="white" if m[i][j] > max(map(max, m)) / 2 else "black")
        ax.set_xticks([0, 1], ["Benign", "High-grade"])
        ax.set_yticks([0, 1], ["Benign", "High-grade"])
        ax.set_xlabel("Predicted")
        ax.set_title(f"Split {row['split']}  (acc {row['accuracy']:.2f})")
    axes[0][0].set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(summary.drop(columns=["run_dir"]).to_string(index=False))
    print("\nMean across splits:\n" + mean_row.round(3).to_string())
    print("\nStd across splits:\n" + std_row.round(3).to_string())
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
