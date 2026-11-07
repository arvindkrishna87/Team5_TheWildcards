#!/usr/bin/env python3
"""compare KimiaNet and UNI2 when both get the same input.

Writes to results_summary/ in this folder: same_input_summary.csv, same_input_by_split.csv, same_input_paired_tests.txt.
Usage:  python compare_same_input.py
"""
import os
import re
import glob

import pandas as pd
from scipy.stats import binomtest

X = os.path.expanduser("~/STAT390_coder/same_input_comparison")
MAIN = os.path.expanduser("~/STAT390_coder/presentation2/results_summary/all_test_predictions.csv")
OUT = os.path.join(X, "results_summary")

main = pd.read_csv(MAIN)
frames = [main[main.encoder == "kimianet"].assign(condition="KimiaNet, original size"),
          main[main.encoder == "uni2"].assign(condition="UNI2, 224x224")]

NAMES = {"A": "KimiaNet, 224x224", "B": "UNI2, original size"}
for path in sorted(glob.glob(os.path.join(X, "logs", "SAME_INPUT_[AB]_s*.out"))):
    text = open(path).read()
    test, split = re.search(r"SAME_INPUT_([AB])_s(\d)_", path).groups()
    run_dir = re.search(r"Created run directory: (\S+)", text).group(1)
    assert "Training completed successfully" in text, path
    d = pd.read_csv(glob.glob(os.path.join(run_dir, "predictions*.csv"))[0])
    frames.append(d.assign(condition=NAMES[test], split=int(split)))
preds = pd.concat(frames, ignore_index=True)


def metrics(d):
    b, h = d[d.true_label == 0], d[d.true_label == 1]
    conf = d[["prob_class0", "prob_class1"]].max(axis=1)
    return pd.Series({"n": len(d), "correct": int(d.correct.sum()), "accuracy": d.correct.mean(),
                      "benign_recall": b.correct.mean(), "high_grade_recall": h.correct.mean(),
                      "median_conf_when_wrong": conf[~d.correct].median()})


order = ["KimiaNet, original size", "KimiaNet, 224x224", "UNI2, original size", "UNI2, 224x224"]
summary = preds.groupby("condition").apply(metrics, include_groups=False).loc[order]
by_split = preds.pivot_table(index="split", columns="condition", values="correct", aggfunc="mean")[order]

wide = preds.pivot_table(index="case_id", columns="condition", values="correct", aggfunc="first").astype(bool)
assert len(wide) == 93 and wide.notna().all().all()


def paired(a, b):
    only_a, only_b = int((wide[a] & ~wide[b]).sum()), int((~wide[a] & wide[b]).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return f"{a} vs {b}: only first right {only_a}, only second right {only_b}, exact McNemar p={p:.4f}"


lines = ["Same input, different encoder:", paired("KimiaNet, 224x224", "UNI2, 224x224"),
         paired("KimiaNet, original size", "UNI2, original size"), "",
         "Same encoder, different input:", paired("KimiaNet, original size", "KimiaNet, 224x224"),
         paired("UNI2, original size", "UNI2, 224x224")]

os.makedirs(OUT, exist_ok=True)
summary.to_csv(os.path.join(OUT, "same_input_summary.csv"))
by_split.to_csv(os.path.join(OUT, "same_input_by_split.csv"))
open(os.path.join(OUT, "same_input_paired_tests.txt"), "w").write("\n".join(lines) + "\n")
pd.set_option("display.width", 200)
print(summary.round(3).to_string())
print()
print(by_split.round(3).to_string())
print()
print("\n".join(lines))
