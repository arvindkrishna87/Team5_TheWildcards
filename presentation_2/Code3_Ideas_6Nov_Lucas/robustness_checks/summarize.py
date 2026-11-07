#!/usr/bin/env python3
"""Summarize the robustness checks into RESULTS_SUMMARY.md and results/*.csv. Safe to rerun; skips what is missing."""
import os
import re
import glob

import numpy as np
import pandas as pd

O = os.path.expanduser("~/STAT390_coder/robustness_checks")
MAIN = os.path.expanduser("~/STAT390_coder/presentation2/results_summary/all_test_predictions.csv")
R = os.path.join(O, "results")
NAMES = {"kimianet": "KimiaNet", "uni2": "UNI2"}


def score(d):
    b, h = d[d.true_label == 0], d[d.true_label == 1]
    return pd.Series({"cases": len(d), "accuracy": d.correct.mean(), "benign_recall": b.correct.mean(),
                      "high_grade_recall": h.correct.mean()})


rows, failed = [], []
for path in sorted(glob.glob(os.path.join(O, "logs", "RC_*_s?_*.out"))):
    name = os.path.basename(path)
    m = re.match(r"RC_(cv|lc)_(kimianet|uni2)_(seed\d+|f[\d.]+)_s(\d)_", name)
    text = open(path).read()
    if "Training completed successfully" not in text:
        failed.append(name)
        continue
    run = re.search(r"Created run directory: (\S+)", text).group(1)
    d = pd.read_csv(glob.glob(os.path.join(run, "predictions*.csv"))[0])
    rows.append(d.assign(kind=m.group(1), encoder=m.group(2), setting=m.group(3), split=int(m.group(4))))
runs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
main = pd.read_csv(MAIN).assign(kind="main", setting="seed436")

out = ["# Robustness checks (exploratory)", "",
       "Not part of the main results. Every run is included; none was chosen by test score.", ""]
out.append(f"Finished training jobs: {len(rows)} of 50. Failed or unfinished: {len(failed)}.")
out += [f"- {f}" for f in failed]

if len(runs):
    cv = pd.concat([main, runs[runs.kind == "cv"]], ignore_index=True)
    t = cv.groupby(["encoder", "setting"]).apply(score, include_groups=False).reset_index()
    t.to_csv(os.path.join(R, "repeated_cv_by_seed.csv"), index=False)
    agg = t.groupby("encoder")[["accuracy", "benign_recall", "high_grade_recall"]].agg(["mean", "std", "min", "max"])
    agg.to_csv(os.path.join(R, "repeated_cv_summary.csv"))
    w = cv.pivot_table(index=["setting", "case_id"], columns="encoder", values="correct", aggfunc="first").astype(bool)
    disc = w.groupby("setting").apply(lambda g: pd.Series({"only_kimianet": int((g.kimianet & ~g.uni2).sum()),
                                                           "only_uni2": int((~g.kimianet & g.uni2).sum())}),
                                      include_groups=False)
    disc.to_csv(os.path.join(R, "repeated_cv_paired.csv"))
    hard = cv.groupby(["encoder", "case_id"]).correct.mean().unstack(0)
    hard[hard.uni2 < 0.5].to_csv(os.path.join(R, "repeated_cv_hard_cases_uni2.csv"))
    out += ["", "## 1. Repeated cross-validation (4 split seeds, including 436)", "", t.round(3).to_string(index=False), "",
            "Cases where only one encoder was right, per split seed:", "", disc.to_string()]

    lc = pd.concat([runs[runs.kind == "lc"], main.assign(kind="lc", setting="f1.0")], ignore_index=True)
    c = lc.groupby(["encoder", "setting"]).apply(score, include_groups=False).reset_index()
    c.to_csv(os.path.join(R, "learning_curve.csv"), index=False)
    out += ["", "## 2. Learning curve (seed 436; f = share of training cases)", "", c.round(3).to_string(index=False)]

f = os.path.join(R, "baseline_logreg_predictions.csv")
if os.path.exists(f):
    b = pd.read_csv(f)
    t = b.groupby("encoder").apply(score, include_groups=False).reset_index()
    t.to_csv(os.path.join(R, "baseline_logreg_summary.csv"), index=False)
    m = main.groupby("encoder").apply(score, include_groups=False).reset_index()
    out += ["", "## 3. Logistic regression baseline (no attention) vs attention model, seed 436", "",
            "Baseline:", "", t.round(3).to_string(index=False), "", "Attention model:", "", m.round(3).to_string(index=False)]

f = os.path.join(R, "stain_knockout_predictions.csv")
if os.path.exists(f):
    k = pd.read_csv(f)
    t = k.groupby(["encoder", "stains"]).apply(score, include_groups=False).reset_index()
    t.to_csv(os.path.join(R, "stain_knockout_summary.csv"), index=False)
    chk = k[k.stains == "all"].merge(main, on=["encoder", "split", "case_id"], suffixes=("", "_main"))
    diff = float((chk.prob_class1 - chk.prob_class1_main).abs().max())
    out += ["", "## 4. Stain knockout (no retraining; cases missing the kept stains are left out)", "",
            f"Check: the 'all' row matches the main predictions to within {diff:.6f}.", "", t.round(3).to_string(index=False)]

open(os.path.join(O, "RESULTS_SUMMARY.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
