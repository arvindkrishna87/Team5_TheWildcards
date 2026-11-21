#!/usr/bin/env python3
"""
make_results.py  (Presentation 3 — slide 32; DECISIONS.md D15)

Reads runs/<run>/outer_<G>/ for G in A-E (whatever exists) and writes to <run>/results/:
  results_table.csv / .md   one row per outer test group + pooled row + range across groups
  pooled_predictions.csv    all test predictions (every case tested once)
  selected_hyperparameters.csv
  plots/outer_<G>_refit.png  per refit model: (1) loss per update, (2) loss per epoch,
                             (3) predicted P(high-grade) vs true label
Accuracy uses threshold 0.5 on P(high-grade), as in the base code's argmax.
The "validation" curve for a refit model is the outer test group's loss, RECORDED for
monitoring only (D13); it never influenced training or selection.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = ["A", "B", "C", "D", "E"]
HP_COLS = ["learning_rate", "weight_decay", "classifier_dropout", "entropy_weight", "projector_dropout"]


def metrics(df):
    pred = (df.prob_high_grade >= 0.5).astype(int)
    return {"n": len(df), "accuracy": float((pred == df.label).mean()),
            "benign_recall": float((pred[df.label == 0] == 0).mean()),
            "high_grade_recall": float((pred[df.label == 1] == 1).mean())}


def plot_refit(g, hist, preds, summ, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    up = np.array(hist["update_train_loss"])
    ax[0].plot(np.arange(1, len(up) + 1), up, lw=0.6, alpha=0.5, color="#5b3a8e", label="train loss (each update)")
    k = 10
    if len(up) >= k:
        ax[0].plot(np.arange(k, len(up) + 1), np.convolve(up, np.ones(k) / k, mode="valid"),
                   color="#5b3a8e", lw=1.6, label=f"train loss ({k}-update mean)")
    if hist["update_monitor"]:
        u, m = zip(*hist["update_monitor"])
        ax[0].plot(u, m, color="#d08a1e", lw=1.6, label="test-group loss (monitor only)")
    ax[0].set(xlabel="update", ylabel="loss", title=f"Outer {g} refit: loss per update")
    ax[0].legend(fontsize=8)

    ep = pd.DataFrame(hist["epoch"])
    ax[1].plot(ep.epoch, ep.train_loss, "o-", color="#5b3a8e", label="train loss (weighted CE − λH)")
    ax[1].plot(ep.epoch, ep.val_loss, "o-", color="#d08a1e", label="test-group balanced CE (monitor only)")
    ax[1].set(xlabel="epoch", ylabel="loss", title=f"Loss per epoch ({summ['refit_epochs']} epochs)")
    ax[1].legend(fontsize=8)

    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.12, 0.12, len(preds))
    correct = ((preds.prob_high_grade >= 0.5).astype(int) == preds.label)
    for ok, mk, col in [(True, "o", "#5b3a8e"), (False, "X", "#c0392b")]:
        s = preds[correct == ok]
        ax[2].scatter(s.label + jit[correct.values == ok], s.prob_high_grade, marker=mk, color=col, s=45,
                      label="correct" if ok else "misclassified")
    for _, r in preds[~correct].iterrows():
        ax[2].annotate(str(r.case_id), (r.label, r.prob_high_grade), xytext=(8, 0), textcoords="offset points", fontsize=8)
    ax[2].axhline(0.5, ls="--", color="gray", lw=1)
    ax[2].set(xticks=[0, 1], xticklabels=["benign", "high-grade"], xlim=(-0.5, 1.5), ylim=(-0.03, 1.03),
              ylabel="predicted P(high-grade)", title=f"Test group {g}: probability vs true label")
    ax[2].legend(fontsize=8, loc="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    out = os.path.join(args.run_dir, "results")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)

    rows, hps, allp = [], [], []
    for g in GROUPS:
        d = os.path.join(args.run_dir, f"outer_{g}")
        if not os.path.exists(os.path.join(d, "summary.json")):
            print(f"outer {g}: not finished, skipped")
            continue
        summ = json.load(open(os.path.join(d, "summary.json")))
        preds = pd.read_csv(os.path.join(d, "test_predictions.csv"))
        hist = json.load(open(os.path.join(d, "refit_history.json")))
        allp.append(preds)
        rows.append({"outer_test_group": g, **metrics(preds)})
        hps.append({"outer_test_group": g, "selected_trial": summ["selected_trial"],
                    "f_selected": summ["selected_f"], **summ["selected_hp"],
                    "refit_epochs": summ["refit_epochs"], "trials_complete": summ["n_trials_complete"],
                    "edge_flags": ";".join(summ["edge_flags"])})
        plot_refit(g, hist, preds, summ, os.path.join(out, "plots", f"outer_{g}_refit.png"))

    per = pd.DataFrame(rows)
    pooled = pd.concat(allp, ignore_index=True)
    assert pooled.case_id.is_unique, "a case was tested more than once"
    pooled.to_csv(os.path.join(out, "pooled_predictions.csv"), index=False)
    pr = {"outer_test_group": "pooled", **metrics(pooled)}
    rg = {"outer_test_group": "range across groups", "n": "",
          **{m: f"{per[m].min():.3f}–{per[m].max():.3f}" for m in ["accuracy", "benign_recall", "high_grade_recall"]}}
    table = pd.concat([per, pd.DataFrame([pr, rg])], ignore_index=True)
    table.to_csv(os.path.join(out, "results_table.csv"), index=False)
    pd.DataFrame(hps).to_csv(os.path.join(out, "selected_hyperparameters.csv"), index=False)
    # plain-text tables (to_markdown needs `tabulate`, which the shared env lacks; we don't modify that env)
    with open(os.path.join(out, "results_table.txt"), "w") as fh:
        fh.write(table.to_string(index=False) + "\n\n" + pd.DataFrame(hps).to_string(index=False) + "\n")
    print(table.to_string(index=False))
    print(pd.DataFrame(hps).to_string(index=False))


if __name__ == "__main__":
    main()
