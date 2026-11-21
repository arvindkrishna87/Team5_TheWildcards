#!/usr/bin/env python3
"""Slide-ready figures for Presentation 3 (Coders: nested CV with UNI2).

Same look as the Presentation 2 figures (presentation2/make_slide_figures.py and compare_encoders.py):
same palette, fonts and layout, so the P3 slides match the P2 slides.
Why images only: Lucas places them on the team Google Slides himself and adds the code-link
footnotes there later (decided 2026-11-20, see DECISIONS.md D18). No links are drawn into the images.

Reads only finished outputs of the full run (runs/full_40trials/outer_*/ and results/). It does not
retrain or rescore anything, so the numbers are exactly the ones in DECISIONS.md D16.

Writes to runs/full_40trials/results/slides/:
  1_nested_cv_design.png       what was run (outer 5 groups, inner 4-fold Optuna, refit, test once)
  2_results_table.png          accuracy and recalls per outer test group + pooled 99 + range
  3_confusion_matrix.png       pooled over all 99 cases (each tested exactly once)
  4_loss_per_update.png        refit loss per gradient update, groups A-E
  5_loss_per_epoch.png         refit loss per epoch, groups A-E
  6_prediction_every_case.png  true label vs predicted probability for all 99 cases
  7_selected_hyperparameters.png  chosen x-hat per group, edge flags marked

Usage:  python make_slide_figures.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RUN = os.path.expanduser("~/STAT390_coder/presentation3/runs/full_40trials")
RES = os.path.join(RUN, "results")
OUT = os.path.join(RES, "slides")
GROUPS = ["A", "B", "C", "D", "E"]

# Presentation 2 palette (validated there); kept identical so the decks match
BLUE, ORANGE = "#2a78d6", "#eb6834"
TRAIN, VAL = "#1baf7a", "#7a5bd1"
SURFACE, INK, INK2, MUTED, GRID, HEAD = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#f1efe9"
POOLED_BG, FLAG_BG = "#f7efe8", "#fde3d6"
SEQ = ["#eef4fc", "#c7dcf5", "#8fb8eb", "#4f8fdc", "#2a78d6", "#1d5aa6"]
LOSS_YMAX = 3  # same cut-off as P2: early spikes above 3 are cut so later curves stay readable

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 13, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load():
    summ, hist = {}, {}
    for g in GROUPS:
        d = os.path.join(RUN, f"outer_{g}")
        summ[g] = json.load(open(os.path.join(d, "summary.json")))
        hist[g] = json.load(open(os.path.join(d, "refit_history.json")))
    preds = pd.read_csv(os.path.join(RES, "pooled_predictions.csv"))
    return summ, hist, preds


def canvas():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    return fig, ax


def title(ax, head, sub):
    ax.text(0.6, 8.3, head, fontsize=26, weight="bold", color=INK, va="center")
    ax.text(0.6, 7.75, sub, fontsize=14, color=INK2, va="center")


# ---------------------------------------------------------------- 1. design
def design(summ):
    """Slide 1, v3 (2026-11-20): restyled to match Lucas's own Google Slides tables (Poppins font,
    grey header #f1f0ed, blue-grey cells #e8ebf4, salmon highlight #ef8a80, pink pooled row #f7efe8).
    No title in the image (his slides carry the title) and a transparent background so it sits on his
    slide background. Content is the same 3 steps as v2."""
    from matplotlib import font_manager
    fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    # Poppins (google/fonts, OFL) is not in the repo; without it matplotlib falls back to its default font
    for f in (os.listdir(fdir) if os.path.isdir(fdir) else []):
        if f.endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(fdir, f))
    HEADC, CELL, TESTC, BANNER, VALC, TXT, TXT2 = "#f1f0ed", "#e8ebf4", "#ef8a80", "#f7efe8", "#3b7fe0", "#1f1f1f", "#4a4a4a"

    with plt.rc_context({"font.family": "Poppins"}):
        fig = plt.figure(figsize=(16, 7.6))
        fig.patch.set_alpha(0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 7.6)
        ax.axis("off")

        def cell(x, y, w, h, text, fc, fs=15, bold=False, color=TXT):
            ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="white", lw=2.5))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                    weight="bold" if bold else "normal", color=color)

        def header(x, w, n, text):
            ax.add_patch(plt.Rectangle((x, 6.55), w, 0.85, fc=HEADC, ec="white", lw=2.5))
            ax.text(x + 0.25, 6.975, f"{n}.  {text}", va="center", fontsize=17, weight="bold", color=TXT)

        def arrow(x0, x1, y=4.0):
            ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color="#9a9a9a", lw=2.2,
                                                              mutation_scale=22))

        # step 1
        header(0.3, 4.7, 1, "Hold out a test group")
        for i, g in enumerate(GROUPS):
            test = g == "A"
            cell(0.45 + i * 0.88, 3.6, 0.82, 0.82, g, TESTC if test else CELL, fs=18, bold=test)
        ax.text(2.65, 2.95, "A is locked away.\nIts labels are not used\nuntil the very end.",
                ha="center", va="top", fontsize=14.5, color=TXT2, linespacing=1.5)
        arrow(5.0, 5.45)

        # step 2
        header(5.55, 5.0, 2, "Pick settings using only B–E")
        inner = ["B", "C", "D", "E"]
        for r, v in enumerate(inner):
            y = 5.55 - r * 0.62
            for c, g in enumerate(inner):
                val = g == v
                cell(5.75 + c * 0.76, y, 0.76, 0.62, g, VALC if val else CELL, fs=13, bold=val,
                     color="white" if val else TXT)
            ax.text(8.95, y + 0.31, f"model {r + 1}", va="center", fontsize=13, color=TXT2)
        ax.text(8.05, 2.95, "For each of 40 settings, train 4 models.\nEach is checked on a different group (blue).\n"
                            "Keep the setting with the lowest\naverage validation loss.",
                ha="center", va="top", fontsize=14.5, color=TXT2, linespacing=1.5)
        arrow(10.55, 11.0)

        # step 3
        header(11.1, 4.6, 3, "Train once, test once")
        for c, g in enumerate(inner):
            cell(11.65 + c * 0.88, 4.75, 0.82, 0.7, g, CELL, fs=16)
        ax.annotate("", (13.4, 3.85), (13.4, 4.65), arrowprops=dict(arrowstyle="-|>", color="#9a9a9a", lw=2,
                                                                    mutation_scale=18))
        cell(12.99, 3.0, 0.82, 0.82, "A", TESTC, fs=18, bold=True)
        ax.text(13.4, 2.7, "Train on B–E with the best\nsetting, then predict A's\n20 cases. Done once.",
                ha="center", va="top", fontsize=14.5, color=TXT2, linespacing=1.5)

        # banner, styled like the pooled row of his tables
        ax.add_patch(plt.Rectangle((0.3, 0.25), 15.4, 0.9, fc=BANNER, ec="white", lw=2.5))
        ax.text(8.0, 0.7, "Repeat with B, C, D and E as the test group, so each of the 99 cases is predicted exactly once",
                ha="center", va="center", fontsize=16, weight="bold", color=TXT)
        fig.savefig(os.path.join(OUT, "1_nested_cv_design.png"), dpi=200, transparent=True)
        plt.close(fig)


# ---------------------------------------------------------------- 2. results table
def metrics(df):
    pred = (df.prob_high_grade >= 0.5).astype(int)
    b, h = df.label == 0, df.label == 1
    return dict(n=len(df), acc=(pred == df.label).mean(), ben=(pred[b] == 0).mean(), hg=(pred[h] == 1).mean(),
                ben_k=f"{int((pred[b] == 0).sum())}/{int(b.sum())}", hg_k=f"{int((pred[h] == 1).sum())}/{int(h.sum())}",
                acc_k=f"{int((pred == df.label).sum())}/{len(df)}")


def results_table(preds):
    fig, ax = canvas()
    title(ax, "Nested CV results: 93 of 99 cases correct",
          "Every case is tested exactly once, by a model that never saw its label during tuning")
    top, x0, w = 7.1, 0.6, 14.8
    cols = [("Outer test group", 0.8, "left"), ("Cases", 4.6, "center"), ("Accuracy", 7.2, "center"),
            ("Benign recall", 10.2, "center"), ("High-grade recall", 13.3, "center")]
    ax.add_patch(plt.Rectangle((x0, top - 0.75), w, 0.75, color=HEAD))
    for name, x, ha in cols:
        ax.text(x, top - 0.375, name, fontsize=15, weight="bold", color=INK, ha=ha, va="center")

    rows = [(f"Group {g}", metrics(preds[preds.outer_group == g]), False) for g in GROUPS]
    rows.append(("All 99 cases", metrics(preds), True))
    y = top - 0.75
    for name, m, pooled in rows:
        y -= 0.62
        if pooled:
            ax.add_patch(plt.Rectangle((x0, y), w, 0.62, color=POOLED_BG))
        wt = "bold" if pooled else "normal"
        ax.text(0.8, y + 0.31, name, fontsize=14, color=INK, va="center", weight=wt)
        ax.text(4.6, y + 0.31, str(m["n"]), fontsize=14, color=INK, va="center", ha="center", weight=wt)
        for key, kk, x in [("acc", "acc_k", 7.2), ("ben", "ben_k", 10.2), ("hg", "hg_k", 13.3)]:
            ax.text(x, y + 0.31, f"{m[key]:.3f}  ({m[kk]})", fontsize=14, color=INK, va="center", ha="center",
                    weight=wt)
        ax.plot([x0, x0 + w], [y, y], color=GRID, lw=1)
    # range row
    per = [metrics(preds[preds.outer_group == g]) for g in GROUPS]
    y -= 0.62
    ax.text(0.8, y + 0.31, "Range across groups", fontsize=14, color=INK2, va="center", style="italic")
    for key, x in [("acc", 7.2), ("ben", 10.2), ("hg", 13.3)]:
        vals = [p[key] for p in per]
        ax.text(x, y + 0.31, f"{min(vals):.3f} – {max(vals):.3f}", fontsize=14, color=INK2, va="center",
                ha="center", style="italic")

    wrong = preds[(preds.prob_high_grade >= 0.5).astype(int) != preds.label]
    fp = (wrong.label == 0).sum()
    ax.text(0.6, 1.3, f"{len(wrong)} errors: {fp} benign cases called high-grade, {len(wrong) - fp} high-grade "
                      f"case called benign. Group A has 4 of the 6.", fontsize=15, color=INK, va="center")
    ax.text(0.6, 0.75, "One partition (seed 436) and one training seed per model. Groups have only 19–20 cases, "
                       "so per-group numbers are noisy.", fontsize=13, color=INK2, va="center")
    fig.savefig(os.path.join(OUT, "2_results_table.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 3. confusion matrix
def confusion(preds):
    pred = (preds.prob_high_grade >= 0.5).astype(int)
    cm = np.array([[((preds.label == a) & (pred == b)).sum() for b in (0, 1)] for a in (0, 1)])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=cm.sum() * 0.6)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=30, fontweight="bold",
                    color="white" if cm[i, j] > cm.sum() * 0.3 else INK)
    ax.set_xticks([0, 1], ["Benign", "High-grade"])
    ax.set_yticks([0, 1], ["Benign", "High-grade"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"UNI2, nested CV: accuracy {(pred == preds.label).mean():.2f}")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "3_confusion_matrix.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 4/5. loss curves
def loss_curves(summ, hist):
    k = 10  # same 10-update smoothing as make_results.py; raw per-update train loss is batch-size-1 noise
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), sharey=True)
    for ax, g in zip(axes, GROUPS):
        up = np.array(hist[g]["update_train_loss"])
        ax.plot(np.arange(1, len(up) + 1), up, color=TRAIN, lw=0.5, alpha=0.25)
        ax.plot(np.arange(k, len(up) + 1), np.convolve(up, np.ones(k) / k, mode="valid"), color=TRAIN, lw=1.6,
                label=f"Train ({k}-update mean)")
        u, m = zip(*hist[g]["update_monitor"])
        ax.plot(u, m, color=VAL, lw=1.6, label="Test group (monitor only)")
        ax.set_ylim(0, LOSS_YMAX)
        ax.set_xlabel("Gradient update")
        ax.set_title(f"Group {g} refit ({summ[g]['refit_epochs']} epochs)", fontsize=13)
        ax.grid(color=GRID, lw=0.8)
    axes[0].set_ylabel("Loss")
    axes[-1].legend(frameon=False, loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "4_loss_per_update.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), sharey=True)
    for ax, g in zip(axes, GROUPS):
        ep = pd.DataFrame(hist[g]["epoch"])
        ax.plot(ep.epoch, ep.train_loss, "o-", color=TRAIN, lw=1.8, ms=5, label="Train (weighted CE − λH)")
        ax.plot(ep.epoch, ep.val_loss, "o-", color=VAL, lw=1.8, ms=5, label="Test group, balanced CE (monitor only)")
        ax.set_ylim(0, 1.2)
        ax.set_xticks(ep.epoch)
        ax.set_xlabel("Epoch")
        ax.set_title(f"Group {g} refit", fontsize=13)
        ax.grid(color=GRID, lw=0.8)
    axes[0].set_ylabel("Loss")
    axes[-1].legend(frameon=False, loc="upper right", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "5_loss_per_epoch.png"), dpi=200)
    plt.close(fig)


def loss_combined(summ, hist):
    """4+5 on one slide (Lucas, 2026-11-20): top row per update, bottom row per epoch, one legend."""
    k = 10
    fig, axes = plt.subplots(2, 5, figsize=(22, 9.2), sharey="row")
    for j, g in enumerate(GROUPS):
        ax = axes[0, j]
        up = np.array(hist[g]["update_train_loss"])
        ax.plot(np.arange(1, len(up) + 1), up, color=TRAIN, lw=0.5, alpha=0.25)
        ax.plot(np.arange(k, len(up) + 1), np.convolve(up, np.ones(k) / k, mode="valid"), color=TRAIN, lw=1.6)
        u, m = zip(*hist[g]["update_monitor"])
        ax.plot(u, m, color=VAL, lw=1.6)
        ax.set_ylim(0, LOSS_YMAX)
        ax.set_xlabel("Gradient update")
        ax.set_title(f"Group {g} refit ({summ[g]['refit_epochs']} epochs)", fontsize=13)
        ax.grid(color=GRID, lw=0.8)

        ax = axes[1, j]
        ep = pd.DataFrame(hist[g]["epoch"])
        ax.plot(ep.epoch, ep.train_loss, "o-", color=TRAIN, lw=1.8, ms=5)
        ax.plot(ep.epoch, ep.val_loss, "o-", color=VAL, lw=1.8, ms=5)
        ax.set_ylim(0, 1.2)
        ax.set_xticks(ep.epoch)
        ax.set_xlabel("Epoch")
        ax.grid(color=GRID, lw=0.8)
    axes[0, 0].set_ylabel("Loss per update")
    axes[1, 0].set_ylabel("Loss per epoch")
    handles = [plt.Line2D([], [], color=TRAIN, lw=2, label="Train: weighted CE − λH (per update: 10-update mean)"),
               plt.Line2D([], [], color=VAL, lw=2, label="Test group: balanced CE (monitor only, not used to choose anything)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "4+5_loss_combined.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 6. prediction for every case
def every_case(preds):
    df = preds.sort_values(["label", "case_id"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 15))
    for i, r in df.iterrows():
        ax.plot([r.label, r.prob_high_grade], [i, i], color=MUTED, lw=0.8, zorder=1)
    ax.scatter(df.label, df.index, s=40, color=TRAIN, zorder=2, label="True label")
    ax.scatter(df.prob_high_grade, df.index, s=36, color=INK, marker="x", linewidth=1.2, zorder=3,
               label="Predicted probability")
    ax.axvline(0.5, color=MUTED, ls="--", lw=1.2, label="0.5 cutoff")
    wrong = (df.prob_high_grade >= 0.5).astype(int) != df.label
    labels = [f"{c} ({g})" for c, g in zip(df.case_id, df.outer_group)]
    ax.set_yticks(df.index, labels, fontsize=7.5)
    for t, w in zip(ax.get_yticklabels(), wrong):
        if w:
            t.set_color(ORANGE)
            t.set_fontweight("bold")
    ax.set_ylim(-1, len(df))
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("True label (0 or 1) vs predicted probability of high-grade")
    ax.set_ylabel("Case (outer test group), sorted by true label")
    ax.set_title("UNI2, nested CV: all 99 test cases")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "6_prediction_every_case.png"), dpi=200)
    plt.close(fig)


def every_case_split(preds):
    """6 split by true label (Lucas, 2026-11-20): benign panel | high-grade panel, side by side like P2's
    KimiaNet | UNI2 layout. Wider and shorter than the 99-row plot, so it fits a 16:9 slide."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10.5))
    for ax, lab, name in [(axes[0], 0, "Benign"), (axes[1], 1, "High-grade")]:
        df = preds[preds.label == lab].sort_values("case_id").reset_index(drop=True)
        wrong = (df.prob_high_grade >= 0.5).astype(int) != df.label
        for i, r in df.iterrows():
            ax.plot([r.label, r.prob_high_grade], [i, i], color=ORANGE if wrong[i] else MUTED,
                    lw=1.4 if wrong[i] else 0.8, zorder=1)
        ax.scatter(df.label, df.index, s=40, color=TRAIN, zorder=2, label="True label")
        ax.scatter(df.prob_high_grade, df.index, s=36, color=INK, marker="x", linewidth=1.2, zorder=3,
                   label="Predicted probability")
        ax.axvline(0.5, color=MUTED, ls="--", lw=1.2, label="0.5 cutoff")
        ax.set_yticks(df.index, [f"{c} ({g})" for c, g in zip(df.case_id, df.outer_group)],
                      fontsize=9 if lab == 0 else 7.5)
        for t, w in zip(ax.get_yticklabels(), wrong):
            if w:
                t.set_color(ORANGE)
                t.set_fontweight("bold")
        ax.set_ylim(-1, len(df))
        ax.invert_yaxis()  # lowest case number at the top, reads like a list
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("Predicted probability of high-grade")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_title(f"{name}: {int((~wrong).sum())} of {len(df)} correct", fontsize=15)
    axes[0].set_ylabel("Case (outer test group)")
    axes[1].legend(frameon=False, loc="lower left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "6_prediction_every_case_split.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 7. selected hyperparameters
def hyperparameters(summ):
    fig, ax = canvas()
    title(ax, "Each outer study chose different settings",
          "Best trial x̂ per outer test group (chosen on inner validation loss only)")
    cols = [("Group", 0.8, "left"), ("Learning\nrate", 3.0, "center"), ("Weight\ndecay", 5.0, "center"),
            ("Entropy\nweight λ", 7.0, "center"), ("Classifier\ndropout", 9.0, "center"),
            ("Projector\ndropout", 11.0, "center"), ("Refit\nepochs", 12.9, "center"), ("Inner\nf(x̂)", 14.6, "center")]
    top, x0, w = 7.1, 0.6, 14.8
    ax.add_patch(plt.Rectangle((x0, top - 1.0), w, 1.0, color=HEAD))
    for name, x, ha in cols:
        ax.text(x, top - 0.5, name, fontsize=14, weight="bold", color=INK, ha=ha, va="center", linespacing=1.2)

    def sci(v):
        if v == 0:
            return "0 (off)"
        e = int(np.floor(np.log10(v)))
        if round(v / 10 ** e, 1) >= 10:  # 9.96e-5 must print as 1.0×10⁻⁴, not 10.0×10⁻⁵
            e += 1
        return f"{v / 10 ** e:.1f}×10{str(e).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"

    y = top - 1.0
    for g in GROUPS:
        y -= 0.66
        s, hp = summ[g], summ[g]["selected_hp"]
        vals = [("learning_rate", sci(hp["learning_rate"]), 3.0), ("weight_decay", sci(hp["weight_decay"]), 5.0),
                ("entropy_weight", sci(hp["entropy_weight"]), 7.0),
                ("classifier_dropout", f"{hp['classifier_dropout']:.2f}", 9.0),
                ("projector_dropout", f"{hp['projector_dropout']:.2f}", 11.0)]
        ax.text(0.8, y + 0.33, f"Group {g}", fontsize=14, color=INK, va="center")
        for key, txt, x in vals:
            if key in s["edge_flags"]:
                ax.add_patch(FancyBboxPatch((x - 0.85, y + 0.08), 1.7, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                                            fc=FLAG_BG, ec=ORANGE, lw=1.2))
            ax.text(x, y + 0.33, txt, fontsize=14, color=INK, ha="center", va="center")
        ax.text(12.9, y + 0.33, str(s["refit_epochs"]), fontsize=14, color=INK, ha="center", va="center")
        ax.text(14.6, y + 0.33, f"{s['selected_f']:.3f}", fontsize=14, color=INK, ha="center", va="center")
        ax.plot([x0, x0 + w], [y, y], color=GRID, lw=1)

    ax.add_patch(FancyBboxPatch((0.6, 1.62), 0.4, 0.3, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=FLAG_BG, ec=ORANGE, lw=1.2))
    ax.text(1.15, 1.77, "Edge of range: entropy weight in B and E is at the bottom of 10⁻³ – 1. Not rerun with a "
                        "wider range (time); the search prefers a small or zero entropy weight.",
            fontsize=13, color=INK, va="center")
    ax.text(0.6, 1.15, "All 40 trials finished in every group. Weight decay stayed far below the 10³ upper end.",
            fontsize=13, color=INK2, va="center")
    ax.text(0.6, 0.7, "Refit epochs = mean of the 4 inner models' best epochs (fixed before seeing the test group).",
            fontsize=13, color=INK2, va="center")
    fig.savefig(os.path.join(OUT, "7_selected_hyperparameters.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    summ, hist, preds = load()
    # guard: the slide numbers must match make_results.py's table (D16) exactly
    assert len(preds) == 99 and preds.case_id.is_unique, "expected 99 unique test cases"
    assert int(((preds.prob_high_grade >= 0.5).astype(int) == preds.label).sum()) == 93
    design(summ)
    results_table(preds)
    confusion(preds)
    loss_curves(summ, hist)
    loss_combined(summ, hist)
    every_case(preds)
    every_case_split(preds)
    hyperparameters(summ)
    print("Saved to", OUT)


if __name__ == "__main__":
    main()
