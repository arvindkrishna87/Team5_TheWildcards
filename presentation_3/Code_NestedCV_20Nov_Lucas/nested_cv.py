#!/usr/bin/env python3
"""
nested_cv.py  (Presentation 3, Coders: nested cross-validation — slides 25-33)
Decision log: ~/STAT390_coder/presentation3/DECISIONS.md (D13 describes this script).

One run = one OUTER test group (A-E). It
  1. holds the outer group out (its labels are never read until step 5),
  2. runs an Optuna study on the other 4 groups; each trial trains 4 inner models
     (train on 3 groups, validate on the 4th) and returns f(x) = mean of the 4
     best balanced validation losses                                     (slide 25)
  3. selects x_hat = the trial with the lowest f(x)                       (slide 27)
  4. refits one model on all 4 groups with x_hat, for the mean of the inner
     models' best epoch counts                                            (slide 27)
  5. tests once on the outer group and saves each case's probability      (slide 27)

Search space (slide 28): lr 1e-5..1e-3 log; weight decay {0} or 1e-5..1e3 log;
classifier dropout 0..0.5; entropy weight {0} or 1e-3..1 log; projector dropout 0..0.5.
"Test zero separately" = an on/off switch, then a log-scale value only when on.
Losses (slide 30): training = class-weighted CE, reduction="sum", weights from that
training split's own counts, minus lambda_ent * H; validation = balanced CE (mean of
the benign mean CE and the high-grade mean CE), no entropy, used for selection,
the LR scheduler and early stopping.
Optuna (slide 31): 16 scrambled Sobol startup points, then TPE with good-set size
max(6, ceil(0.15 n)).
"""
import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from cached_dataset import InMemoryCaseDataset, load_cache
from config import MODEL_CONFIG, TRAINING_CONFIG
from models import HierarchicalAttnMIL

GROUPS = ["A", "B", "C", "D", "E"]

# Fixed (not tuned) training settings, taken from the base config.py so only the
# five slide-28 hyperparameters change between trials.
FIXED = {
    "max_epochs": TRAINING_CONFIG["epochs"],                          # 30
    "early_stopping_patience": TRAINING_CONFIG["early_stopping_patience"],      # 8
    "early_stopping_min_delta": TRAINING_CONFIG["early_stopping_min_delta"],    # 0.001
    "early_stopping_min_epochs": TRAINING_CONFIG["early_stopping_min_epochs"],  # 10
    "scheduler_factor": TRAINING_CONFIG["scheduler_factor"],          # 0.2139
    "scheduler_patience": TRAINING_CONFIG["scheduler_patience"],      # 4
    "scheduler_min_lr": TRAINING_CONFIG["scheduler_min_lr"],          # 1e-6
    "entropy_eps": TRAINING_CONFIG["patch_entropy_eps"],              # 1e-8
    "per_slice_cap": MODEL_CONFIG["per_slice_cap"],                   # 800 (D12)
    "max_slices_per_stain": MODEL_CONFIG["max_slices_per_stain"],     # None (D12)
    "embed_dim": MODEL_CONFIG["embed_dim"],                           # 512
}

# Search space (slide 28). (low, high) for the log-scale parts.
LR_RANGE = (1e-5, 1e-3)
WD_RANGE = (1e-5, 1e3)
ENT_RANGE = (1e-3, 1.0)
DROPOUT_RANGE = (0.0, 0.5)


# --------------------------------------------------------------------------- data
_CACHE = None  # one copy per process


def get_cache(path: str):
    global _CACHE
    if _CACHE is None:
        _CACHE = load_cache(path)
    return _CACHE


def read_groups(groups_csv: str) -> Dict[str, List[int]]:
    import pandas as pd
    df = pd.read_csv(groups_csv)
    return {g: sorted(df.case_id[df.group == g].astype(int).tolist()) for g in GROUPS}


# --------------------------------------------------------------------------- losses
def patch_attention_entropy(attn: Dict, eps: float) -> torch.Tensor:
    """Mean normalised entropy of every slice's patch-attention vector.
    Same formula as base trainer._compute_patch_attention_entropy (normalize=True)."""
    ents = []
    for stain_dict in attn.get("stain_weights", {}).values():
        for w in stain_dict.get("patch_weights", []):
            if w is None or w.numel() == 0:
                continue
            p = w.clamp_min(eps)
            h = -(p * torch.log(p)).sum()
            if p.numel() > 1:
                h = h / math.log(p.numel())
            ents.append(h)
    return torch.stack(ents).mean() if ents else torch.tensor(0.0)


def class_weights(ds: InMemoryCaseDataset) -> torch.Tensor:
    """slide 30: w_c = N_train / (2 n_c,train), from THIS training split's counts."""
    n = ds.label_counts()
    total = n[0] + n[1]
    return torch.tensor([total / (2 * n[0]), total / (2 * n[1])], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, ds: InMemoryCaseDataset):
    """Balanced CE (slide 30) + per-case probabilities. No entropy term, eval mode."""
    model.eval()
    ce = {0: [], 1: []}
    rows = []
    for i in range(len(ds)):
        item = ds[i]
        logits = model(item["stain_slices"]).unsqueeze(0)
        y = int(item["label"])
        ce[y].append(nn.functional.cross_entropy(logits, torch.tensor([y])).item())
        p_hg = torch.softmax(logits, dim=1)[0, 1].item()
        rows.append({"case_id": item["case_id"], "label": y, "prob_high_grade": p_hg})
    means = [float(np.mean(ce[c])) for c in (0, 1) if ce[c]]
    return float(np.mean(means)), rows


# --------------------------------------------------------------------------- training
def train_model(cache_path: str, train_ids: List[int], val_ids: List[int], hp: Dict, seed: int,
                fixed_epochs: Optional[int] = None, monitor_every: int = 0) -> Dict:
    """Train one model.
    Inner models: fixed_epochs=None -> early stopping + plateau scheduler on the balanced
      validation loss; weights restored to the best epoch.
    Refit: fixed_epochs=E -> train exactly E epochs on all 4 groups, no early stopping and
      no scheduler (there is no validation set to drive them; see D13). If monitor_every>0,
      the loss on val_ids (the outer test group) is RECORDED for the slide-32 plots only;
      it never changes training.
    """
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    cache = get_cache(cache_path)
    tr = InMemoryCaseDataset(cache, train_ids, FIXED["per_slice_cap"], FIXED["max_slices_per_stain"],
                             shuffle_patches=True, seed=seed)
    va = InMemoryCaseDataset(cache, val_ids, FIXED["per_slice_cap"], FIXED["max_slices_per_stain"],
                             shuffle_patches=False)
    dim = next(iter(cache["cases"].values()))["stains"]
    dim = next(sl["x"].shape[1] for s in dim.values() for sl in s)

    model = HierarchicalAttnMIL(embed_dim=FIXED["embed_dim"], pooled_dim=dim,
                                projector_dropout=hp["projector_dropout"],
                                classifier_dropout=hp["classifier_dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    # reduction="sum": with one case per batch, "mean" divides the weight back out (slide 33).
    crit = nn.CrossEntropyLoss(weight=class_weights(tr), reduction="sum")
    lam = hp["entropy_weight"]
    sched = None
    if fixed_epochs is None:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=FIXED["scheduler_factor"],
            patience=FIXED["scheduler_patience"], min_lr=FIXED["scheduler_min_lr"])

    n_epochs = fixed_epochs if fixed_epochs is not None else FIXED["max_epochs"]
    hist = {"update_train_loss": [], "update_monitor": [], "epoch": []}
    best, best_epoch, best_state, bad_epochs = float("inf"), 0, None, 0
    order_rng = random.Random(seed + 1)
    update = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = list(range(len(tr)))
        order_rng.shuffle(order)
        ep_losses = []
        for i in order:
            item = tr[i]
            y = item["label"].view(1)
            if lam > 0:
                logits, attn = model(item["stain_slices"], return_attn_weights=True)
                loss = crit(logits.unsqueeze(0), y) - lam * patch_attention_entropy(attn, FIXED["entropy_eps"])
            else:
                loss = crit(model(item["stain_slices"]).unsqueeze(0), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update += 1
            ep_losses.append(loss.item())
            hist["update_train_loss"].append(loss.item())
            if monitor_every and update % monitor_every == 0:
                m, _ = evaluate(model, va)
                hist["update_monitor"].append((update, m))
                model.train()
        val_loss, _ = evaluate(model, va)
        hist["epoch"].append({"epoch": epoch, "train_loss": float(np.mean(ep_losses)),
                              "val_loss": val_loss, "lr": opt.param_groups[0]["lr"]})
        if fixed_epochs is not None:
            continue
        sched.step(val_loss)
        if val_loss < best - FIXED["early_stopping_min_delta"]:
            best, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if epoch >= FIXED["early_stopping_min_epochs"] and bad_epochs >= FIXED["early_stopping_patience"]:
            break

    out = {"history": hist, "class_weights": class_weights(tr).tolist()}
    if fixed_epochs is None:
        out.update(best_val_loss=best, best_epoch=best_epoch)
    else:
        test_loss, rows = evaluate(model, va)
        out.update(final_monitor_loss=test_loss, predictions=rows)
    return out


def _inner_job(args):
    cache_path, train_ids, val_ids, hp, seed, threads = args
    torch.set_num_threads(threads)
    r = train_model(cache_path, train_ids, val_ids, hp, seed)
    r.pop("history")  # inner histories are not needed for the slides; keep results small
    return r


# --------------------------------------------------------------------------- Optuna
def sobol_startup(n: int, seed: int) -> List[Dict]:
    """16 scrambled Sobol points (slide 31) over 7 dims: 5 hyperparameters plus the two
    on/off switches for 'zero tested separately'. Each value on its own scale."""
    from scipy.stats import qmc
    u = qmc.Sobol(d=7, scramble=True, seed=seed).random(n)
    pts = []
    for r in u:
        p = {"learning_rate": 10 ** (math.log10(LR_RANGE[0]) + r[0] * (math.log10(LR_RANGE[1]) - math.log10(LR_RANGE[0]))),
             "weight_decay_on": bool(r[1] >= 0.5),
             "classifier_dropout": DROPOUT_RANGE[0] + r[3] * (DROPOUT_RANGE[1] - DROPOUT_RANGE[0]),
             "entropy_on": bool(r[4] >= 0.5),
             "projector_dropout": DROPOUT_RANGE[0] + r[6] * (DROPOUT_RANGE[1] - DROPOUT_RANGE[0])}
        if p["weight_decay_on"]:
            p["weight_decay"] = 10 ** (math.log10(WD_RANGE[0]) + r[2] * (math.log10(WD_RANGE[1]) - math.log10(WD_RANGE[0])))
        if p["entropy_on"]:
            p["entropy_weight"] = 10 ** (math.log10(ENT_RANGE[0]) + r[5] * (math.log10(ENT_RANGE[1]) - math.log10(ENT_RANGE[0])))
        pts.append(p)
    return pts


def suggest(trial) -> Dict:
    hp = {"learning_rate": trial.suggest_float("learning_rate", *LR_RANGE, log=True)}
    hp["weight_decay"] = (trial.suggest_float("weight_decay", *WD_RANGE, log=True)
                          if trial.suggest_categorical("weight_decay_on", [False, True]) else 0.0)
    hp["classifier_dropout"] = trial.suggest_float("classifier_dropout", *DROPOUT_RANGE)
    hp["entropy_weight"] = (trial.suggest_float("entropy_weight", *ENT_RANGE, log=True)
                            if trial.suggest_categorical("entropy_on", [False, True]) else 0.0)
    hp["projector_dropout"] = trial.suggest_float("projector_dropout", *DROPOUT_RANGE)
    return hp


def good_set_size(n: int) -> int:
    """slide 31: n_good = max(6, 0.15 n), rounded up, never more than n."""
    return min(n, max(6, math.ceil(0.15 * n)))


def edge_flags(hp: Dict) -> List[str]:
    """slide 27: flag a selected value in the outer 5% of its range (log scale where used)."""
    flags = []
    def near(v, lo, hi, log):
        a, b, x = (math.log10(lo), math.log10(hi), math.log10(v)) if log else (lo, hi, v)
        return (x - a) / (b - a) < 0.05 or (b - x) / (b - a) < 0.05
    if near(hp["learning_rate"], *LR_RANGE, True): flags.append("learning_rate")
    if hp["weight_decay"] > 0 and near(hp["weight_decay"], *WD_RANGE, True): flags.append("weight_decay")
    if hp["entropy_weight"] > 0 and near(hp["entropy_weight"], *ENT_RANGE, True): flags.append("entropy_weight")
    if near(hp["classifier_dropout"], *DROPOUT_RANGE, False): flags.append("classifier_dropout")
    if near(hp["projector_dropout"], *DROPOUT_RANGE, False): flags.append("projector_dropout")
    return flags


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", required=True, choices=GROUPS)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--groups_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_trials", type=int, default=40)          # slide 31
    ap.add_argument("--n_startup", type=int, default=16)         # slide 31
    ap.add_argument("--seed", type=int, default=436)
    ap.add_argument("--workers", type=int, default=4, help="inner models trained in parallel")
    ap.add_argument("--threads_per_worker", type=int, default=4)
    ap.add_argument("--monitor_every", type=int, default=10, help="refit: record test-group loss every N updates (plots only)")
    args = ap.parse_args()

    import optuna
    from optuna.samplers import TPESampler

    out = os.path.join(args.out_dir, f"outer_{args.outer}")
    os.makedirs(out, exist_ok=True)
    groups = read_groups(args.groups_csv)
    test_ids = groups[args.outer]
    inner_groups = [g for g in GROUPS if g != args.outer]
    tuning_ids = sorted(c for g in inner_groups for c in groups[g])
    assert not set(test_ids) & set(tuning_ids)
    outer_idx = GROUPS.index(args.outer)

    # Study: one per outer fold, stored on disk so a timed-out job can resume.
    storage = f"sqlite:///{os.path.join(out, 'study.db')}"
    sampler = TPESampler(n_startup_trials=args.n_startup, gamma=good_set_size,
                         seed=args.seed + outer_idx)
    study = optuna.create_study(study_name=f"nested_outer_{args.outer}", storage=storage,
                                direction="minimize", sampler=sampler, load_if_exists=True)
    if len(study.trials) == 0:
        for p in sobol_startup(args.n_startup, seed=args.seed + outer_idx):
            study.enqueue_trial(p)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers)
    log_path = os.path.join(out, "inner_results.jsonl")

    def objective(trial):
        hp = suggest(trial)
        jobs = []
        for k, vg in enumerate(inner_groups):  # inner 4-fold rotation (slide 26)
            tr_ids = sorted(c for g in inner_groups if g != vg for c in groups[g])
            seed = args.seed * 1000 + outer_idx * 100 + trial.number * 4 + k
            jobs.append((args.cache, tr_ids, groups[vg], hp, seed, args.threads_per_worker))
        t0 = time.time()
        res = pool.map(_inner_job, jobs)
        f = float(np.mean([r["best_val_loss"] for r in res]))
        rec = {"trial": trial.number, "hp": hp, "f": f, "seconds": round(time.time() - t0, 1),
               "inner": [{"val_group": vg, "best_val_loss": r["best_val_loss"],
                          "best_epoch": r["best_epoch"], "class_weights": r["class_weights"]}
                         for vg, r in zip(inner_groups, res)]}
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        trial.set_user_attr("hp", hp)
        trial.set_user_attr("best_epochs", [r["best_epoch"] for r in res])
        print(f"[outer {args.outer}] trial {trial.number}: f={f:.4f} "
              f"epochs={[r['best_epoch'] for r in res]} ({rec['seconds']} s) hp={hp}", flush=True)
        return f

    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done < args.n_trials:
        study.optimize(objective, n_trials=args.n_trials - done)
    pool.close(); pool.join()
    study.trials_dataframe().to_csv(os.path.join(out, "trials.csv"), index=False)

    # Select (slide 27) and refit for the mean inner best-epoch count.
    best = study.best_trial
    hp = best.user_attrs["hp"]
    refit_epochs = max(1, int(round(float(np.mean(best.user_attrs["best_epochs"])))))
    torch.set_num_threads(args.workers * args.threads_per_worker)
    r = train_model(args.cache, tuning_ids, test_ids, hp, seed=args.seed * 1000 + outer_idx * 100 + 99,
                    fixed_epochs=refit_epochs, monitor_every=args.monitor_every)

    # Test once (slide 27): predictions after the final refit epoch.
    import pandas as pd
    preds = pd.DataFrame(r["predictions"]).assign(outer_group=args.outer,
                                                   pred=lambda d: (d.prob_high_grade >= 0.5).astype(int))
    preds.to_csv(os.path.join(out, "test_predictions.csv"), index=False)
    acc = float((preds.pred == preds.label).mean())
    rec_b = float((preds.pred[preds.label == 0] == 0).mean())
    rec_h = float((preds.pred[preds.label == 1] == 1).mean())
    summary = {"outer": args.outer, "selected_trial": best.number, "selected_f": best.value,
               "selected_hp": hp, "edge_flags": edge_flags(hp), "inner_best_epochs": best.user_attrs["best_epochs"],
               # count only COMPLETE trials: enqueued Sobol points not yet run also sit in study.trials
               "refit_epochs": refit_epochs,
               "n_trials_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
               "n_test": len(preds), "accuracy": acc, "benign_recall": rec_b, "high_grade_recall": rec_h,
               "refit_class_weights": r["class_weights"]}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    json.dump(r["history"], open(os.path.join(out, "refit_history.json"), "w"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
