"""
validate_metadata.py - check the CMIL metadata table before anything trains on it.

Exits with an error (non-zero status) if any hard check fails, so it can sit in front of
training in an sbatch script:  python validate_metadata.py ... && python main.py ...

Example:
    python validate_metadata.py \
        --patches /projects/e32998/metadata/patches_v1.parquet \
        --splits_dir no_train_val_leakage_splits \
        --check_pt 200
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

errors, warnings = [], []


def err(msg):
    errors.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  WARN  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


def main(args):
    df = pd.read_parquet(args.patches)
    inc = df[df["exclude_reason"].isna()]
    print(f"{args.patches}: {len(df):,} rows, {len(inc):,} included\n")

    print("1. table integrity")
    dup = df["patch_id"].duplicated()
    err(f"{dup.sum()} duplicate patch_ids, e.g. {df.loc[dup, 'patch_id'].head(3).tolist()}") if dup.any() \
        else ok("patch_id is unique")
    n_unparsed = (df["exclude_reason"] == "filename did not parse").sum()
    warn(f"{n_unparsed} filenames did not parse (see build summary)") if n_unparsed else ok("every filename parsed")
    bad_stain = ~inc["stain"].isin(["h&e", "melan", "sox10"])
    err(f"{bad_stain.sum()} rows with unknown stain") if bad_stain.any() else ok("stains are h&e / melan / sox10")
    bad_label = ~inc["label"].isin([0, 1])
    err(f"{bad_label.sum()} included rows without a 0/1 label") if bad_label.any() else ok("every included patch has a label")

    print("\n2. embeddings")
    missing = inc[inc["emb_path"].isna()]
    if len(missing):
        by_case = missing.groupby("case_id").size().sort_values(ascending=False)
        top = {int(c): int(n) for c, n in by_case.head(10).items()}
        msg = f"{len(missing):,} included patches have no .pt (case: count {top})"
        warn(msg) if args.allow_missing_emb else err(msg)
    else:
        ok("every included patch has an embedding")
    if args.check_pt:
        import torch
        sample = inc.dropna(subset=["emb_path"]).sample(min(args.check_pt, len(inc)), random_state=0)
        bad = []
        for p in sample["emb_path"]:
            try:
                t = torch.load(p, map_location="cpu")
                if tuple(t.shape) != (args.emb_dim,) or torch.isnan(t.float()).any():
                    bad.append((p, tuple(t.shape)))
            except Exception as e:  # unreadable file
                bad.append((p, str(e)[:60]))
        err(f"{len(bad)}/{len(sample)} sampled .pt files bad, e.g. {bad[:3]}") if bad \
            else ok(f"{len(sample)} sampled .pt files have shape ({args.emb_dim},) and no NaNs")

    print("\n3. folds and leakage")
    in_folds = inc[inc["fold"].notna()]
    no_fold = inc.loc[inc["fold"].isna(), "case_id"].unique()
    warn(f"{len(no_fold)} cases have patches but no fold (not in any split yet): {sorted(map(int, no_fold))}") \
        if len(no_fold) else ok("every included case has a fold")
    per_patient = in_folds.groupby("patient_id")["fold"].nunique()
    leak = per_patient[per_patient > 1]
    err(f"patients in more than one fold: {leak.index.tolist()}") if len(leak) else ok("no patient spans two folds")
    per_case = in_folds.groupby("case_id")["label"].nunique()
    err(f"cases with mixed labels: {per_case[per_case > 1].index.tolist()}") if (per_case > 1).any() \
        else ok("one label per case")

    cases = in_folds.groupby("case_id").agg(fold=("fold", "first"), label=("label", "first"))
    tbl = cases.groupby("fold")["label"].agg(n="size", benign=lambda s: int((s == 0).sum()),
                                             high_grade=lambda s: int((s == 1).sum()))
    tbl["benign_share"] = (tbl["benign"] / tbl["n"]).round(2)
    print(tbl.to_string())
    spread = tbl["benign_share"].max() - tbl["benign_share"].min()
    warn(f"benign share differs by {spread:.2f} across folds") if spread > 0.10 else ok("class balance similar across folds")

    if args.splits_dir:
        print("\n4. table reproduces the existing split files")
        fold_cases = {f: set(g.index) for f, g in cases.groupby("fold")}
        for k in range(1, 6):
            f = Path(args.splits_dir) / f"data_splits_new_0{k}.npz"
            if not f.exists():
                warn(f"{f.name} not found")
                continue
            z = np.load(f, allow_pickle=True)
            test_f, val_f = f"F{k}", f"F{(k - 2) % 5 + 1}"  # split k: test Fk, val F(k-1)
            train_f = [f"F{i}" for i in range(1, 6) if f"F{i}" not in (test_f, val_f)]
            expect = {
                "test_cases": fold_cases.get(test_f, set()),
                "val_cases": fold_cases.get(val_f, set()),
                "train_cases": set().union(*(fold_cases.get(x, set()) for x in train_f)),
            }
            mismatch = False
            for key, exp in expect.items():
                got = set(int(c) for c in z[key])
                if got != exp:
                    mismatch = True
                    err(f"{f.name} {key}: only in file {sorted(got - exp)}, only in table {sorted(exp - got)}")
            if not mismatch:
                ok(f"{f.name} matches (train {train_f}, val {val_f}, test {test_f})")

    print("\n5. patch counts (look for outliers)")
    counts = inc.groupby(["case_id", "stain"]).size()
    print(counts.describe().round(1).to_string())
    q1, q3 = counts.quantile([0.25, 0.75])
    lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
    out = counts[(counts < max(lo, 1)) | (counts > hi)]
    warn(f"{len(out)} case/stain groups with unusual patch counts: {out.head(10).to_dict()}") if len(out) \
        else ok("no extreme patch-count outliers")
    stains = inc.groupby("case_id")["stain"].nunique()
    if (stains < 3).any():
        warn(f"{(stains < 3).sum()} cases are missing at least one stain: "
             f"{sorted(int(c) for c in stains[stains < 3].index)}")

    if "side_px" in df.columns:
        print("\n6. patch sizes")
        unread = inc["png_path"].notna() & inc["width"].isna()
        err(f"{int(unread.sum())} PNGs could not be read") if unread.any() else ok("every PNG's size was read")
        small = inc["side_px"] < args.small_px
        miss = inc["emb_path"].isna()
        if miss.any():
            ok(f"{int((small & miss).sum()):,} of {int(miss.sum()):,} missing embeddings are patches "
               f"smaller than {args.small_px}px (stage 2 skips those on purpose)")
        big_missing = (~small & miss).sum()
        if big_missing:
            warn(f"{int(big_missing):,} normal-sized patches have no embedding "
                 f"(case: count {{{', '.join(f'{int(c)}: {int(n)}' for c, n in inc[~small & miss].groupby('case_id').size().sort_values(ascending=False).head(10).items())}}})")

    print(f"\n{len(errors)} failed, {len(warnings)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True, help="patches_<version>.parquet")
    ap.add_argument("--splits_dir", default=None, help="compare against data_splits_new_01..05.npz")
    ap.add_argument("--check_pt", type=int, default=0, help="open N random .pt files to check shape")
    ap.add_argument("--emb_dim", type=int, default=4096)
    ap.add_argument("--allow_missing_emb", action="store_true", help="treat missing embeddings as a warning")
    ap.add_argument("--small_px", type=int, default=32)
    main(ap.parse_args())