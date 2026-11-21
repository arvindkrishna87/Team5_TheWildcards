"""
build_metadata.py - build the CMIL master metadata table (one row per patch).

Reads the existing files (patch PNGs, embedding .pt files, label CSV, split .npz files)
and writes:
    <out_dir>/patches_<version>.parquet        one row per patch (now with width, height, side_px)
    <out_dir>/cases_<version>.parquet (+.csv)  one row per case: label, fold, stains, missing
                                              stains, and patch features (choice 2): patches
                                              per slice, patch size median/IQR, share per
                                              stain, missing embeddings
    <out_dir>/build_summary_<version>.txt      human-readable summary

Patch size is read from each PNG's header (the image itself is not decoded).
Use --skip_sizes to leave it out.

Nothing in the existing pipeline is changed. Rows are never silently dropped:
anything that can't be used gets an `exclude_reason`.

Example:
    python build_metadata.py \
        --patches_dir /projects/e32998/patches_benign_split \
        --emb_dir /projects/e32998/patches_varsize_pooled4096 \
        --labels_csv extra_benign_case_grade_match.csv \
        --splits_dir no_train_val_leakage_splits \
        --out_dir /projects/e32998/metadata \
        --version v1
"""
import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# case_{id}_{slice}_{stain}_patch{n}.png   e.g. case_12_s3_h&e_patch0045.png
PATCH_RE = re.compile(
    r"^case_(?P<case>\d+)_(?P<slice>.+?)_(?P<stain>h&e|melan|sox10)_patch(?P<idx>\d+)\.png$",
    re.IGNORECASE,
)

# Pseudo-cases created by splitting benign patients (slide 7 of the walkthrough deck)
PATIENT_OF = {107: 22, 108: 22, 109: 26, 110: 26, 111: 26, 118: 24, 119: 25}

# Class 1.0 -> benign (0); Class 3.0 / 4.0 -> high-grade (1); anything else excluded
CLASS_TO_LABEL = {1.0: 0, 3.0: 1, 4.0: 1}

STAIN_NAMES = {"h&e": "h&e", "melan": "melan", "sox10": "sox10"}
ALL_STAINS = ["h&e", "melan", "sox10"]
GRADE_NAME = {0: "Benign", 1: "High-grade"}


def read_size(path):
    """(width, height) from the PNG header only - the pixels are never decoded."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (None, None)


def add_patch_sizes(df, workers):
    """Choice 2, step 1: patch size (width x height in pixels) on every patch row."""
    paths = df["png_path"].fillna("").tolist()
    print(f"reading patch sizes for {len(paths):,} PNGs ({workers} threads) ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        sizes = list(ex.map(lambda p: read_size(p) if p else (None, None), paths, chunksize=256))
    df["width"] = pd.array([s[0] for s in sizes], dtype="Int64")
    df["height"] = pd.array([s[1] for s in sizes], dtype="Int64")
    df["area_px"] = (df["width"] * df["height"]).astype("Int64")
    # one number per patch: side length of a square with the same area
    df["side_px"] = np.sqrt(df["area_px"].astype("float")).round(1)
    n_bad = int(df["width"].isna().sum() - df["png_path"].isna().sum())
    if n_bad:
        print(f"  WARNING: {n_bad} PNGs could not be read")
    return df


def build_case_features(inc, cases, small_px):
    """Choice 2, step 2: one row per case describing its patches."""
    has_size = "side_px" in inc.columns
    g = inc.groupby("case_id")
    per_slice = inc.groupby(["case_id", "stain", "slice_id"]).size()  # patches in each stain/slice
    feat = pd.DataFrame({
        "n_patches": g.size(),
        "n_slices": g["slice_id"].nunique(),            # same definition as the cases table
        "n_stain_slices": per_slice.groupby(level="case_id").size(),
        "n_stains": g["stain"].nunique(),
        "n_missing_emb": g["emb_path"].apply(lambda s: int(s.isna().sum())),
    })
    feat["patches_per_slice"] = (feat["n_patches"] / feat["n_slices"]).round(1)
    feat["max_patches_in_a_slice"] = per_slice.groupby(level="case_id").max()
    for st in ALL_STAINS:
        name = st.replace("&", "")
        sub = inc[inc["stain"] == st].groupby("case_id")
        feat[f"n_{name}"] = sub.size().reindex(feat.index, fill_value=0)
        feat[f"share_{name}"] = (feat[f"n_{name}"] / feat["n_patches"]).round(3)
        feat[f"slices_{name}"] = sub["slice_id"].nunique().reindex(feat.index, fill_value=0)
    feat["missing_stains"] = [
        ",".join(s for s in ALL_STAINS if feat.loc[c, f"n_{s.replace('&', '')}"] == 0) or None
        for c in feat.index]
    feat["pct_missing_emb"] = (100 * feat["n_missing_emb"] / feat["n_patches"]).round(2)
    if has_size:
        side = g["side_px"]
        feat["median_patch_size"] = side.median().round(1)
        q1, q3 = side.quantile(0.25), side.quantile(0.75)
        feat["patch_size_iqr"] = (q3 - q1).round(1)
        feat["min_patch_size"] = side.min()
        feat["max_patch_size"] = side.max()
        small = inc["side_px"] < small_px
        feat[f"n_small_lt{small_px}"] = small.groupby(inc["case_id"]).sum().reindex(feat.index, fill_value=0)
        feat["n_missing_emb_small"] = (small & inc["emb_path"].isna()).groupby(inc["case_id"]).sum() \
            .reindex(feat.index, fill_value=0)
    return feat.reset_index()


# Column order of the cases table: the instructor's example columns first, then the rest.
CASE_COLUMNS = [
    "case_id", "grade", "n_patches", "patches_per_slice", "median_patch_size", "patch_size_iqr",
    "share_he", "share_melan", "share_sox10", "n_missing_emb",
    "patient_id", "label", "grade_raw", "fold", "n_slices", "n_stains", "stains", "missing_stains",
    "n_he", "n_melan", "n_sox10", "slices_he", "slices_melan", "slices_sox10",
    "n_stain_slices", "max_patches_in_a_slice", "pct_missing_emb", "n_missing_emb_small",
    "min_patch_size", "max_patch_size", "n_small_lt32",
]


def add_features_to_cases(cases, feat):
    """Put the patch features into the cases table (one case table, not two)."""
    cases = cases.drop(columns=[c for c in ("n_slices", "n_missing_emb", "missing_stains") if c in cases])
    feat = feat.drop(columns=["n_patches"])  # cases already has it (0 for cases with no patches)
    cases = cases.merge(feat, on="case_id", how="left")
    cases.insert(1, "grade", cases["label"].map(lambda x: GRADE_NAME.get(x) if pd.notna(x) else None))
    for c in cases.columns:  # counts as whole numbers (blank for cases with no patches)
        if c.startswith(("n_", "slices_", "max_patches")) or c in ("patient_id", "label"):
            cases[c] = cases[c].astype("Int64")
    first = [c for c in CASE_COLUMNS if c in cases.columns]
    return cases[first + [c for c in cases.columns if c not in first]]


def normalize_slice(s: str) -> str:
    """'s3' -> 's_3', 's_3' -> 's_3'; anything else is kept as-is."""
    m = re.fullmatch(r"s_?(\d+)", s, re.IGNORECASE)
    return f"s_{int(m.group(1))}" if m else s


def list_files(folder: Path, suffix: str) -> list:
    """Fast, sorted listing (os.scandir is much faster than glob on Quest)."""
    with os.scandir(folder) as it:
        return sorted(e.name for e in it if e.is_file() and e.name.endswith(suffix))


def load_labels(path: Path, case_col: str, class_col: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    print(f"label CSV columns: {list(raw.columns)}  ({len(raw)} rows)")
    df = raw[[case_col, class_col]].rename(columns={case_col: "case_id", class_col: "grade_raw"})
    df["csv_row"] = raw.index + 2  # +2 = header line + 1-based, so it matches what you see in Excel
    df["case_id"] = pd.to_numeric(df["case_id"], errors="coerce")
    df["grade_raw"] = pd.to_numeric(df["grade_raw"], errors="coerce")
    df["label"] = df["grade_raw"].map(CLASS_TO_LABEL)
    df["label_exclude_reason"] = np.where(
        df["case_id"].isna(), "csv: case id not a number",
        np.where(df["label"].isna(), "csv: class not in {1,3,4}", None),
    )
    return df


def load_folds(splits_dir: Path) -> dict:
    """data_splits_new_0k.npz has test fold Fk (see slide 9), so test_cases of split k = fold Fk."""
    fold_of = {}
    for k in range(1, 6):
        f = splits_dir / f"data_splits_new_0{k}.npz"
        if not f.exists():
            print(f"WARNING: {f} not found, folds will be incomplete")
            continue
        test_cases = np.load(f, allow_pickle=True)["test_cases"]
        for c in test_cases:
            c = int(c)
            if c in fold_of:
                print(f"WARNING: case {c} is a test case in more than one split")
            fold_of[c] = f"F{k}"
    return fold_of


def build(args):
    patches_dir = Path(args.patches_dir)
    emb_dir = Path(args.emb_dir)

    print("listing patches ...")
    pngs = list_files(patches_dir, ".png")
    print(f"  {len(pngs):,} PNG files")
    print("listing embeddings ...")
    pts = set(list_files(emb_dir, ".pt")) if emb_dir.exists() else set()
    print(f"  {len(pts):,} .pt files")

    rows, unparsed = [], []
    for name in pngs:
        m = PATCH_RE.match(name)
        if not m:
            unparsed.append(name)
            rows.append(dict(patch_id=Path(name).stem, png_name=name,
                             exclude_reason="filename did not parse"))
            continue
        pt_name = name[:-4] + ".pt"
        case = int(m["case"])
        rows.append(dict(
            patch_id=Path(name).stem,
            case_id=case,
            patient_id=PATIENT_OF.get(case, case),
            slice_raw=m["slice"],
            slice_id=normalize_slice(m["slice"]),
            stain=STAIN_NAMES[m["stain"].lower()],
            patch_idx=int(m["idx"]),
            png_name=name,
            png_path=str(patches_dir / name),
            emb_path=str(emb_dir / pt_name) if pt_name in pts else None,
            exclude_reason=None,
        ))
    df = pd.DataFrame(rows)
    for col in ("case_id", "patient_id", "patch_idx"):  # whole numbers, blank allowed
        df[col] = df[col].astype("Int64")

    # labels
    labels = load_labels(Path(args.labels_csv), args.case_col, args.class_col)
    good_labels = labels[labels["label_exclude_reason"].isna()].drop_duplicates("case_id").copy()
    good_labels["case_id"] = good_labels["case_id"].astype("Int64")
    good_labels["label"] = good_labels["label"].astype("Int64")
    df = df.merge(good_labels[["case_id", "grade_raw", "label"]], on="case_id", how="left")
    df.loc[df["exclude_reason"].isna() & df["label"].isna(), "exclude_reason"] = "case has no valid label"

    # folds
    fold_of = load_folds(Path(args.splits_dir)) if args.splits_dir else {}
    df["fold"] = df["case_id"].map(fold_of)

    # patch sizes (choice 2, step 1)
    if not args.skip_sizes:
        df = add_patch_sizes(df, args.size_workers)

    # bookkeeping
    df["emb_version"] = np.where(df["emb_path"].notna(), args.emb_version, None)
    df["dataset_version"] = args.version
    df["built_at"] = datetime.now().isoformat(timespec="seconds")
    df = df.sort_values(["case_id", "stain", "slice_id", "patch_idx"], na_position="last").reset_index(drop=True)

    # case table
    inc = df[df["exclude_reason"].isna()]
    cases = (inc.groupby("case_id")
             .agg(patient_id=("patient_id", "first"), label=("label", "first"),
                  grade_raw=("grade_raw", "first"), fold=("fold", "first"),
                  n_patches=("patch_id", "size"),
                  n_slices=("slice_id", lambda s: s.nunique()),
                  stains=("stain", lambda s: ",".join(sorted(set(s)))),
                  n_missing_emb=("emb_path", lambda s: int(s.isna().sum())))
             .reset_index())
    # choice 1: which of the three stains each case is missing
    cases["missing_stains"] = cases["stains"].map(
        lambda s: ",".join(x for x in ALL_STAINS if x not in s.split(",")) or None)
    case_features = build_case_features(inc, cases, args.small_px)
    # labeled cases with no patches at all (e.g. the missing high-grade cases, or 112/113)
    no_patch = good_labels[~good_labels["case_id"].isin(cases["case_id"])].copy()
    no_patch = no_patch.assign(patient_id=no_patch["case_id"].map(lambda c: PATIENT_OF.get(int(c), int(c))),
                               n_patches=0, note="labeled but no patches found")
    cases = pd.concat([cases, no_patch[["case_id", "patient_id", "label", "grade_raw", "n_patches", "note"]]],
                      ignore_index=True, sort=False)
    cases = add_features_to_cases(cases, case_features)
    excluded_csv = labels[labels["label_exclude_reason"].notna()]

    # write
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p_path = out / f"patches_{args.version}.parquet"
    c_path = out / f"cases_{args.version}.parquet"
    if p_path.exists() and not args.overwrite:
        raise SystemExit(f"{p_path} exists. Use a new --version (or --overwrite).")
    df.to_parquet(p_path, index=False)
    cases.to_parquet(c_path, index=False)
    cases.to_csv(out / f"cases_{args.version}.csv", index=False)  # opens in Excel
    missing_st = cases[cases["missing_stains"].notna() & (cases["n_patches"] > 0)]

    lines = [
        f"CMIL metadata build {args.version}  ({datetime.now():%Y-%m-%d %H:%M})",
        f"patches dir: {patches_dir}",
        f"embeddings dir: {emb_dir}",
        "",
        f"patches total:        {len(df):,}",
        f"patches included:     {len(inc):,}",
        f"patches excluded:     {df['exclude_reason'].notna().sum():,}",
        f"missing embeddings:   {inc['emb_path'].isna().sum():,}",
        f"cases with patches:   {inc['case_id'].nunique()}",
        f"patients:             {inc['patient_id'].nunique()}",
        f"benign / high-grade cases: "
        f"{(cases.query('n_patches > 0')['label'] == 0).sum()} / {(cases.query('n_patches > 0')['label'] == 1).sum()}",
        "",
        "exclusion reasons:",
        df["exclude_reason"].value_counts().to_string() if df["exclude_reason"].notna().any() else "  none",
        "",
        "label CSV rows excluded:",
        excluded_csv.to_string(index=False) if len(excluded_csv) else "  none",
        "",
        "labeled cases with no patches:",
        ", ".join(str(int(c)) for c in no_patch["case_id"]) or "  none",
        "",
        "cases missing a stain:",
        missing_st[["case_id", "label", "n_slices", "n_patches", "missing_stains"]].to_string(index=False)
        if len(missing_st) else "  none",
        "",
    ]
    if "side_px" in df.columns:
        small = inc["side_px"] < args.small_px
        lines += [
            f"patch size (side, px): median {inc['side_px'].median():.0f}, "
            f"IQR {inc['side_px'].quantile(.25):.0f}-{inc['side_px'].quantile(.75):.0f}, "
            f"min {inc['side_px'].min():.0f}, max {inc['side_px'].max():.0f}",
            f"patches smaller than {args.small_px}px: {int(small.sum()):,}",
            f"missing embeddings that are smaller than {args.small_px}px: "
            f"{int((small & inc['emb_path'].isna()).sum()):,} of {int(inc['emb_path'].isna().sum()):,}",
            "",
        ]
    lines += [
        f"unparsed filenames (first 20 of {len(unparsed)}):",
        "\n".join(unparsed[:20]) or "  none",
    ]
    summary = "\n".join(lines)
    (out / f"build_summary_{args.version}.txt").write_text(summary)
    print("\n" + summary)
    print(f"\nwrote {p_path}\nwrote {c_path} (+ .csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", required=True)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--splits_dir", default=None, help="folder with data_splits_new_01..05.npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--emb_version", default="kimianet_pool2x2")
    ap.add_argument("--case_col", default="Case")
    ap.add_argument("--class_col", default="Class")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_sizes", action="store_true", help="don't read PNG sizes (faster)")
    ap.add_argument("--size_workers", type=int, default=8, help="threads for reading PNG sizes")
    ap.add_argument("--small_px", type=int, default=32, help="stage 2 skips patches smaller than this")
    build(ap.parse_args())