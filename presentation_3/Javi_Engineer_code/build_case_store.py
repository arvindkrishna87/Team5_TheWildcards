"""
build_case_store.py - build the per-case data structure (one file per case).

Reads the metadata table (patches_<version>.parquet) and, for every usable case, writes
<out_dir>/case_<id>.pt holding that case's metadata AND all its patch embeddings,
organized the way the model uses them:

    {
      "case_id": 1, "patient_id": 1, "label": 1, "grade_raw": 4.0, "fold": "F2",
      "dataset_version": "v1", "emb_version": "kimianet_pool2x2", "emb_dim": 4096,
      "patch_order": "text",
      "case_info": {...},                        # the case's row from the cases table (optional)
      "stains": {
        "h&e": {
          "match_1": {
            "emb":       FloatTensor[N, 4096],   # every patch of this slice, stacked
            "patch_ids": [...],                  # same order as the rows of emb
            "patch_idx": [...],
            "png_paths": [...],                  # relative to --project_root
          },
          ...
        },
        "melan": {...}, "sox10": {...},
      },
    }

Design decisions:
  * Hyperparameters are NOT stored: one dataset is trained under many configs.
  * Training caps (per_slice_cap, max_slices_per_stain) are NOT applied: every usable
    patch is stored; dataset.py applies caps when it samples.
  * PNG paths ARE stored (relative), so attention analysis can find the image.
  * Each stain's slices and each slice's patches are stored in the same order the current
    pipeline uses (--order), so "first 500 patches" and "first 5 slices" pick the same ones.

Also writes <out_dir>/manifest.json (per-case counts, file sizes, anything skipped).
Nothing in the patch or embedding folders is changed: they are only read.

Example:
    python build_case_store.py \
        --patches /home/gyr0436/metadata_table/metadata/patches_v1.parquet \
        --out_dir /home/gyr0436/metadata_table/case_store/v1 \
        --order text \
        --project_root /projects/e32998
"""
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


def natural_key(s: str):
    """'match_10' sorts after 'match_2' (numbers compared as numbers)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def order_rows(rows: pd.DataFrame, order: str) -> pd.DataFrame:
    if order == "text":      # what sorted(os.listdir(...)) gives: patch0, patch1, patch10, patch100 ...
        return rows.sort_values("png_name", kind="stable")
    return rows.sort_values("patch_idx", kind="stable")  # numeric: patch0, patch1, patch2 ...


def order_slices(slices, order: str):
    return sorted(slices) if order == "text" else sorted(slices, key=natural_key)


def rel_path(p: str, root) -> str:
    if root is None:
        return p
    try:
        return os.path.relpath(p, root)
    except ValueError:
        return p


def load_emb(path: str, emb_dim: int) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):  # tolerate {"emb": tensor}-style files
        t = next(v for v in t.values() if torch.is_tensor(v))
    t = t.detach().float().reshape(-1)
    if t.numel() != emb_dim:
        raise ValueError(f"expected {emb_dim} values, got {t.numel()}")
    if torch.isnan(t).any():
        raise ValueError("contains NaN")
    return t


_CASES = None


def case_info_of(case_id, args):
    """The case's row from the cases table (grade, missing stains, patch-size features...), as plain
    Python values. Kept so everything about the case travels with it; see CASES_DATA_DICTIONARY.md."""
    global _CASES
    if not args.cases_table:
        return None
    if _CASES is None:
        _CASES = pd.read_parquet(args.cases_table).set_index("case_id")
    if case_id not in _CASES.index:
        return None
    row = _CASES.loc[case_id]
    return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}


def build_case(case_id, rows, args, meta):
    stains, skipped = {}, []
    for stain in sorted(rows["stain"].unique()):
        s_rows = rows[rows["stain"] == stain]
        stains[stain] = {}
        for slice_id in order_slices(s_rows["slice_id"].unique(), args.order):
            sl = order_rows(s_rows[s_rows["slice_id"] == slice_id], args.order)
            embs, ids, idxs, pngs = [], [], [], []
            for r in sl.itertuples(index=False):
                try:
                    embs.append(load_emb(r.emb_path, args.emb_dim))
                except Exception as e:  # unreadable/bad file: skip it, but record it
                    skipped.append({"patch_id": r.patch_id, "reason": str(e)[:120]})
                    continue
                ids.append(r.patch_id)
                idxs.append(int(r.patch_idx))
                pngs.append(rel_path(r.png_path, args.project_root))
            if embs:
                stains[stain][slice_id] = {
                    "emb": torch.stack(embs),
                    "patch_ids": ids,
                    "patch_idx": idxs,
                    "png_paths": pngs,
                }
        if not stains[stain]:
            del stains[stain]

    record = {
        "case_id": int(case_id),
        "patient_id": int(meta["patient_id"]),
        "label": int(meta["label"]),
        "grade_raw": float(meta["grade_raw"]),
        "fold": None if pd.isna(meta["fold"]) else str(meta["fold"]),
        "dataset_version": args.version,
        "emb_version": str(meta["emb_version"]),
        "emb_dim": args.emb_dim,
        "patch_order": args.order,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "case_info": case_info_of(case_id, args),
        "stains": stains,
    }
    return record, skipped


def main(args):
    t0 = time.time()
    df = pd.read_parquet(args.patches)
    usable = df[df["exclude_reason"].isna() & df["emb_path"].notna()]
    n_no_emb = int((df["exclude_reason"].isna() & df["emb_path"].isna()).sum())
    if args.version is None:
        args.version = str(df["dataset_version"].iloc[0])
    case_ids = sorted(usable["case_id"].dropna().astype(int).unique())
    if args.cases:
        wanted = {int(c) for c in args.cases.split(",")}
        case_ids = [c for c in case_ids if c in wanted]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = {"cases": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    print(f"{len(usable):,} usable patches in {len(case_ids)} cases "
          f"({n_no_emb:,} included patches have no embedding and are left out)")

    for i, cid in enumerate(case_ids, 1):
        f = out / f"case_{cid}.pt"
        if f.exists() and not args.overwrite:
            print(f"[{i}/{len(case_ids)}] case {cid}: exists, skipping (use --overwrite)")
            continue
        rows = usable[usable["case_id"] == cid]
        meta = rows.iloc[0]
        record, skipped = build_case(cid, rows, args, meta)

        tmp = f.with_suffix(".pt.tmp")  # write then rename, so a crash never leaves half a file
        torch.save(record, tmp)
        os.replace(tmp, f)

        counts = {s: {sl: int(v["emb"].shape[0]) for sl, v in d.items()} for s, d in record["stains"].items()}
        n = sum(sum(c.values()) for c in counts.values())
        manifest["cases"][str(cid)] = {
            "file": f.name, "label": record["label"], "fold": record["fold"],
            "patient_id": record["patient_id"], "n_patches": n, "counts": counts,
            "skipped": skipped, "size_mb": round(f.stat().st_size / 1e6, 2),
        }
        print(f"[{i}/{len(case_ids)}] case {cid}: {n:,} patches, "
              f"{len(record['stains'])} stains, {f.stat().st_size / 1e6:.1f} MB"
              + (f", {len(skipped)} skipped" if skipped else ""))

    manifest.update({
        "dataset_version": args.version,
        "source_table": str(Path(args.patches).resolve()),
        "patch_order": args.order,
        "emb_dim": args.emb_dim,
        "project_root": args.project_root,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(manifest["cases"]),
        "n_patches": sum(c["n_patches"] for c in manifest["cases"].values()),
        "total_size_mb": round(sum(c["size_mb"] for c in manifest["cases"].values()), 1),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    n_skipped = sum(len(c["skipped"]) for c in manifest["cases"].values())
    print(f"\n{manifest['n_cases']} case files, {manifest['n_patches']:,} patches, "
          f"{manifest['total_size_mb']:,} MB, {n_skipped} unreadable embeddings skipped "
          f"({time.time() - t0:.0f}s)\nwrote {manifest_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True, help="patches_<version>.parquet from build_metadata.py")
    ap.add_argument("--out_dir", required=True, help="e.g. .../case_store/v1")
    ap.add_argument("--order", choices=["text", "numeric"], required=True,
                    help="patch/slice order: 'text' = sorted filenames (patch0, patch1, patch10...), "
                         "'numeric' = by patch number. Must match data_utils.py.")
    ap.add_argument("--project_root", default=None,
                    help="store PNG paths relative to this folder (e.g. /projects/e32998)")
    ap.add_argument("--version", default=None, help="defaults to the table's dataset_version")
    ap.add_argument("--emb_dim", type=int, default=4096)
    ap.add_argument("--cases", default=None, help="comma-separated case ids, to build only some")
    ap.add_argument("--cases_table", default=None,
                    help="cases_<version>.parquet; its row for each case is stored as 'case_info'")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
