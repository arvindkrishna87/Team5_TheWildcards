"""
check_case_store.py - confirm the per-case files match the metadata table exactly.

Checks, for every case file:
  1. every usable case in the table has a file (and no extra files exist)
  2. label, fold and patient match the table
  3. patch counts per stain and slice match the table
  4. every tensor is [N, emb_dim] with no NaNs, and patch_ids line up with its rows
  5. a random sample of rows is identical to the original .pt files

Exits with an error if anything fails.

Example:
    python check_case_store.py \
        --patches /home/gyr0436/metadata_table/metadata/patches_v1.parquet \
        --store /home/gyr0436/metadata_table/case_store/v1 \
        --sample 300
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

errors = []


def fail(msg):
    errors.append(msg)
    print(f"  FAIL  {msg}")


def main(args):
    df = pd.read_parquet(args.patches)
    usable = df[df["exclude_reason"].isna() & df["emb_path"].notna()]
    store = Path(args.store)
    manifest = json.loads((store / "manifest.json").read_text())
    skipped = {s["patch_id"] for c in manifest["cases"].values() for s in c["skipped"]}
    if skipped:
        print(f"  note  {len(skipped)} patches were skipped as unreadable at build time; "
              "they are excluded from the comparison")
        usable = usable[~usable["patch_id"].isin(skipped)]

    expected = set(usable["case_id"].astype(int).unique())
    found = {int(p.stem.split("_")[1]) for p in store.glob("case_*.pt")}
    if expected - found:
        fail(f"cases with no file: {sorted(expected - found)}")
    if found - expected:
        fail(f"files for cases not in the table: {sorted(found - expected)}")
    print(f"1. {len(found & expected)} of {len(expected)} cases have a file")

    rng = random.Random(0)
    sample_pool = []
    for cid in sorted(found & expected):
        rec = torch.load(store / f"case_{cid}.pt", map_location="cpu", weights_only=False)
        rows = usable[usable["case_id"] == cid]
        m = rows.iloc[0]
        if rec["label"] != int(m["label"]) or rec["patient_id"] != int(m["patient_id"]) \
                or (rec["fold"] or None) != (None if pd.isna(m["fold"]) else m["fold"]):
            fail(f"case {cid}: label/patient/fold differ from the table")

        want = rows.groupby(["stain", "slice_id"]).size().to_dict()
        got = {}
        for stain, slices in rec["stains"].items():
            for sl, v in slices.items():
                e = v["emb"]
                got[(stain, sl)] = e.shape[0]
                if e.dim() != 2 or e.shape[1] != args.emb_dim or torch.isnan(e).any():
                    fail(f"case {cid} {stain}/{sl}: bad tensor {tuple(e.shape)}")
                if not (len(v["patch_ids"]) == len(v["patch_idx"]) == len(v["png_paths"]) == e.shape[0]):
                    fail(f"case {cid} {stain}/{sl}: patch_ids do not line up with tensor rows")
                for j, pid in enumerate(v["patch_ids"]):
                    sample_pool.append((cid, stain, sl, j, pid))
        if got != want:
            diff = {k: (want.get(k), got.get(k)) for k in set(want) | set(got) if want.get(k) != got.get(k)}
            fail(f"case {cid}: patch counts differ (table, file): {dict(list(diff.items())[:5])}")
    print("2-4. metadata, counts and tensor shapes checked")

    path_of = dict(zip(usable["patch_id"], usable["emb_path"]))
    picks = rng.sample(sample_pool, min(args.sample, len(sample_pool)))
    cache, bad = {}, 0
    for cid, stain, sl, j, pid in picks:
        if cid not in cache:
            cache = {cid: torch.load(store / f"case_{cid}.pt", map_location="cpu", weights_only=False)}
        stored = cache[cid]["stains"][stain][sl]["emb"][j]
        orig = torch.load(path_of[pid], map_location="cpu")
        if isinstance(orig, dict):
            orig = next(v for v in orig.values() if torch.is_tensor(v))
        if not torch.equal(stored, orig.detach().float().reshape(-1)):
            bad += 1
    if bad:
        fail(f"{bad}/{len(picks)} sampled embeddings differ from the original .pt files")
    print(f"5. {len(picks) - bad}/{len(picks)} sampled embeddings identical to the originals")

    print(f"\n{'FAILED: ' + str(len(errors)) + ' problem(s)' if errors else 'all checks passed'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--emb_dim", type=int, default=4096)
    main(ap.parse_args())
