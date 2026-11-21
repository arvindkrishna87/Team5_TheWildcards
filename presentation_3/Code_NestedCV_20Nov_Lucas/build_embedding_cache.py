#!/usr/bin/env python3
"""
build_embedding_cache.py  (Presentation 3 — see DECISIONS.md D8 and D11)

WHY: the base StainBagCasePooledFeatureDataset re-reads every patch's .pt file from
GPFS each time a case is used (~15 ms per file; 10-25 s for a large case). With
nested CV training ~800 models, that disk reading is almost all of the run time.
This script reads every embedding ONCE and saves one cache file, so training can
hold all cases in memory (~0.3 GB in fp16 for 99 cases).

WHAT IS STORED: every patch that has an embedding, for every gradable case, with
NO cap or shuffling applied. Caps and shuffling are applied later, at training
time, by InMemoryCaseDataset (cached_dataset.py), exactly as the base dataset does.

Case / slice / patch ordering follows the base code exactly:
  data_utils.get_all_patch_files -> group_patches_by_slice -> build_slice_to_class_map
  -> build_case_dict (slices sorted by slice_id, patch paths sorted within a slice).
Patches with no embedding (under 32x32 px, skipped when embeddings were made) are left
out and counted, as dataset.py does.

Cache layout (torch.save):
  {
    "meta": {...},
    "cases": {
      case_id: {
        "label": 0 or 1,
        "stains": {stain: [ {"x": fp16 Tensor(P, D), "paths": [png paths, row-aligned]} , ... per slice ]}
      }
    }
  }
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from config import MODEL_CONFIG
from data_utils import (build_case_dict, build_slice_to_class_map, get_all_patch_files,
                        group_patches_by_slice, load_labels)


def feat_path(emb_dir: str, png_path: str) -> str:
    # Same mapping as dataset.StainBagCasePooledFeatureDataset._patch_path_to_feat_path
    return os.path.join(emb_dir, os.path.splitext(os.path.basename(png_path))[0] + ".pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", default="extra_benign_case_grade_match.csv")
    ap.add_argument("--patches_dir", default="/projects/e32998/patch_images")
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=str, default="", help="comma-separated case IDs (for testing); default all")
    ap.add_argument("--threads", type=int, default=16, help="parallel file reads (GPFS is I/O-bound)")
    args = ap.parse_args()

    t0 = time.time()
    labels = load_labels(args.labels_csv)
    files = get_all_patch_files(args.patches_dir)
    patches = group_patches_by_slice(files, args.patches_dir)
    s2c = build_slice_to_class_map(patches, labels)  # only gradable classes get a mapping
    case_dict, label_map = build_case_dict(list(s2c.keys()), patches, s2c)
    if args.cases:
        keep = {int(c) for c in args.cases.split(",")}
        case_dict = {c: v for c, v in case_dict.items() if c in keep}
    print(f"{len(case_dict)} cases to cache")

    stains = list(MODEL_CONFIG["stains"])
    jobs = []  # (case, stain, slice_index, [paths with an embedding])
    missing_total = 0
    for c in sorted(case_dict):
        for stain in stains:
            for s_i, sl in enumerate(case_dict[c].get(stain, []) or []):
                have = [p for p in sl if os.path.isfile(feat_path(args.embeddings_dir, p))]
                missing_total += len(sl) - len(have)
                if not have:
                    raise ValueError(f"case {c} {stain} slice {s_i}: no patch has an embedding")
                jobs.append((c, stain, s_i, have))

    all_paths = [p for *_, ps in jobs for p in ps]
    print(f"reading {len(all_paths)} embeddings ({missing_total} patches have none: under 32 px)")

    def load(p):
        v = torch.load(feat_path(args.embeddings_dir, p), map_location="cpu")
        return v.view(-1).to(torch.float16)

    with ThreadPoolExecutor(args.threads) as ex:
        vecs = list(ex.map(load, all_paths, chunksize=64))
    vec_of = dict(zip(all_paths, vecs))
    dims = {v.numel() for v in vecs}
    assert len(dims) == 1, f"embeddings have different lengths: {dims}"

    cases = {}
    for c, stain, s_i, ps in jobs:
        entry = cases.setdefault(c, {"label": int(label_map[c]), "stains": {s: [] for s in stains}})
        entry["stains"][stain].append({"x": torch.stack([vec_of[p] for p in ps]), "paths": ps})

    meta = {
        "embeddings_dir": args.embeddings_dir, "patches_dir": args.patches_dir,
        "labels_csv": os.path.abspath(args.labels_csv), "dim": dims.pop(),
        "n_cases": len(cases), "n_patches": len(all_paths), "n_missing_embeddings": missing_total,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"meta": meta, "cases": cases}, args.out)
    print(meta)
    print(f"saved {args.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
