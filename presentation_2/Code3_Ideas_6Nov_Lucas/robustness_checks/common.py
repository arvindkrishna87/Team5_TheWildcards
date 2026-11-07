"""Shared helpers: load cases exactly as main.py does for validation and test (sorted, no shuffling)."""
import os
import sys
import glob
import re
import numpy as np

O = os.path.expanduser("~/STAT390_coder/robustness_checks")
P2 = os.path.expanduser("~/STAT390_coder/presentation2")
sys.path.insert(0, os.path.join(O, "code"))

from config import DATA_PATHS
from data_utils import (load_labels, get_all_patch_files, group_patches_by_slice,
                        build_slice_to_class_map, build_case_dict)
from dataset import StainBagCasePooledFeatureDataset

ENC = {"kimianet": ("/projects/e32998/patches_varsize_pooled4096", 4096),
       "uni2": ("/projects/e32998/STAT390_Fall2026/Team5_TheWildcards/patch_embeddings_UNI2", 1536)}
STAINS = ("h&e", "melan", "sox10")


def load_index():
    labels = load_labels()
    patches = group_patches_by_slice(get_all_patch_files(), DATA_PATHS["patches_dir"])
    return patches, build_slice_to_class_map(patches, labels)


def dataset_for(cases, enc, patches, s2c):
    cases = set(int(c) for c in cases)
    case_dict, label_map = build_case_dict([k for k in s2c if k[0] in cases], patches, s2c)
    return StainBagCasePooledFeatureDataset(case_dict, label_map, embeddings_dir=ENC[enc][0], per_slice_cap=500,
                                            max_slices_per_stain=5, shuffle_patches=False)


def split_file(k):
    return os.path.join(P2, "splits", f"data_splits_new_0{k}.npz")


def main_runs():
    """{(encoder, split): run_dir} for the main Presentation 2 runs."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(P2, "logs", "P2_*_s?_*.out"))):
        m = re.search(r"P2_(kimianet|uni2)_s(\d)_", path)
        text = open(path).read()
        if m and "Training completed successfully" in text:
            runs[(m.group(1), int(m.group(2)))] = re.search(r"Created run directory: (\S+)", text).group(1)
    return runs
