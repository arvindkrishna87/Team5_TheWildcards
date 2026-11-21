"""
cached_dataset.py  (Presentation 3 — see DECISIONS.md D11)

InMemoryCaseDataset: same output as dataset.StainBagCasePooledFeatureDataset, but
reads from the cache made by build_embedding_cache.py instead of one .pt file per patch.

WHY: disk reading was almost all of the training time (D8). Holding every case in
memory makes one training step ~14-42 ms on CPU (D9), so nested CV becomes feasible.

Behaviour copied from dataset.py, in the same order, so the model sees the same data:
  1. keep only the first max_slices_per_stain slices of each stain (if set)
  2. patches with no embedding are already left out (done when the cache was built)
  3. if shuffle_patches: shuffle the slice's patches (a NEW draw every time the case is used)
  4. keep the first per_slice_cap patches
Returns {"case_id", "stain_slices", "stain_paths", "label"} like the base dataset, with
stain_paths row-aligned to stain_slices (needed for attention analysis).
"""
import random
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import Dataset

from config import MODEL_CONFIG


def load_cache(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


class InMemoryCaseDataset(Dataset):
    def __init__(
        self,
        cache: Dict[str, Any],
        case_ids: Iterable[int],
        per_slice_cap: Optional[int] = None,
        max_slices_per_stain: Optional[int] = None,
        shuffle_patches: bool = True,
        feature_dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        self.cases = cache["cases"]
        self.case_ids = sorted(int(c) for c in case_ids)
        missing = [c for c in self.case_ids if c not in self.cases]
        if missing:
            raise KeyError(f"cases not in cache: {missing}")
        self.stains = list(MODEL_CONFIG["stains"])
        self.per_slice_cap = per_slice_cap
        self.max_slices_per_stain = max_slices_per_stain
        self.shuffle_patches = shuffle_patches
        self.feature_dtype = feature_dtype
        self.rng = random.Random(seed)  # own generator: reproducible, independent of other code

    def __len__(self) -> int:
        return len(self.case_ids)

    def label_counts(self) -> Dict[int, int]:
        out = {0: 0, 1: 0}
        for c in self.case_ids:
            out[self.cases[c]["label"]] += 1
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        c = self.case_ids[idx]
        entry = self.cases[c]
        stain_slices, stain_paths = {}, {}
        for stain in self.stains:
            slices = entry["stains"].get(stain, [])
            if self.max_slices_per_stain is not None:
                slices = slices[: self.max_slices_per_stain]
            xs, ps = [], []
            for sl in slices:
                order = list(range(len(sl["paths"])))
                if self.shuffle_patches:
                    self.rng.shuffle(order)
                if self.per_slice_cap:
                    order = order[: self.per_slice_cap]
                idx_t = torch.tensor(order, dtype=torch.long)
                xs.append(sl["x"].index_select(0, idx_t).to(self.feature_dtype))
                ps.append([sl["paths"][i] for i in order])
            stain_slices[stain] = xs
            stain_paths[stain] = ps
        return {"case_id": c, "stain_slices": stain_slices, "stain_paths": stain_paths,
                "label": torch.tensor(entry["label"], dtype=torch.long)}
