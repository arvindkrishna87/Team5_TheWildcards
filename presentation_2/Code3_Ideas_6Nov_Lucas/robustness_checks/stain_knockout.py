#!/usr/bin/env python3
"""Stain knockout (exploratory, no retraining): score each test case with stains removed.
Uses the main runs' best.pth, read-only. Usage: python stain_knockout.py [max_cases]"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from common import O, ENC, STAINS, load_index, dataset_for, split_file, main_runs
from models import create_model

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
SETS = {"all": STAINS, "no h&e": ("melan", "sox10"), "no melan": ("h&e", "sox10"), "no sox10": ("h&e", "melan"),
        "h&e only": ("h&e",), "melan only": ("melan",), "sox10 only": ("sox10",)}
patches, s2c = load_index()
runs = main_runs()
rows = []
for (enc, k), run in sorted(runs.items()):
    model = create_model(num_classes=2, embed_dim=512, pooled_dim=ENC[enc][1])
    ckpt = torch.load(os.path.join(run, "checkpoints", "best.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ds = dataset_for(np.load(split_file(k))["test_cases"].tolist(), enc, patches, s2c)
    for i in range(min(len(ds), limit or len(ds))):
        item = ds[i]
        for name, keep in SETS.items():
            sub = {s: item["stain_slices"].get(s, []) for s in keep}
            if not any(sub.values()):
                continue
            with torch.inference_mode():
                p = torch.softmax(model(sub), dim=0)[1].item()
            rows.append(dict(encoder=enc, split=k, case_id=int(item["case_id"]), true_label=int(item["label"]),
                             stains=name, prob_class1=p, correct=(p >= 0.5) == bool(item["label"])))
    print(enc, k, "done", flush=True)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(O, "results", "stain_knockout_predictions.csv"), index=False)
main = pd.read_csv(os.path.join(os.path.dirname(O), "presentation2/results_summary/all_test_predictions.csv"))
chk = out[out.stains == "all"].merge(main, on=["encoder", "split", "case_id"], suffixes=("", "_main"))
print("check vs main predictions: cases", len(chk), "max prob difference %.6f" % (chk.prob_class1 - chk.prob_class1_main).abs().max())
