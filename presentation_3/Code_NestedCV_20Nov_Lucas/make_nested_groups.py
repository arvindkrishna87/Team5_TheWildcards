#!/usr/bin/env python3
"""
make_nested_groups.py  (Presentation 3, nested cross-validation — see DECISIONS.md D7)

Splits the 99 gradable cases into 5 groups (A-E). The same 5 groups are used for
both loops of nested CV (slides 25-27): each group is the outer test set once, and
the other 4 rotate as inner validation folds. So there is ONE partition, not
separate train/val/test splits.

Rules, and where each comes from:
  * 5 groups of 20/20/20/20/19 cases ............ slide 5
  * cases from the same patient stay together .... base code make_splits.py
    (GROUPED_CASES in config.py) and slide 39; prevents a patient being in
    both training and testing
  * equal grade mix in every group ................ base code make_splits.py
    (StratifiedGroupKFold); 35 benign / 64 high-grade -> 7 benign per group
  * seed 436 ...................................... our choice (same as P2),
    fixed before seeing any results; the slides do not name a seed

The seed relabels patient-group IDs; StratifiedGroupKFold keeps shuffle=False.
This is the same approach as the base make_splits.py (our P2 fix: shuffle=True
distorted the class balance).

Outputs (in --out_dir):
  nested_groups_seed<seed>.csv  one row per case: case_id, label, grade, group
  nested_groups_seed<seed>.npz  group_A ... group_E arrays of case IDs
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import GROUPED_CASES

VALID_CLASSES = (1, 3, 4)  # 1 = benign, 3/4 = high-grade (as in config.VALID_CLASSES)
GROUP_NAMES = ["A", "B", "C", "D", "E"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", default="extra_benign_case_grade_match.csv")
    ap.add_argument("--patches_dir", default="/projects/e32998/patch_images")
    ap.add_argument("--seed", type=int, default=436)
    ap.add_argument("--out_dir", default="../splits")
    args = ap.parse_args()

    # Cases = every case ID with patches in patches_dir and a gradable label.
    ids = sorted({int(f.split("_")[1]) for f in os.listdir(args.patches_dir) if f.endswith(".png")})
    lab = pd.read_csv(args.labels_csv)
    cls = dict(zip(lab.Case, lab.Class))
    dropped = {i: cls.get(i) for i in ids if cls.get(i) not in VALID_CLASSES}
    ids = [i for i in ids if cls.get(i) in VALID_CLASSES]
    y = np.array([0 if cls[i] == 1 else 1 for i in ids])  # 0 benign, 1 high-grade
    print(f"cases with patches and a gradable label: {len(ids)} "
          f"(benign {int((y == 0).sum())}, high-grade {int(y.sum())})")
    print(f"left out (class not in {VALID_CLASSES}): {dropped}")

    # Patient groups: related cases share the first case's ID.
    patient = {i: i for i in ids}
    for grp in GROUPED_CASES:
        for c in grp:
            if c in patient:
                patient[c] = grp[0]
    uniq = sorted(set(patient.values()))
    relabel = dict(zip(uniq, np.random.RandomState(args.seed).permutation(len(uniq))))
    groups = np.array([relabel[patient[i]] for i in ids])

    assign = {}
    sgkf = StratifiedGroupKFold(n_splits=5)  # shuffle=False on purpose (see docstring)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(ids)), y, groups)):
        for j in te:
            assign[ids[j]] = GROUP_NAMES[k]

    df = pd.DataFrame({
        "case_id": ids,
        "label": y,
        "grade": ["benign" if v == 0 else "high-grade" for v in y],
        "group": [assign[i] for i in ids],
    })

    # Checks: every case once; patients not split; counts per group.
    assert len(df) == df.case_id.nunique() == len(assign)
    for grp in GROUPED_CASES:
        present = [c for c in grp if c in assign]
        assert len({assign[c] for c in present}) <= 1, f"patient cases {grp} split across groups"

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"nested_groups_seed{args.seed}")
    df.to_csv(base + ".csv", index=False)
    np.savez(base + ".npz", **{f"group_{g}": np.array(sorted(df.case_id[df.group == g])) for g in GROUP_NAMES})

    print(df.groupby("group").grade.value_counts().unstack().assign(total=lambda t: t.sum(axis=1)))
    for g in GROUP_NAMES:
        print(f"group {g}: {sorted(df.case_id[df.group == g])}")
    print(f"saved {base}.csv and {base}.npz")


if __name__ == "__main__":
    main()
