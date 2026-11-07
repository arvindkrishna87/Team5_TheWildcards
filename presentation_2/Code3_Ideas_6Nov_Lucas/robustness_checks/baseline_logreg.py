#!/usr/bin/env python3
"""Baseline without attention: average each stain's patch features, then logistic regression.
The regularization strength is chosen on each split's validation cases only."""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from common import O, ENC, STAINS, load_index, dataset_for, split_file

patches, s2c = load_index()
all_cases = sorted({k[0] for k in s2c})
rows = []
for enc in ENC:
    ds = dataset_for(all_cases, enc, patches, s2c)
    X, y = {}, {}
    for i in range(len(ds)):
        item = ds[i]
        parts = []
        for stain in STAINS:
            sl = item["stain_slices"].get(stain, [])
            parts.append(np.concatenate([t.numpy() for t in sl]).mean(0) if sl else np.zeros(ENC[enc][1], np.float32))
        X[int(item["case_id"])] = np.concatenate(parts)
        y[int(item["case_id"])] = int(item["label"])
    for k in range(1, 6):
        d = np.load(split_file(k))
        tr, va, te = (d[n].tolist() for n in ("train_cases", "val_cases", "test_cases"))
        get = lambda ids: (np.stack([X[int(c)] for c in ids]), np.array([y[int(c)] for c in ids]))
        Xtr, ytr = get(tr); Xva, yva = get(va); Xte, yte = get(te)
        sc = StandardScaler().fit(Xtr)
        best = None
        for C in 10.0 ** np.arange(-5, 2):
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(Xtr), ytr)
            loss = log_loss(yva, m.predict_proba(sc.transform(Xva))[:, 1], labels=[0, 1])
            if best is None or loss < best[0]:
                best = (loss, C, m)
        p = best[2].predict_proba(sc.transform(Xte))[:, 1]
        for c, t, q in zip(te, yte, p):
            rows.append(dict(encoder=enc, split=k, case_id=int(c), true_label=int(t), prob_class1=float(q),
                             correct=bool((q >= 0.5) == t), chosen_C=best[1]))
        print(enc, k, "C", best[1], "val log loss %.3f" % best[0], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(O, "results", "baseline_logreg_predictions.csv"), index=False)
print("done")
