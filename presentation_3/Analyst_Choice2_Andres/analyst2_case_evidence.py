"""STAT 390 Presentation 3 — Analyst Choice 2.

Read-only signed evidence analysis for Cases 46 and 22 from the SAME Nov 19
UNI2-compatible checkpoint. Uses exact filename-aligned attention records.
The output is one-model evidence, NOT R=10 persistent-error evidence.

Place this file beside models.py in ~/STAT390_P3_Analyst2 and run with the
QUEST Python interpreter. It saves CSVs into analyst2_results/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models import create_model

ROOT = Path(__file__).resolve().parent
RUN = Path(
    "/projects/e32998/MIL_training/final_runs/"
    "run_20261119_115429_786671_2230960"
)
EMBEDDINGS = Path(
    "/projects/e32998/STAT390_Fall2026/"
    "Team3_DivideNConquer/presentation_3/embeddings_uni2_99"
)
OUTPUT = ROOT / "analyst2_results"
CASE_IDS = (46, 22)


def load_case_features(rows):
    """Load each exact patch filename's UNI2 feature vector in CSV row order."""
    vectors = []
    for filename in rows["patch_file"]:
        feature_path = EMBEDDINGS / Path(str(filename)).with_suffix(".pt").name
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing embedding: {feature_path}")
        feature = torch.load(feature_path, map_location="cpu", weights_only=True)
        if not isinstance(feature, torch.Tensor) or feature.numel() != 1536:
            raise ValueError(f"Unexpected feature shape: {feature_path}")
        vectors.append(feature.reshape(1536).float())
    return torch.stack(vectors)


def main():
    OUTPUT.mkdir(exist_ok=True)
    attn = pd.read_csv(RUN / "attention_analysis" / "patch_attention_by_filename.csv")
    preds = pd.read_csv(RUN / "predictions_best.csv")
    checkpoint = torch.load(
        RUN / "checkpoints" / "best.pth", map_location="cpu", weights_only=True
    )
    model = create_model(pooled_dim=1536)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    linear = model.classifier[1]
    weight_diff = (linear.weight[1] - linear.weight[0]).detach()
    bias_diff = float((linear.bias[1] - linear.bias[0]).detach())

    summaries = []
    top_tables = []
    for case_id in CASE_IDS:
        rows = attn.loc[attn["case_id"].astype(int) == case_id].copy()
        pred_rows = preds.loc[preds["case_id"].astype(int) == case_id]
        if rows.empty or len(pred_rows) != 1:
            raise ValueError(f"Expected attention and one prediction for case {case_id}")
        p = pred_rows.iloc[0]
        if not (
            (rows["true_label"].astype(int) == int(p["true_label"])).all()
            and (rows["pred_label"].astype(int) == int(p["predicted_label"])).all()
        ):
            raise ValueError(f"Labels differ between attention CSV and predictions for {case_id}")
        if rows["patch_file"].duplicated().any():
            raise ValueError(f"Repeated patch filenames in case {case_id}")

        with torch.no_grad():
            features = load_case_features(rows)
            projections = model.patch_projector(features)
            intrinsic_push = (projections @ weight_diff).cpu().numpy()

        weights = rows["effective_weight"].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError(f"Invalid attention weights in case {case_id}")
        attention_sum = float(weights.sum())
        if abs(attention_sum - 1.0) > 0.01:
            raise ValueError(
                f"Case {case_id}: attention sum {attention_sum:.6f} is not near 1; "
                "may be missing rows or using a different attention definition"
            )

        rows["signed_evidence"] = weights * intrinsic_push
        rows["supports"] = np.where(
            rows["signed_evidence"] > 0,
            "high-grade",
            np.where(rows["signed_evidence"] < 0, "benign", "neutral"),
        )
        reconstructed_margin = bias_diff + float(rows["signed_evidence"].sum())
        # CSV probabilities are rounded; allow a small reconstruction difference.
        p0 = float(p["prob_class0"])
        p1 = float(p["prob_class1"])
        if p0 <= 0 or p1 <= 0:
            raise ValueError("Rounded probability is zero; cannot verify log-odds")
        csv_margin = float(np.log(p1 / p0))
        margin_error = abs(reconstructed_margin - csv_margin)
        print(f"\nCASE {case_id} | actual={int(p['true_label'])} "
              f"predicted={int(p['predicted_label'])} | patches={len(rows)}")
        print(f"Attention weights sum: {attention_sum:.6f}")
        print(f"Saved prediction logit margin (from probabilities): {csv_margin:.6f}")
        print(f"Reconstructed logit margin: {reconstructed_margin:.6f}")
        print(f"Absolute margin difference: {margin_error:.6f}")
        if margin_error > 0.02:
            raise RuntimeError(
                f"Case {case_id}: checkpoint, embeddings, and attention CSV do NOT "
                "reconstruct the saved prediction. Do not use this evidence."
            )

        ranked = rows.sort_values("effective_weight", ascending=False)
        top = ranked.head(5).copy()
        top.insert(0, "rank", range(1, len(top) + 1))
        top.insert(0, "case_id_for_report", case_id)
        print("\nTop 5 patches ranked by effective attention:")
        print(top[["patch_file", "stain", "effective_weight", "signed_evidence", "supports"]]
              .to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        top_tables.append(top)
        summaries.append({
            "case_id": case_id,
            "true_label": int(p["true_label"]),
            "predicted_label": int(p["predicted_label"]),
            "num_patches": len(rows),
            "attention_sum": attention_sum,
            "csv_margin": csv_margin,
            "reconstructed_margin": reconstructed_margin,
            "absolute_margin_difference": margin_error,
            "positive_evidence_sum": float(rows.loc[rows.signed_evidence > 0, "signed_evidence"].sum()),
            "negative_evidence_sum": float(rows.loc[rows.signed_evidence < 0, "signed_evidence"].sum()),
            "top5_positive_count": int((top.signed_evidence > 0).sum()),
            "top5_negative_count": int((top.signed_evidence < 0).sum()),
        })

    pd.DataFrame(summaries).to_csv(OUTPUT / "case46_vs22_evidence_summary.csv", index=False)
    pd.concat(top_tables, ignore_index=True).to_csv(
        OUTPUT / "case46_vs22_top_attention_evidence.csv", index=False
    )
    print("\nVerified output saved to:", OUTPUT)
    print("NOTE: One-checkpoint comparison only; not repeated-error frequency.")


if __name__ == "__main__":
    main()