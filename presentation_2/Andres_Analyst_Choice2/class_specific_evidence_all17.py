"""STAT 390 Presentation 2 — Analysts Choice 2: class-specific patch evidence.

Uses the saved KimiaNet MIL checkpoint and its recorded 17 test-case IDs.
Does not modify dataset.py, models.py, the trained model, or any splits.
Outputs additive class-margin contributions using model-local patch indices.

Run from the Presentation2_base_code directory in an allocated QUEST compute session:
    python class_specific_evidence_all17.py
"""

import os
import re

import matplotlib
matplotlib.use("Agg")  # Save figures on QUEST without a GUI.
import matplotlib.pyplot as plt
import pandas as pd
import torch

from models import create_model
from data_utils import (
    load_labels,
    group_patches_by_slice,
    build_slice_to_class_map,
    build_case_dict,
)
from dataset import StainBagCasePooledFeatureDataset

RUN_DIR = "/projects/e32998/MIL_training/final_runs/run_20261105_110826_kimianet_split01"
CHECKPOINT = os.path.join(RUN_DIR, "checkpoints", "best.pth")
PREDICTIONS = os.path.join(RUN_DIR, "predictions_best.csv")
PATCH_DIR = "/projects/e32998/patches_benign_split"
EMBEDDING_DIR = "/projects/e32998/patches_varsize_pooled4096"
OUTPUT_DIR = "choice2_results"
EXAMPLE_CASES = {5, 18, 23}  # Two correct examples + one incorrect example.


@torch.inference_mode()
def calculate_class_evidence(model, stain_slices):
    """Exactly decompose the two-class logit margin for this MIL architecture."""
    model.eval()
    logits, attention = model(stain_slices, return_attn_weights=True)
    if len(attention.get("stain_order", [])) == 0:
        raise ValueError("No usable stains / patch features in case")

    classifier = model.classifier[1]
    if classifier.weight.shape[0] != 2:
        raise ValueError("This procedure expects a two-class linear classifier")

    weight_diff = classifier.weight[1] - classifier.weight[0]
    bias_diff = classifier.bias[1] - classifier.bias[0]
    records = []

    for stain_idx, stain in enumerate(attention["stain_order"]):
        stain_weight = attention["case_weights"][stain_idx]
        stain_info = attention["stain_weights"][stain]

        for slice_idx, slice_tensor in enumerate(stain_slices[stain]):
            slice_weight = stain_info["slice_weights"][slice_idx]
            patch_weights = stain_info["patch_weights"][slice_idx]

            # The same projection used in the original, evaluation-mode model.
            projected = model.patch_projector(slice_tensor.float())
            effective_attention = stain_weight * slice_weight * patch_weights
            evidence = effective_attention * (projected @ weight_diff)

            for patch_idx in range(evidence.numel()):
                records.append({
                    "stain": stain,
                    "slice_idx": slice_idx,
                    "patch_idx": patch_idx,
                    "attention": float(effective_attention[patch_idx]),
                    "evidence": float(evidence[patch_idx]),
                })

    original_margin = float(logits[1] - logits[0])
    reconstructed_margin = float(bias_diff) + sum(r["evidence"] for r in records)
    return records, logits, original_margin, reconstructed_margin


def build_test_dataset(case_ids):
    """Read patch PNG *names* once, then reuse the existing dataset pipeline."""
    selected_names = []
    for entry in os.scandir(PATCH_DIR):
        if not entry.is_file() or not entry.name.lower().endswith(".png"):
            continue
        match = re.match(r"case_(\d+)_", entry.name, flags=re.IGNORECASE)
        if match and int(match.group(1)) in case_ids:
            selected_names.append(entry.name)

    if not selected_names:
        raise FileNotFoundError("No PNG filenames found for listed test cases")

    patches = group_patches_by_slice(sorted(selected_names), PATCH_DIR)
    labels = load_labels()
    slice_to_class = build_slice_to_class_map(patches, labels)

    case_dict, label_map = build_case_dict(
        list(slice_to_class.keys()), patches, slice_to_class
    )

    dataset = StainBagCasePooledFeatureDataset(
        case_dict, label_map,
        embeddings_dir=EMBEDDING_DIR,
        per_slice_cap=800,
        shuffle_patches=False,
    )

    index_for_case = {
        int(case_id): i
        for i, (case_id, _) in enumerate(dataset.items)
    }
    return dataset, index_for_case

def save_case_plot(case_id, records, predicted_label):
    top = sorted(records, key=lambda r: abs(r["evidence"]), reverse=True)[:10]
    top.sort(key=lambda r: r["evidence"])
    labels = [f'{r["stain"]} | slice {r["slice_idx"]}, patch {r["patch_idx"]}' for r in top]
    scores = [r["evidence"] for r in top]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(labels, scores)
    ax.axvline(0, linewidth=1, color="black")
    ax.set_xlabel("Class 1 minus benign logit contribution")
    ax.set_title(f"Case {case_id}: strongest directional patch evidence\n"
                 f"Model predicted class {predicted_label}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"case_{case_id}_evidence.png"), dpi=220)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = pd.read_csv(PREDICTIONS)
    saved["case_id"] = saved["case_id"].astype(int)
    if saved["case_id"].duplicated().any():
        raise ValueError("Saved prediction CSV has duplicate case IDs")

    test_ids = set(saved["case_id"])
    print(f"Saved test cases: {len(test_ids)}", flush=True)
    dataset, index_for_case = build_test_dataset(test_ids)
    missing = test_ids - set(index_for_case)
    if missing:
        raise ValueError(f"Test cases missing from dataset: {sorted(missing)}")

    model = create_model()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    summaries = []
    top_rows = []

    for _, saved_row in saved.iterrows():
        case_id = int(saved_row["case_id"])
        case_data = dataset[index_for_case[case_id]]
        records, logits, margin, reconstructed = calculate_class_evidence(
            model, case_data["stain_slices"]
        )
        probs = torch.softmax(logits, dim=0)
        predicted = int(torch.argmax(logits))
        true_label = int(case_data["label"])
        err = abs(margin - reconstructed)
        p1 = float(probs[1])

        # Compare with the saved evaluation, which helps detect input/checkpoint mismatches.
        same_saved = (
            predicted == int(saved_row["predicted_label"])
            and abs(p1 - float(saved_row["prob_class1"])) < 1e-3
            and true_label == int(saved_row["true_label"])
        )
        top_five = sorted(records, key=lambda r: abs(r["evidence"]), reverse=True)[:5]
        supports_pred = sum(
            (r["evidence"] > 0 if predicted == 1 else r["evidence"] < 0)
            for r in top_five
        )

        summaries.append({
            "case_id": case_id,
            "true_label": true_label,
            "predicted_label": predicted,
            "correct": predicted == true_label,
            "prob_class0": float(probs[0]),
            "prob_class1": p1,
            "n_patches": len(records),
            "top5_support_predicted_class": supports_pred,
            "positive_evidence_count": sum(r["evidence"] > 0 for r in records),
            "negative_evidence_count": sum(r["evidence"] < 0 for r in records),
            "original_margin": margin,
            "reconstructed_margin": reconstructed,
            "reconstruction_error": err,
            "matches_saved_prediction": same_saved,
        })

        for rank, rec in enumerate(top_five, 1):
            top_rows.append({"case_id": case_id, "rank": rank, **rec,
                             "favors_class": (1 if rec["evidence"] > 0 else 0)})

        if case_id in EXAMPLE_CASES:
            save_case_plot(case_id, records, predicted)

        print(f"Case {case_id:>3}: true={true_label}, pred={predicted}, "
              f"P(class1)={p1:.4f}, top5 support predicted={supports_pred}/5, "
              f"reconstruction error={err:.2e}, "
              f"saved output match={same_saved}", flush=True)

    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(OUTPUT_DIR, "all17_summary.csv"), index=False)
    pd.DataFrame(top_rows).to_csv(
        os.path.join(OUTPUT_DIR, "all17_top5_evidence.csv"), index=False
    )

    print("\n=== Final summary ===")
    print("Number of analyzed cases:", len(summary))
    print("Correct predictions:", int(summary["correct"].sum()))
    print("Saved predictions reproduced:", int(summary["matches_saved_prediction"].sum()),
          "/", len(summary))
    print("Maximum reconstruction error:", summary["reconstruction_error"].max())
    print("Top five all support predicted class in:",
          int((summary["top5_support_predicted_class"] == 5).sum()), "cases")
    print("Files saved to:", os.path.abspath(OUTPUT_DIR))
    if not summary["matches_saved_prediction"].all():
        print("WARNING: Some model outputs differ from the saved evaluation. "
              "Do not interpret those cases without checking the inputs.")


if __name__ == "__main__":
    main()