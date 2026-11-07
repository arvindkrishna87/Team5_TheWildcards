
import torch
from models import create_model

CHECKPOINT = "/projects/e32998/MIL_training/final_runs/run_20261105_110826_kimianet_split01/checkpoints/best.pth"


@torch.no_grad()
def calculate_class_evidence(model, stain_slices):
    model.eval()

    # Original prediction and attention weights
    logits, attention = model(
        stain_slices, return_attn_weights=True
    )

    # Difference between classifier weights for classes 1 and 0
    classifier = model.classifier[1]
    weight_diff = classifier.weight[1] - classifier.weight[0]
    bias_diff = classifier.bias[1] - classifier.bias[0]

    records = []

    for stain_idx, stain in enumerate(attention["stain_order"]):
        stain_weight = attention["case_weights"][stain_idx]
        stain_info = attention["stain_weights"][stain]

        for slice_idx, slice_tensor in enumerate(stain_slices[stain]):
            slice_weight = stain_info["slice_weights"][slice_idx]
            patch_weights = stain_info["patch_weights"][slice_idx]

            # Project original patch features
            embeddings = model.patch_projector(slice_tensor)

            # Effective attention for each patch
            effective_attention = (
                stain_weight * slice_weight * patch_weights
            )

            # Contribution to class 1 minus class 0
            evidence = effective_attention * (embeddings @ weight_diff)

            for patch_idx, score in enumerate(evidence):
                records.append({
                    "stain": stain,
                    "slice_idx": slice_idx,
                    "patch_idx": patch_idx,
                    "attention": effective_attention[patch_idx].item(),
                    "evidence": score.item()
                })

    # Validate the mathematical decomposition
    original_margin = (logits[1] - logits[0]).item()
    reconstructed_margin = (
        bias_diff.item() + sum(r["evidence"] for r in records)
    )

    return records, original_margin, reconstructed_margin


# Load trained MIL model
model = create_model()
checkpoint = torch.load(
    CHECKPOINT, map_location="cpu", weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Small artificial example to verify our calculation
torch.manual_seed(42)
sample = {
    "h&e": [torch.randn(4, 4096)],
    "melan": [torch.randn(3, 4096)]
}

records, original, reconstructed = calculate_class_evidence(
    model, sample
)

print("Original logit difference:", original)
print("Reconstructed difference:", reconstructed)
print("Absolute error:", abs(original - reconstructed))

print("\nMost influential patches:")
for r in sorted(
    records, key=lambda x: abs(x["evidence"]), reverse=True
)[:5]:
    direction = "Class 1" if r["evidence"] > 0 else "Class 0"
    print(r, "Supports:", direction)



# %%
# TEST CLASS-SPECIFIC EVIDENCE ON REAL CASE 23

import os
import glob

from data_utils import (
    load_labels,
    group_patches_by_slice,
    build_slice_to_class_map,
    build_case_dict
)
from dataset import StainBagCasePooledFeatureDataset

case_id = 5

patch_dir = "/projects/e32998/patches_benign_split"
embedding_dir = "/projects/e32998/patches_varsize_pooled4096"

# Find the original patch files belonging to Case 23
patch_paths = sorted(set(
    glob.glob(os.path.join(patch_dir, f"case_{case_id}_*.png"))
    + glob.glob(os.path.join(patch_dir, f"case_{case_id:03d}_*.png"))
))

if not patch_paths:
    raise FileNotFoundError("No patch files found for Case 23")

# Reuse the professor's original data preparation
patches = group_patches_by_slice(
    [os.path.basename(p) for p in patch_paths],
    patch_dir
)

labels = load_labels()
slice_to_class = build_slice_to_class_map(patches, labels)

case_dict, label_map = build_case_dict(
    list(slice_to_class.keys()),
    patches,
    slice_to_class
)

# Load the real feature embeddings
dataset = StainBagCasePooledFeatureDataset(
    case_dict,
    label_map,
    embeddings_dir=embedding_dir,
    per_slice_cap=800,
    shuffle_patches=False
)

case_data = dataset[0]

# Calculate our class-specific evidence
records, original, reconstructed = calculate_class_evidence(
    model,
    case_data["stain_slices"]
)

# Get the model's prediction
with torch.no_grad():
    logits = model(case_data["stain_slices"])
    probabilities = torch.softmax(logits, dim=0)

print("\nREAL CASE:", case_id)
print("Predicted class:", int(torch.argmax(logits)))
print("Probability benign:", probabilities[0].item())
print("Probability class 1:", probabilities[1].item())

print("\nOriginal margin:", original)
print("Reconstructed margin:", reconstructed)
print("Absolute error:", abs(original - reconstructed))

print("\nTop 5 patches by absolute evidence:")
for r in sorted(
    records,
    key=lambda x: abs(x["evidence"]),
    reverse=True
)[:5]:
    direction = "Class 1" if r["evidence"] > 0 else "Benign"
    print(r, "Supports:", direction)
