import pandas as pd
import matplotlib.pyplot as plt 

path = (
    "/projects/e32998/STAT390_Fall2026/"
    "Team2_TheConsultancy/Presentation_1/"
    "Alyssa_Presentation1_base_code/analysis/loss_history.csv"
)

df = pd.read_csv(path)
df["split"] = df["split"].astype(str).str.zfill(2)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

summary = []

for i, split in enumerate(sorted(df["split"].unique())):
    d = df[df["split"] == split].sort_values("epoch")
    ax = axes.flat[i]

    ax.plot(d["epoch"], d["train_loss_total"], label="Training loss")
    ax.plot(d["epoch"], d["val_loss"], label="Validation loss")

    best = d.loc[d["val_loss"].idxmin()]

    ax.scatter(
        best["epoch"], best["val_loss"],
        marker="*", s=130, zorder=5
    )

    ax.set_title(f"Team 2 — Split {split}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)

    summary.append({
        "split": split,
        "epochs": len(d),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": best["val_loss"],
        "train_loss_at_best": best["train_loss_total"],
        "gap_at_best": (
            best["val_loss"] - best["train_loss_total"]
        ),
        "final_train_loss": d.iloc[-1]["train_loss_total"],
        "final_val_loss": d.iloc[-1]["val_loss"]
    })

axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig("team2_loss_comparison.png", dpi=200)
plt.close()

summary = pd.DataFrame(summary)
summary.to_csv("team2_loss_summary.csv", index=False)

print(summary.round(4).to_string(index=False))
print("\nSaved team2_loss_comparison.png")
