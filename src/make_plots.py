import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("test_results.json") as f:
    results = json.load(f)
with open("log_baseline.json") as f:
    log_b = json.load(f)
with open("log_proposed.json") as f:
    log_p = json.load(f)

CLASSES = results["baseline"]["classes"]

# --- Confusion matrices side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mode, title in zip(axes, ["baseline", "proposed"],
                            ["Baseline CNN\n(cross-entropy, no attention)",
                             "Proposed Model\n(SE-attention + focal loss + class weighting + phys. aug.)"]):
    cm = np.array(results[mode]["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

# --- Training curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for log, name, color in [(log_b, "Baseline", "tab:blue"), (log_p, "Proposed", "tab:orange")]:
    epochs = [h["epoch"] for h in log["history"]]
    val_f1 = [h["val_macro_f1"] for h in log["history"]]
    val_loss = [h["val_loss"] for h in log["history"]]
    axes[0].plot(epochs, val_f1, label=name, color=color, marker="o", ms=3)
    axes[1].plot(epochs, val_loss, label=name, color=color, marker="o", ms=3)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Macro-F1"); axes[0].set_title("Validation Macro-F1 per Epoch")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss"); axes[1].set_title("Validation Loss per Epoch")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# --- Per-class F1 bar comparison ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(CLASSES))
width = 0.35
f1_b = [results["baseline"]["classification_report"][c]["f1-score"] for c in CLASSES]
f1_p = [results["proposed"]["classification_report"][c]["f1-score"] for c in CLASSES]
ax.bar(x - width/2, f1_b, width, label="Baseline", color="tab:blue")
ax.bar(x + width/2, f1_p, width, label="Proposed", color="tab:orange")
ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=20, ha="right")
ax.set_ylabel("Test F1-score"); ax.set_ylim(0.9, 1.01)
ax.set_title("Per-class Test F1: Baseline vs Proposed")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("per_class_f1.png", dpi=150)
plt.close()

print("Saved: confusion_matrices.png, training_curves.png, per_class_f1.png")
