import json
import numpy as np
import torch
import cv2
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                              roc_auc_score, accuracy_score)
from common import BaselineCNN, AppleLeafDataset, CLASSES, NUM_CLASSES
from common_v3 import AttentiveNetV3, AppleLeafDatasetV3, to_tensor, SPLITS

torch.manual_seed(42)
device = torch.device("cpu")

# ---- Load models ----
baseline = BaselineCNN().to(device)
baseline.load_state_dict(torch.load("best_baseline.pt", map_location=device))
baseline.eval()

v3 = AttentiveNetV3().to(device)
v3.load_state_dict(torch.load("best_v3.pt", map_location=device))
v3.eval()

# ---- Baseline test set (64x64, its native resolution) ----
test_items_64 = SPLITS["test"]  # (path, label) list, shared across both since same splits.json

def load_64(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return img

def load_96(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    return img

all_true = []
baseline_probs = []
v3_probs = []

with torch.no_grad():
    for path, label in test_items_64:
        all_true.append(label)

        # Baseline @ 64x64, no TTA (matches its training distribution)
        img64 = load_64(path)
        x64 = to_tensor(img64).unsqueeze(0)
        out_b = baseline(x64)
        p_b = torch.softmax(out_b, dim=1).numpy()[0]
        baseline_probs.append(p_b)

        # v3 @ 96x96, WITH test-time augmentation (original + horizontal flip)
        img96 = load_96(path)
        img96_flip = cv2.flip(img96, 1)
        x96 = torch.stack([to_tensor(img96), to_tensor(img96_flip)])
        out_v3 = v3(x96)
        p_v3 = torch.softmax(out_v3, dim=1).numpy()
        p_v3_avg = p_v3.mean(axis=0)
        v3_probs.append(p_v3_avg)

all_true = np.array(all_true)
baseline_probs = np.array(baseline_probs)
v3_probs = np.array(v3_probs)
ensemble_probs = (baseline_probs + v3_probs) / 2.0

results = {}

def evaluate(name, probs, y_true):
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro")
    weighted_f1 = f1_score(y_true, preds, average="weighted")
    report = classification_report(y_true, preds, target_names=CLASSES,
                                     output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, preds).tolist()
    y_true_onehot = np.eye(NUM_CLASSES)[y_true]
    macro_auc = roc_auc_score(y_true_onehot, probs, average="macro", multi_class="ovr")

    print(f"\n===== {name} — TEST SET =====")
    print(f"Accuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}  Weighted-F1: {weighted_f1:.4f}  Macro-AUC: {macro_auc:.4f}")
    for c in CLASSES:
        r = report[c]
        print(f"  {c:20s} precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"f1={r['f1-score']:.3f} support={int(r['support'])}")
    print("Confusion matrix:", CLASSES)
    for row in cm:
        print(row)
    errors = int(np.array(cm).sum() - np.trace(np.array(cm)))
    print(f"Total errors: {errors} / {len(y_true)}")

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "macro_auc": macro_auc, "confusion_matrix": cm,
            "classification_report": report, "errors": errors}

results["baseline_v2run"] = evaluate("BASELINE (64px, re-eval)", baseline_probs, all_true)
results["v3_tta"] = evaluate("V3 (96px + SE-attn + light-aug + TTA)", v3_probs, all_true)
results["ensemble"] = evaluate("ENSEMBLE (baseline + v3 avg)", ensemble_probs, all_true)

with open("test_results_v3.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved test_results_v3.json")
