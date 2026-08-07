import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                              roc_auc_score, accuracy_score)
from common import AppleLeafDataset, BaselineCNN, AttentiveFocalNet, CLASSES, NUM_CLASSES

torch.manual_seed(42)
device = torch.device("cpu")

test_ds = AppleLeafDataset("test", augment=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

results = {}

for mode, ModelCls, ckpt in [("baseline", BaselineCNN, "best_baseline.pt"),
                              ("proposed", AttentiveFocalNet, "best_proposed.pt")]:
    model = ModelCls().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_logits, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            all_logits.append(out.numpy())
            all_true.extend(yb.numpy().tolist())
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    all_preds = all_probs.argmax(axis=1)
    all_true = np.array(all_true)

    acc = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro")
    weighted_f1 = f1_score(all_true, all_preds, average="weighted")

    report = classification_report(all_true, all_preds, target_names=CLASSES,
                                     output_dict=True, zero_division=0)
    cm = confusion_matrix(all_true, all_preds).tolist()

    # one-vs-rest macro AUC
    y_true_onehot = np.eye(NUM_CLASSES)[all_true]
    try:
        macro_auc = roc_auc_score(y_true_onehot, all_probs, average="macro", multi_class="ovr")
        per_class_auc = roc_auc_score(y_true_onehot, all_probs, average=None, multi_class="ovr")
    except Exception as e:
        macro_auc = None
        per_class_auc = None

    results[mode] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc.tolist() if per_class_auc is not None else None,
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": CLASSES,
    }

    print(f"\n===== {mode.upper()} — TEST SET =====")
    print(f"Accuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}  Weighted-F1: {weighted_f1:.4f}  "
          f"Macro-AUC: {macro_auc:.4f}" if macro_auc else "")
    print("Per-class:")
    for c in CLASSES:
        r = report[c]
        print(f"  {c:20s} precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"f1={r['f1-score']:.3f} support={int(r['support'])}")
    print("Confusion matrix (rows=true, cols=pred):")
    print("Classes:", CLASSES)
    for row in cm:
        print(row)

with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved test_results.json")
