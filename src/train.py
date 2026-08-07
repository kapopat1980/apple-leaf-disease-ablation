import sys, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from common import (AppleLeafDataset, BaselineCNN, AttentiveFocalNet, FocalLoss,
                     CLASSES, NUM_CLASSES, count_params, SPLITS)

torch.manual_seed(42)
np.random.seed(42)

MODE = sys.argv[1]  # 'baseline' or 'proposed'
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
PATIENCE = 7
BATCH = 64
device = torch.device("cpu")

# class counts for weighting
counts = np.zeros(NUM_CLASSES)
for _, label in SPLITS["train"]:
    counts[label] += 1
print("Train class counts:", dict(zip(CLASSES, counts.astype(int))))

if MODE == "baseline":
    train_ds = AppleLeafDataset("train", augment=True)  # simple flips/rotation still applied via augment fn
    model = BaselineCNN().to(device)
    class_weights = None
    criterion = torch.nn.CrossEntropyLoss()
    log_path = "log_baseline.json"
    ckpt_path = "best_baseline.pt"
elif MODE == "proposed":
    # minority boost: physically-inspired augmentation applied more to minority classes
    max_count = counts.max()
    minority_boost = {i: max(1, round(max_count / counts[i] / 2)) for i in range(NUM_CLASSES)}
    minority_boost = {k: min(v, 4) for k, v in minority_boost.items()}  # cap multiplier
    print("Minority boost multipliers:", dict(zip(CLASSES, [minority_boost[i] for i in range(NUM_CLASSES)])))
    train_ds = AppleLeafDataset("train", augment=True, minority_boost=minority_boost)
    model = AttentiveFocalNet().to(device)
    inv_freq = 1.0 / np.sqrt(counts)
    alpha = inv_freq / inv_freq.sum() * NUM_CLASSES
    class_weights = torch.tensor(alpha, dtype=torch.float32)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    log_path = "log_proposed.json"
    ckpt_path = "best_proposed.pt"
else:
    raise ValueError(MODE)

val_ds = AppleLeafDataset("val", augment=False)

print(f"[{MODE}] params={count_params(model):,} train_size={len(train_ds)} val_size={len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

best_val_f1 = -1
best_epoch = -1
patience_ctr = 0
history = []

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_preds, val_true = [], []
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
            preds = out.argmax(1)
            val_preds.extend(preds.numpy().tolist())
            val_true.extend(yb.numpy().tolist())
    val_loss /= len(val_ds)
    val_macro_f1 = f1_score(val_true, val_preds, average="macro")
    val_acc = float(np.mean(np.array(val_preds) == np.array(val_true)))

    scheduler.step(val_macro_f1)
    elapsed = time.time() - t0
    print(f"[{MODE}] epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
          f"val_acc={val_acc:.4f} val_macroF1={val_macro_f1:.4f} elapsed={elapsed:.0f}s", flush=True)
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "val_acc": val_acc, "val_macro_f1": val_macro_f1})

    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        best_epoch = epoch
        patience_ctr = 0
        torch.save(model.state_dict(), ckpt_path)
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"[{MODE}] early stopping at epoch {epoch} (best epoch {best_epoch}, "
                  f"best val_macroF1={best_val_f1:.4f})")
            break

with open(log_path, "w") as f:
    json.dump({"mode": MODE, "history": history, "best_epoch": best_epoch,
               "best_val_macro_f1": best_val_f1, "params": count_params(model),
               "total_time_s": time.time() - t0}, f, indent=2)

print(f"[{MODE}] DONE best_epoch={best_epoch} best_val_macroF1={best_val_f1:.4f} "
      f"total_time={time.time()-t0:.0f}s")
