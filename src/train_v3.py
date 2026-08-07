import time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from common_v3 import AppleLeafDatasetV3, AttentiveNetV3, LabelSmoothCE, CLASSES, NUM_CLASSES, SPLITS
from common import count_params

torch.manual_seed(42)
np.random.seed(42)

EPOCHS = 20
PATIENCE = 6
BATCH = 32  # smaller batch since images are larger (96x96) -> more memory per sample
device = torch.device("cpu")

counts = np.zeros(NUM_CLASSES)
for _, label in SPLITS["train"]:
    counts[label] += 1
print("Train class counts:", dict(zip(CLASSES, counts.astype(int))))

inv_freq = 1.0 / np.sqrt(counts)
alpha = inv_freq / inv_freq.sum() * NUM_CLASSES
class_weights = torch.tensor(alpha, dtype=torch.float32)

train_ds = AppleLeafDatasetV3("train", augment=True)
val_ds = AppleLeafDatasetV3("val", augment=False)

model = AttentiveNetV3().to(device)
criterion = LabelSmoothCE(weight=class_weights, smoothing=0.1)

print(f"[v3] params={count_params(model):,} train_size={len(train_ds)} val_size={len(val_ds)}")

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
    print(f"[v3] epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
          f"val_acc={val_acc:.4f} val_macroF1={val_macro_f1:.4f} elapsed={elapsed:.0f}s", flush=True)
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "val_acc": val_acc, "val_macro_f1": val_macro_f1})

    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        best_epoch = epoch
        patience_ctr = 0
        torch.save(model.state_dict(), "best_v3.pt")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"[v3] early stopping at epoch {epoch} (best epoch {best_epoch}, "
                  f"best val_macroF1={best_val_f1:.4f})")
            break

with open("log_v3.json", "w") as f:
    json.dump({"mode": "v3", "history": history, "best_epoch": best_epoch,
               "best_val_macro_f1": best_val_f1, "params": count_params(model),
               "total_time_s": time.time() - t0}, f, indent=2)

print(f"[v3] DONE best_epoch={best_epoch} best_val_macroF1={best_val_f1:.4f} "
      f"total_time={time.time()-t0:.0f}s")
