import os, random, json
random.seed(42)

ROOT = "apple_data"
CLASSES = sorted(os.listdir(ROOT))
splits = {"train": [], "val": [], "test": []}

for ci, c in enumerate(CLASSES):
    files = [os.path.join(ROOT, c, f) for f in os.listdir(os.path.join(ROOT, c))]
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)
    n = len(files)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    for f in train_files:
        splits["train"].append((f, ci))
    for f in val_files:
        splits["val"].append((f, ci))
    for f in test_files:
        splits["test"].append((f, ci))
    print(f"{c}: total={n} train={len(train_files)} val={len(val_files)} test={len(test_files)}")

random.shuffle(splits["train"])
random.shuffle(splits["val"])
random.shuffle(splits["test"])

with open("splits.json", "w") as f:
    json.dump({"classes": CLASSES, "splits": splits}, f)

print("Classes:", CLASSES)
print("Totals:", {k: len(v) for k, v in splits.items()})
