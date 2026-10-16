# =============================================================================
# Apple-leaf (PlantVillage) controlled ablation  --  Kaggle / T4 GPU script
# =============================================================================
# HOW TO RUN
#   1. New Kaggle notebook -> Accelerator: GPU T4 x1 -> Internet: ON
#      (internet is only needed to download ImageNet weights for experiment E3)
#   2. Add data:  https://www.kaggle.com/datasets/mohitsingh1804/plantvillage
#   3. Paste this file as cells (each "# %%" starts a new cell) or upload and run.
#   4. First do a smoke test with QUICK = True (about 5 min), then QUICK = False.
#
# WHAT IT DOES (maps to the reviewer comments)
#   E1  component-wise ablation, 10 configs x 5 seeds          (comments 1,2,4,8,10)
#   E2  input-resolution factor (64 / 96 / 128)                (comment 8)
#   E3  random-init vs ImageNet-pretrained backbones, with and
#       without the auxiliary components                       (comment 5)
#   A*  mean+-SD, paired deltas, McNemar, bootstrap CIs, AUC CIs,
#       stability metrics, hard-image + image-quality analysis,
#       Grad-CAM, params / MACs / latency                      (comments 2,3,9,11,12,13)
#
# NOTES
#   * Each seed controls BOTH the stratified 70/15/15 split and the weight
#     initialisation, so the SD reflects split + init + minibatch-order variance.
#     All configs share the same split for a given seed (paired comparison).
#   * The Kaggle dataset only ships train/ and val/. We pool them (after exact
#     duplicate removal) and re-split, so test images never leak into training.
#   * Resumable: finished runs are stored in /kaggle/working/runs and skipped.
#   * The script has NOT been executed by me on a GPU. Run QUICK mode first.
# =============================================================================

# %% [0] Imports, config, reproducibility ---------------------------------------
import os, sys, json, time, math, random, copy, glob, hashlib, warnings, shutil, platform
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                             confusion_matrix, precision_recall_fscore_support)
import scipy
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

QUICK = False                      # True = tiny smoke test
RUN_E1, RUN_E2, RUN_E3 = True, True, True
RUN_ANALYSIS = True

INPUT_ROOT = "/kaggle/input"
OUT = "/kaggle/working"
RUNS = f"{OUT}/runs"; FIGS = f"{OUT}/figs"; TABS = f"{OUT}/tables"
for d in (RUNS, FIGS, TABS):
    os.makedirs(d, exist_ok=True)

SEEDS = [42, 1, 2, 3, 4] if not QUICK else [42, 1]
EPOCHS_SMALL = 25 if not QUICK else 2
EPOCHS_PRE = 12 if not QUICK else 1
BATCH = 64
N_BOOT = 2000 if not QUICK else 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NW = min(4, os.cpu_count() or 1)
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})

ENV = {"python": platform.python_version(), "torch": torch.__version__,
       "torchvision": torchvision.__version__, "numpy": np.__version__,
       "opencv": cv2.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__,
       "device": DEVICE, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None}
json.dump(ENV, open(f"{OUT}/env.json", "w"), indent=2)
print(ENV)
if DEVICE == "cuda":   # fail early if the GPU is unusable (e.g. P100 with a newer PyTorch build)
    try:
        torch.zeros(1, device="cuda") + 1
    except Exception:
        raise RuntimeError("CUDA is present but unusable (e.g. 'no kernel image'). "
                           "Switch the accelerator to GPU T4 x2 and re-run.")
else:
    print("WARNING: no GPU detected - this will be extremely slow. Enable a GPU accelerator.")

# Show what is attached, to catch a missing dataset immediately
for _dp, _dn, _fn in os.walk(INPUT_ROOT):
    print(_dp, len(_fn))
    if _dp.count(os.sep) > 6: _dn.clear()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %% [1] Data: discover apple images, de-duplicate, build splits ----------------
def discover_apple(root=INPUT_ROOT):
    items = []
    for dp, _, fn in os.walk(root):
        base = os.path.basename(dp)
        imgs = [f for f in fn if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imgs and base.lower().startswith("apple"):
            cls = base.split("___")[-1]
            origin = "val" if "val" in dp.lower().split(os.sep) else "train"
            items += [(os.path.join(dp, f), cls, origin) for f in imgs]
    return items


def file_md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


raw_items = discover_apple()
assert len(raw_items) > 0, "No 'Apple*' class folders found under /kaggle/input - check the dataset is attached."
seen, items, dup = {}, [], 0
for p, c, o in sorted(raw_items):
    h = file_md5(p)
    if h in seen:
        dup += 1
        continue
    seen[h] = p
    items.append((p, c, o))
CLASSES = sorted({c for _, c, _ in items})
PATHS = [p for p, _, _ in items]
LABELS = np.array([CLASSES.index(c) for _, c, _ in items])
ORIGIN = np.array([o for _, _, o in items])
C = len(CLASSES)
print(f"{len(items)} unique apple images ({dup} exact duplicates removed); classes = {CLASSES}")
print("per-class counts:", dict(zip(CLASSES, np.bincount(LABELS))))
print("origin (train/val folder):", dict(zip(*np.unique(ORIGIN, return_counts=True))))
if len(items) != 3171:
    print("NOTE: the paper used 3,171 images (630/621/275/1645). This Kaggle copy differs "
          "- report the counts printed above in the manuscript.")
assert C == 4, f"Expected 4 apple classes, found {CLASSES}"

_CACHE = {}


def get_imgs(res):
    if res not in _CACHE:
        arr = np.zeros((len(PATHS), res, res, 3), np.uint8)
        for i, p in enumerate(PATHS):
            im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            arr[i] = cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA)
        _CACHE[res] = arr
    return _CACHE[res]


def make_split(seed):
    idx = np.arange(len(LABELS))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=LABELS, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=LABELS[tmp], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


SPLITS = {s: make_split(s) for s in SEEDS}
for s in SEEDS[:1]:
    print(f"split seed {s}: train/val/test = {[len(x) for x in SPLITS[s]]}")

# %% [2] Augmentation (applied ONLY to the training split, after partitioning) --
# geo  : horizontal flip p=0.5, vertical flip p=0.3, rotation p=0.5 in [-25,25] deg (reflect border)
# phys : CLAHE (p=0.7, clipLimit=2.0, tile 8x8 on LAB-L) + gamma (p=0.7, gamma~U[0.65,1.35])
#        + Gaussian noise (p=0.5, sigma=6 on 0-255 scale) + the geo steps above
def _geo(img):
    h, w = img.shape[:2]
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.3:
        img = cv2.flip(img, 0)
    if random.random() < 0.5:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-25, 25), 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img


def _phys(img):
    img = img.copy()
    if random.random() < 0.7:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    if random.random() < 0.7:
        gamma = random.uniform(0.65, 1.35)
        table = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255).astype(np.uint8)
        img = cv2.LUT(img, table)
    if random.random() < 0.5:
        img = np.clip(img.astype(np.float32) + np.random.normal(0, 6.0, img.shape), 0, 255).astype(np.uint8)
    return _geo(img)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(x, norm):
    if norm == "imagenet":
        return (x - MEAN.to(x.device)) / STD.to(x.device)
    return (x - 0.5) / 0.5


class DS(Dataset):
    def __init__(self, imgs, labels, idxs, aug, norm):
        self.imgs, self.labels, self.idxs, self.aug, self.norm = imgs, labels, list(idxs), aug, norm
        self.mean, self.std = MEAN.numpy().reshape(3), STD.numpy().reshape(3)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        j = self.idxs[i]
        img = self.imgs[j]
        img = _phys(img) if self.aug == "phys" else _geo(img) if self.aug == "geo" else img
        x = img.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std if self.norm == "imagenet" else (x - 0.5) / 0.5
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), int(self.labels[j])


def winit(wid):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s); random.seed(s)


def oversample(tr):
    """Minority multiplier m_c = min(4, max(1, round(max_n / n_c / 2))); applied to TRAIN only."""
    cnt = np.bincount(LABELS[tr], minlength=C)
    mult = {c: int(min(4, max(1, round(cnt.max() / cnt[c] / 2)))) for c in range(C)}
    out = []
    for j in tr:
        out += [j] * mult[LABELS[j]]
    return np.array(out), mult


def class_weights(tr):
    """alpha_c = (1/sqrt(n_c)) / sum_j (1/sqrt(n_j)) * C   (computed on the TRAIN split)."""
    cnt = np.bincount(LABELS[tr], minlength=C).astype(float)
    inv = 1.0 / np.sqrt(cnt)
    return inv / inv.sum() * C


# %% [3] Models ------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, ch, red=8):
        super().__init__()
        r = max(ch // red, 4)
        self.fc = nn.Sequential(nn.Linear(ch, r), nn.ReLU(inplace=True), nn.Linear(r, ch), nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(x.mean((2, 3))).view(b, c, 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, i, o, se):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                  nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.attn = SEBlock(o) if se else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.attn(self.conv(x)))


class SmallCNN(nn.Module):
    """4-block CNN (32-64-128-256), 1.17 M params; identical to the original study.
    se=True inserts an SE block (reduction 8) after each block."""
    def __init__(self, se=False, nc=C):
        super().__init__()
        self.b1, self.b2 = ConvBlock(3, 32, se), ConvBlock(32, 64, se)
        self.b3, self.b4 = ConvBlock(64, 128, se), ConvBlock(128, 256, se)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, nc))

    def forward(self, x):
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.head(x.mean((2, 3)))


def build_model(arch, se=False, pretrained=False):
    if arch == "smallcnn":
        return SmallCNN(se=se)
    if arch == "resnet18":
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        m.fc = nn.Linear(512, C)
        return m
    if arch == "efficientnet_b0":
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(1280, C)
        return m
    raise ValueError(arch)


def cam_layer(model, arch):
    return {"smallcnn": lambda: model.b4.conv, "resnet18": lambda: model.layer4,
            "efficientnet_b0": lambda: model.features[-1]}[arch]()


# %% [4] Losses ------------------------------------------------------------------
def make_loss(kind, alpha):
    """ce   : plain cross-entropy
       cw   : mean_i( alpha[y_i] * CE_i )                  (class weighting only)
       fl   : mean_i( (1-p_t)^2 * CE_i )                   (focal loss, gamma=2, no alpha)
       flcw : mean_i( alpha[y_i] * (1-p_t)^2 * CE_i )      (focal + class weights; original 'v1')
       ls   : CE with label smoothing 0.1"""
    a = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)

    def fn(lo, y):
        lo = lo.float()
        ce = F.cross_entropy(lo, y, reduction="none", label_smoothing=0.1 if kind == "ls" else 0.0)
        if kind in ("ce", "ls"):
            return ce.mean()
        if kind == "cw":
            return (a[y] * ce).mean()
        pt = torch.exp(-ce)
        f = (1 - pt) ** 2 * ce
        return (a[y] * f).mean() if kind == "flcw" else f.mean()
    return fn


# %% [5] Train / evaluate one run ---------------------------------------------
@torch.no_grad()
def predict(model, arr, norm, tta=False, bs=256):
    model.eval()
    out = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
        x = normalize(x, norm)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            p = F.softmax(model(x).float(), 1)
            if tta:  # TTA = mean of softmax(original) and softmax(horizontal flip)
                p = (p + F.softmax(model(torch.flip(x, dims=[3])).float(), 1)) / 2
        out.append(p.cpu().numpy())
    p = np.concatenate(out).astype(np.float64)
    return p / p.sum(1, keepdims=True)


def macro_auc(y, probs):
    """One-vs-rest AUC per class (class vs all others), then unweighted mean over classes.
       probs = softmax outputs (for TTA: mean of the two views' softmax; for ensembles: mean over members)."""
    return roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(C)))


def metrics(y, probs, prefix=""):
    pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(C)), zero_division=0)
    d = {f"{prefix}acc": accuracy_score(y, pred), f"{prefix}macro_f1": f1_score(y, pred, average="macro"),
         f"{prefix}macro_auc": macro_auc(y, probs), f"{prefix}errors": int((pred != y).sum())}
    for k, c in enumerate(CLASSES):
        d[f"{prefix}f1_{c}"], d[f"{prefix}prec_{c}"], d[f"{prefix}rec_{c}"] = f[k], p[k], r[k]
    return d


def run_one(exp, name, cfg, seed, save_ckpt=False):
    tag = f"{exp}__{name}__s{seed}"
    if os.path.exists(f"{RUNS}/{tag}.json"):
        return
    t0 = time.time()
    set_seed(seed)
    imgs = get_imgs(cfg["res"])
    tr, va, te = SPLITS[seed]
    assert len(set(tr) & set(te)) == 0 and len(set(tr) & set(va)) == 0 and len(set(va) & set(te)) == 0
    alpha = class_weights(tr)
    tr_idx, mult = oversample(tr) if cfg["os"] else (tr, None)
    norm = "half" if cfg["arch"] == "smallcnn" else "imagenet"
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(DS(imgs, LABELS, tr_idx, cfg["aug"], norm), batch_size=BATCH, shuffle=True, drop_last=True,
                    num_workers=NW, worker_init_fn=winit, generator=g, pin_memory=(DEVICE == "cuda"),
                    persistent_workers=(NW > 0))
    model = build_model(cfg["arch"], cfg["se"], cfg["pretrained"]).to(DEVICE)
    crit = make_loss(cfg["loss"], alpha)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    hist = {"val_f1": [], "val_acc": [], "val_ce": []}
    best_f1, best_ep, best_state = -1, -1, None
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                lo = model(xb)
            loss = crit(lo, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        pv = predict(model, imgs[va], norm)
        f1 = f1_score(LABELS[va], pv.argmax(1), average="macro")
        hist["val_f1"].append(float(f1)); hist["val_acc"].append(float(accuracy_score(LABELS[va], pv.argmax(1))))
        hist["val_ce"].append(float(-np.log(np.clip(pv[np.arange(len(va)), LABELS[va]], 1e-12, 1)).mean()))
        sched.step(f1)
        if f1 > best_f1:   # checkpoint selection: best validation macro-F1 (first maximum)
            best_f1, best_ep, best_state = f1, ep, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    y = LABELS[te]
    p0, p1 = predict(model, imgs[te], norm), predict(model, imgs[te], norm, tta=True)
    row = dict(exp=exp, name=name, seed=seed, **{k: v for k, v in cfg.items()},
               best_epoch=best_ep, train_time_s=time.time() - t0, n_train=len(tr_idx),
               params=sum(p.numel() for p in model.parameters()),
               **metrics(y, p0), **metrics(y, p1, "tta_"))
    np.savez_compressed(f"{RUNS}/{tag}.npz", idx_te=te, y=y, probs=p0, probs_tta=p1, **hist)
    if save_ckpt:
        torch.save(best_state, f"{RUNS}/{tag}.pt")
    json.dump(row, open(f"{RUNS}/{tag}.json", "w"))
    print(f"[{tag}] acc={row['acc']:.4f} f1={row['macro_f1']:.4f} auc={row['macro_auc']:.4f} "
          f"err={row['errors']} best_ep={best_ep} {row['train_time_s']:.0f}s", flush=True)


# %% [6] Experiment registry -----------------------------------------------------
BASE = dict(arch="smallcnn", se=False, pretrained=False, loss="ce", aug="geo", os=False,
            res=64, epochs=EPOCHS_SMALL, lr=1e-3)


def mk(**kw):
    d = dict(BASE); d.update(kw); return d


E1 = {  # component-wise ablation at fixed 64x64 input, same optimiser / epochs / split / seed
    "base": mk(),                                   # CE, light geometric aug, no SE, no oversampling
    "se": mk(se=True),
    "fl": mk(loss="fl"),
    "cw": mk(loss="cw"),
    "flcw": mk(loss="flcw"),
    "ls": mk(loss="ls"),
    "pa": mk(aug="phys"),                           # == the ORIGINAL baseline (it used phys. aug.)
    "os": mk(os=True),
    "se_flcw": mk(se=True, loss="flcw"),
    "full": mk(se=True, loss="flcw", aug="phys", os=True),   # == original "v1 proposed"
}
E2 = {}
for r in (96, 128):
    E2[f"base@{r}"] = mk(res=r)
    E2[f"full@{r}"] = mk(res=r, se=True, loss="flcw", aug="phys", os=True)
E3 = {}
for arch in ("resnet18", "efficientnet_b0"):
    for pre in (False, True):
        tagp = "pre" if pre else "rand"
        base = dict(arch=arch, pretrained=pre, res=128, epochs=EPOCHS_PRE, lr=5e-4, se=False)
        E3[f"{arch}_{tagp}_base"] = {**BASE, **base}                                    # CE + geo aug
        E3[f"{arch}_{tagp}_aux"] = {**BASE, **base, "loss": "flcw", "aug": "phys", "os": True}

EXPS = []
if RUN_E1: EXPS.append(("E1", E1))
if RUN_E2: EXPS.append(("E2", E2))
if RUN_E3: EXPS.append(("E3", E3))

# %% [7] Run everything (seed-outer loop => complete seeds finish first) ----------
import traceback
t_all = time.time()
for seed in SEEDS:
    for exp, cfgs in EXPS:
        for name, cfg in cfgs.items():
            try:
                run_one(exp, name, cfg, seed, save_ckpt=(exp == "E1" and seed == SEEDS[0]))
            except Exception:
                traceback.print_exc()          # always show the full traceback
                if exp != "E3":                # E3 may fail legitimately (no internet for pretrained weights)
                    raise                      # E1/E2 failures stop the script loudly
    print(f"=== seed {seed} done, elapsed {(time.time() - t_all) / 60:.1f} min ===", flush=True)

# %% [8] Aggregate: mean +- SD ---------------------------------------------------
def load_rows():
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*.json"))]
    return pd.DataFrame(rows)


def ms(x):
    return f"{np.mean(x):.4f} +- {np.std(x, ddof=1):.4f}" if len(x) > 1 else f"{np.mean(x):.4f}"


DF = load_rows()
DF.to_csv(f"{TABS}/all_runs.csv", index=False)
minority = CLASSES[int(np.argmin(np.bincount(LABELS)))]
agg_rows = []
for (exp, name), g in DF.groupby(["exp", "name"], sort=False):
    agg_rows.append(dict(exp=exp, config=name, n_seeds=len(g),
                         acc=ms(g.acc), macro_f1=ms(g.macro_f1), macro_auc=ms(g.macro_auc),
                         **{f"f1_{minority}": ms(g[f"f1_{minority}"])},
                         tta_acc=ms(g.tta_acc), tta_macro_f1=ms(g.tta_macro_f1),
                         errors_mean=f"{g.errors.mean():.1f}", params=int(g.params.iloc[0]),
                         best_epoch=f"{g.best_epoch.mean():.1f}", train_time_s=f"{g.train_time_s.mean():.0f}"))
AGG = pd.DataFrame(agg_rows)
AGG.to_csv(f"{TABS}/summary_mean_sd.csv", index=False)
print(AGG.to_string(index=False))

# %% [9] Paired comparisons vs. base (same split+seed) -------------------------
def load_npz(exp, name, seed):
    return np.load(f"{RUNS}/{exp}__{name}__s{seed}.npz")


def mcnemar_exact(c_base, c_cfg):
    b = int((c_base & ~c_cfg).sum()); c = int((~c_base & c_cfg).sum())
    return (b, c, 1.0) if b + c == 0 else (b, c, stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def paired_table(exp, ref, names, metric="macro_f1"):
    out = []
    for nm in names:
        d, nsig, bc = [], 0, []
        for s in SEEDS:
            r0 = DF[(DF.exp == exp) & (DF.name == ref) & (DF.seed == s)]
            r1 = DF[(DF.exp == exp) & (DF.name == nm) & (DF.seed == s)]
            if r0.empty or r1.empty:
                continue
            d.append(float(r1[metric].iloc[0] - r0[metric].iloc[0]))
            a, b_ = load_npz(exp, ref, s), load_npz(exp, nm, s)
            assert (a["idx_te"] == b_["idx_te"]).all()
            b, c, p = mcnemar_exact(a["probs"].argmax(1) == a["y"], b_["probs"].argmax(1) == b_["y"])
            nsig += int(p < 0.05); bc.append((b, c))
        if len(d) < 2:
            continue
        d = np.array(d); m, sd = d.mean(), d.std(ddof=1)
        h = stats.t.ppf(0.975, len(d) - 1) * sd / math.sqrt(len(d))
        try:
            pw = stats.wilcoxon(d).pvalue
        except ValueError:
            pw = np.nan
        out.append(dict(config=nm, metric=metric, mean_delta=m, sd=sd, ci95_lo=m - h, ci95_hi=m + h,
                        seeds_improved=int((d > 0).sum()), n=len(d), wilcoxon_p=pw,
                        mcnemar_sig_splits=f"{nsig}/{len(d)}", discordant_b_c=str(bc)))
    return pd.DataFrame(out)


if RUN_ANALYSIS and RUN_E1:
    names = [n for n in E1 if n != "base"]
    P1 = pd.concat([paired_table("E1", "base", names, m) for m in ("acc", "macro_f1")])
    P1.to_csv(f"{TABS}/paired_vs_base.csv", index=False)
    # the comparison that mirrors the ORIGINAL manuscript (pa = original baseline, full = original v1)
    P2 = paired_table("E1", "pa", ["full"], "macro_f1")
    P2.to_csv(f"{TABS}/paired_full_vs_pa_original_comparison.csv", index=False)
    print(P1.to_string(index=False)); print(P2.to_string(index=False))

# %% [10] Bootstrap CIs on the primary split (seed = SEEDS[0]) --------------------
def strat_boot(y, rng):
    idx = []
    for c in range(C):
        ii = np.where(y == c)[0]
        idx.append(rng.choice(ii, len(ii), replace=True))
    return np.concatenate(idx)


def boot_ci(exp, names, seed=SEEDS[0], tta=False, nb=N_BOOT):
    rng = np.random.default_rng(0)
    key = "probs_tta" if tta else "probs"
    z = {n: load_npz(exp, n, seed) for n in names}
    y = z[names[0]]["y"]
    B = [strat_boot(y, rng) for _ in range(nb)]
    res = []
    for n in names:
        p = z[n][key]
        a, f, u = [], [], []
        for b in B:
            pr = p[b].argmax(1)
            a.append((pr == y[b]).mean()); f.append(f1_score(y[b], pr, average="macro"))
            u.append(macro_auc(y[b], p[b] / p[b].sum(1, keepdims=True)))
        q = lambda v: np.percentile(v, [2.5, 97.5])
        pt = metrics(y, p)
        res.append(dict(config=n, acc=pt["acc"], acc_lo=q(a)[0], acc_hi=q(a)[1],
                        macro_f1=pt["macro_f1"], f1_lo=q(f)[0], f1_hi=q(f)[1],
                        macro_auc=pt["macro_auc"], auc_lo=q(u)[0], auc_hi=q(u)[1]))
    return pd.DataFrame(res), z, y, B


def paired_boot_delta(z, y, B, ref, names, key="probs"):
    out = []
    for n in names:
        d = []
        for b in B:
            d.append(f1_score(y[b], z[n][key][b].argmax(1), average="macro") -
                     f1_score(y[b], z[ref][key][b].argmax(1), average="macro"))
        out.append(dict(config=n, delta_macro_f1=np.mean(d), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                        p_boot_two_sided=min(1.0, 2 * min((np.array(d) <= 0).mean(), (np.array(d) >= 0).mean()))))
    return pd.DataFrame(out)


if RUN_ANALYSIS and RUN_E1:
    for tta in (False, True):
        CI, Z, Y, BB = boot_ci("E1", list(E1), tta=tta)
        CI.to_csv(f"{TABS}/bootstrap_ci_split{SEEDS[0]}{'_tta' if tta else ''}.csv", index=False)
        paired_boot_delta(Z, Y, BB, "base", [n for n in E1 if n != "base"], "probs_tta" if tta else "probs") \
            .to_csv(f"{TABS}/paired_bootstrap_vs_base_split{SEEDS[0]}{'_tta' if tta else ''}.csv", index=False)
    print(CI.round(4).to_string(index=False))
    # per-class one-vs-rest AUC with CI for the minority class (base and full)
    rng = np.random.default_rng(1); rows = []
    for n in ("base", "full"):
        z = load_npz("E1", n, SEEDS[0]); y = z["y"]
        for k, c in enumerate(CLASSES):
            yy = (y == k).astype(int); a = roc_auc_score(yy, z["probs"][:, k]); bs_ = []
            for _ in range(N_BOOT):
                ii = rng.integers(0, len(y), len(y))
                if yy[ii].min() == yy[ii].max():
                    continue
                bs_.append(roc_auc_score(yy[ii], z["probs"][ii, k]))
            rows.append(dict(config=n, cls=c, support=int(yy.sum()), auc=a, lo=np.percentile(bs_, 2.5), hi=np.percentile(bs_, 97.5)))
    pd.DataFrame(rows).to_csv(f"{TABS}/per_class_auc_ci.csv", index=False)

# %% [11] Same-split ensemble (probability averaging) -------------------------------
# Probabilities of different configs trained on the SAME split are averaged. (Members from different
# splits are not mixed, because one member's training images would appear in another's test set.)
if RUN_ANALYSIS and RUN_E1:
    rows = []
    for s in SEEDS:
        members = ["base", "se", "full"]
        if all(os.path.exists(f"{RUNS}/E1__{m}__s{s}.npz") for m in members):
            zs = [load_npz("E1", m, s) for m in members]
            p = np.mean([z["probs"] for z in zs], 0)
            rows.append(dict(seed=s, **metrics(zs[0]["y"], p)))
    if rows:
        pd.DataFrame(rows).to_csv(f"{TABS}/ensemble_base_se_full_same_split.csv", index=False)

# %% [12] Stability metrics from validation curves (comment 11) -------------------
def stab(h, warm=5):
    f = np.array(h["val_f1"]); ce = np.array(h["val_ce"])
    drops = f[:-1] - f[1:]
    reach = np.where(f >= 0.95)[0]
    return dict(sd_f1_after_warm=f[warm:].std(ddof=1) if len(f) > warm + 1 else np.nan,
                sd_ce_after_warm=ce[warm:].std(ddof=1) if len(ce) > warm + 1 else np.nan,
                max_drop=max(0.0, drops.max()) if len(drops) else np.nan,
                n_collapses=int((drops > 0.05).sum()),
                epochs_to_95=int(reach[0] + 1) if len(reach) else np.nan,
                best_minus_last=float(f.max() - f[-1]))


if RUN_ANALYSIS and RUN_E1:
    rows = []
    for n in E1:
        for s in SEEDS:
            f = f"{RUNS}/E1__{n}__s{s}.npz"
            if os.path.exists(f):
                z = np.load(f); rows.append(dict(config=n, seed=s, **stab({k: z[k] for k in ("val_f1", "val_ce")})))
    ST = pd.DataFrame(rows); ST.to_csv(f"{TABS}/stability_per_run.csv", index=False)
    STS = ST.groupby("config", sort=False).agg(["mean", "std"]).round(4)
    STS.to_csv(f"{TABS}/stability_summary.csv"); print(STS.to_string())

# %% [13] Computational efficiency (comment 13) -----------------------------------
def count_macs(model, res):
    tot = [0]
    def hk(m, i, o):
        if isinstance(m, nn.Conv2d):
            tot[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            tot[0] += m.in_features * m.out_features
    hs = [m.register_forward_hook(hk) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, res, res, device=next(model.parameters()).device))
    for h in hs:
        h.remove()
    return tot[0]


@torch.no_grad()
def latency(model, res, bs, dev, reps=50):
    model = copy.deepcopy(model).to(dev).eval()
    x = torch.zeros(bs, 3, res, res, device=dev)
    for _ in range(5):
        model(x)
    if dev == "cuda": torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        model(x)
    if dev == "cuda": torch.cuda.synchronize()
    return (time.time() - t) / reps / bs * 1000.0   # ms / image


if RUN_ANALYSIS:
    rows = []
    targets = [("smallcnn_base", mk()), ("smallcnn_se", mk(se=True)), ("smallcnn_base@128", mk(res=128)),
               ("resnet18", dict(arch="resnet18", se=False, res=128)),
               ("efficientnet_b0", dict(arch="efficientnet_b0", se=False, res=128))]
    for nm, c in targets:
        m = build_model(c["arch"], c["se"], False).to(DEVICE)
        npar = sum(p.numel() for p in m.parameters())
        rows.append(dict(model=nm, res=c["res"], params_M=npar / 1e6, size_MB_fp32=npar * 4 / 2 ** 20,
                         MMACs=count_macs(m, c["res"]) / 1e6,
                         ms_per_img_cpu_bs1=latency(m.cpu(), c["res"], 1, "cpu", 30),
                         ms_per_img_gpu_bs1=latency(m, c["res"], 1, DEVICE) if DEVICE == "cuda" else np.nan,
                         ms_per_img_gpu_bs64=latency(m, c["res"], 64, DEVICE, 20) if DEVICE == "cuda" else np.nan))
    EFF = pd.DataFrame(rows); EFF.to_csv(f"{TABS}/efficiency.csv", index=False); print(EFF.round(3).to_string(index=False))

# %% [14] Hard images + systematic image-quality analysis (comment 9) -------------
def img_quality(res=128):
    arr = get_imgs(res)
    out = []
    for im in arr:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        out.append((cv2.Laplacian(g, cv2.CV_64F).var(), g.mean(), g.std(), hsv[..., 1].mean()))
    return pd.DataFrame(out, columns=["sharpness_lapvar", "brightness", "contrast_std", "saturation"])


if RUN_ANALYSIS and RUN_E1:
    miss, seen_n = np.zeros(len(LABELS)), np.zeros(len(LABELS))
    for n in E1:
        for s in SEEDS:
            f = f"{RUNS}/E1__{n}__s{s}.npz"
            if os.path.exists(f):
                z = np.load(f); wrong = z["probs"].argmax(1) != z["y"]
                np.add.at(seen_n, z["idx_te"], 1); np.add.at(miss, z["idx_te"], wrong.astype(float))
    freq = np.divide(miss, seen_n, out=np.full_like(miss, np.nan), where=seen_n > 0)
    Q = img_quality(); Q["path"] = PATHS; Q["label"] = [CLASSES[i] for i in LABELS]
    Q["times_in_test"] = seen_n; Q["miss_rate"] = freq
    Q.to_csv(f"{TABS}/image_quality_and_miss_rate.csv", index=False)
    hard = Q[(Q.times_in_test >= 5) & (Q.miss_rate >= 0.5)]
    easy = Q[(Q.times_in_test >= 5) & (Q.miss_rate == 0)]
    print(f"consistently misclassified (>=50% of runs, in test >=5 times): {len(hard)};  never misclassified: {len(easy)}")
    cmp_ = []
    for col in ["sharpness_lapvar", "brightness", "contrast_std", "saturation"]:
        if len(hard) >= 3:
            u = stats.mannwhitneyu(hard[col], easy[col])
            cmp_.append(dict(feature=col, hard_median=hard[col].median(), easy_median=easy[col].median(), mannwhitney_p=u.pvalue))
    pd.DataFrame(cmp_).to_csv(f"{TABS}/hard_vs_easy_image_quality.csv", index=False)
    hard.sort_values("miss_rate", ascending=False).to_csv(f"{TABS}/hard_images.csv", index=False)
    print(pd.DataFrame(cmp_).round(4).to_string(index=False))

# %% [15] Grad-CAM on misclassified test images (comment 9) ------------------------
def gradcam(model, layer, x, target):
    acts, grads = [], []
    def fh(m, i, o):
        acts.append(o); o.register_hook(lambda g: grads.append(g))
    h = layer.register_forward_hook(fh)
    model.eval(); x = x.clone().requires_grad_(True)
    lo = model(x); model.zero_grad(); lo[0, target].backward(); h.remove()
    w = grads[0].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts[0]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def leaf_mask(img):
    """Heuristic foreground mask: pixels far from the median border colour (PlantVillage backgrounds are plain)."""
    b = np.concatenate([img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3), img[:, :4].reshape(-1, 3), img[:, -4:].reshape(-1, 3)])
    d = np.linalg.norm(img.astype(float) - np.median(b, 0), axis=2)
    return d > max(25.0, d.mean())


if RUN_ANALYSIS and RUN_E1:
    s0 = SEEDS[0]; imgs64 = get_imgs(64); rows = []
    for nm in ("base", "se", "full"):
        ck = f"{RUNS}/E1__{nm}__s{s0}.pt"; zf = f"{RUNS}/E1__{nm}__s{s0}.npz"
        if not (os.path.exists(ck) and os.path.exists(zf)):
            continue
        cfg = E1[nm]; model = build_model("smallcnn", cfg["se"]).to(DEVICE); model.load_state_dict(torch.load(ck))
        z = np.load(zf); pred = z["probs"].argmax(1); wrong = np.where(pred != z["y"])[0]
        ok = np.where(pred == z["y"])[0]
        rng = np.random.default_rng(0); ok = rng.choice(ok, min(60, len(ok)), replace=False)

        def inside_frac(j, true_c, pred_c, use_pred=True):
            im = imgs64[z["idx_te"][j]]
            x = normalize(torch.from_numpy(im).permute(2, 0, 1)[None].float().to(DEVICE) / 255.0, "half")
            cam = gradcam(model, cam_layer(model, "smallcnn"), x, int(pred_c if use_pred else true_c))
            m = leaf_mask(im)
            return cam, float((cam * m).sum() / (cam.sum() + 1e-8)), float(m.mean())

        fr_w = [inside_frac(j, z["y"][j], pred[j])[1:] for j in wrong]
        fr_o = [inside_frac(j, z["y"][j], pred[j])[1:] for j in ok]
        rows.append(dict(model=nm, n_wrong=len(wrong), cam_mass_in_leaf_wrong=np.mean([a for a, _ in fr_w]) if fr_w else np.nan,
                         cam_mass_in_leaf_correct=np.mean([a for a, _ in fr_o]), leaf_area_frac_wrong=np.mean([b for _, b in fr_w]) if fr_w else np.nan,
                         leaf_area_frac_correct=np.mean([b for _, b in fr_o])))
        sel = wrong[:12]
        if len(sel):
            fig, ax = plt.subplots(2, len(sel), figsize=(3.2 * len(sel), 7.2), squeeze=False)
            for k, j in enumerate(sel):
                cam, _, _ = inside_frac(j, z["y"][j], pred[j]); im = imgs64[z["idx_te"][j]]
                big = cv2.resize(im, (192, 192), interpolation=cv2.INTER_CUBIC)
                ax[0, k].imshow(big); ax[0, k].axis("off")
                ax[0, k].set_title(f"true: {CLASSES[z['y'][j]]}\npred: {CLASSES[pred[j]]}", fontsize=12)
                ax[1, k].imshow(big); ax[1, k].imshow(cv2.resize(cam, (192, 192)), cmap="jet", alpha=0.45); ax[1, k].axis("off")
            plt.suptitle(f"Grad-CAM (target = predicted class) - {nm}, split {s0}", fontsize=16)
            plt.tight_layout(); plt.savefig(f"{FIGS}/gradcam_misclassified_{nm}.png", dpi=150); plt.close()
    pd.DataFrame(rows).to_csv(f"{TABS}/gradcam_summary.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

# %% [16] Figures (large fonts) -----------------------------------------------------
if RUN_ANALYSIS and RUN_E1:
    order = [n for n in E1 if not DF[(DF.exp == "E1") & (DF.name == n)].empty]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for a, met, ttl in ((ax[0], "acc", "Test accuracy"), (ax[1], "macro_f1", "Test macro-F1")):
        m = [DF[(DF.exp == "E1") & (DF.name == n)][met].mean() for n in order]
        s = [DF[(DF.exp == "E1") & (DF.name == n)][met].std(ddof=1) for n in order]
        a.bar(range(len(order)), m, yerr=s, capsize=4, color="tab:blue")
        for k, n in enumerate(order):
            a.scatter([k] * len(SEEDS), DF[(DF.exp == "E1") & (DF.name == n)][met], color="k", s=14, zorder=3)
        a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=45, ha="right")
        a.set_ylabel(ttl); a.set_ylim(max(0, min(m) - 0.05), 1.0); a.set_title(f"{ttl} (mean +- SD, {len(SEEDS)} splits/seeds)")
    plt.tight_layout(); plt.savefig(f"{FIGS}/e1_component_ablation.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for n, col in (("base", "tab:blue"), ("full", "tab:orange")):
        H = [np.load(f"{RUNS}/E1__{n}__s{s}.npz") for s in SEEDS if os.path.exists(f"{RUNS}/E1__{n}__s{s}.npz")]
        for a, key, lab in ((ax[0], "val_f1", "Validation macro-F1"), (ax[1], "val_ce", "Validation cross-entropy")):
            M = np.stack([h[key] for h in H]); x = np.arange(1, M.shape[1] + 1)
            a.plot(x, M.mean(0), color=col, label=n); a.fill_between(x, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=col, alpha=0.2)
            a.set_xlabel("Epoch"); a.set_ylabel(lab); a.legend(); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIGS}/e1_validation_curves_mean_sd.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for a, n in zip(ax, ("base", "full")):
        cm = confusion_matrix(*[load_npz("E1", n, SEEDS[0])[k] for k in ("y",)], load_npz("E1", n, SEEDS[0])["probs"].argmax(1))
        a.imshow(cm, cmap="Blues"); a.set_xticks(range(C)); a.set_yticks(range(C))
        a.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=13); a.set_yticklabels(CLASSES, fontsize=13)
        for i in range(C):
            for j in range(C):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_title(f"{n} (split {SEEDS[0]})"); a.set_xlabel("Predicted"); a.set_ylabel("True")
    plt.tight_layout(); plt.savefig(f"{FIGS}/e1_confusion_matrices.png", dpi=150); plt.close()

# %% [17] Pack results ---------------------------------------------------------------
shutil.make_archive(f"{OUT}/apple_ablation_results", "zip", OUT, "tables")
shutil.make_archive(f"{OUT}/apple_ablation_figs_runs", "zip", OUT, "figs")
print("Done. Download /kaggle/working/tables, /kaggle/working/figs, env.json (and runs/*.json if wanted).")
