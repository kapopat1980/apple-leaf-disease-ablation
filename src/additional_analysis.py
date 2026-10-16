"""Additional paired analyses computed from the Kaggle output tables (all_runs.csv etc.).
Usage: python additional_analysis.py <tables_dir> <out_dir>
Every comparison is paired by seed (same stratified split + same seed for all configs)."""
import sys, os, math, numpy as np, pandas as pd
from scipy import stats
T = sys.argv[1] if len(sys.argv) > 1 else "res/tables"
O = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(O, exist_ok=True)
D = pd.read_csv(f"{T}/all_runs.csv")
REF = int(D[(D.exp == "E1") & (D.name == "base")].res.iloc[0])                      # E1 input resolution (64 for PlantVillage, 128 for Plant Pathology)
E2R = sorted({int(n.split("@")[1]) for n in D[D.exp == "E2"].name.unique()})        # E2 resolutions
CMP = max(E2R)                                                                       # small-CNN reference for the E3 comparison

def get(exp, name, col):
    g = D[(D.exp == exp) & (D.name == name)].sort_values("seed")
    return g.set_index("seed")[col]

def paired(a, b, label, col):
    """delta = b - a, paired by seed; t-based 95% CI, exact Wilcoxon, #seeds improved."""
    d = (b - a).dropna().values
    n = len(d); m = d.mean(); sd = d.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    try: pw = stats.wilcoxon(d).pvalue
    except ValueError: pw = np.nan
    return dict(comparison=label, metric=col, mean_a=a.mean(), mean_b=b.mean(), mean_delta=m, sd=sd,
                ci95_lo=m - h, ci95_hi=m + h, seeds_b_better=int((d > 0).sum()), seeds_a_better=int((d < 0).sum()),
                n=n, paired_t_p=stats.ttest_1samp(d, 0).pvalue if sd > 0 else np.nan, wilcoxon_p=pw)

rows = []
for col in ("acc", "macro_f1"):
    # E2: resolution factor, and full-vs-base at matched resolution
    for cfg in ("base", "full"):
        for r in E2R:
            rows.append(paired(get("E1", cfg, col), get("E2", f"{cfg}@{r}", col), f"{cfg}: {r}px vs {REF}px", col))
    rows.append(paired(get("E1", "base", col), get("E1", "full", col), f"full vs base @{REF}", col))
    for r in E2R:
        rows.append(paired(get("E2", f"base@{r}", col), get("E2", f"full@{r}", col), f"full vs base @{r}", col))
    # E3: initialisation and auxiliary components
    for arch in ("resnet18", "efficientnet_b0"):
        rows.append(paired(get("E3", f"{arch}_rand_base", col), get("E3", f"{arch}_pre_base", col), f"{arch} base: pretrained vs random", col))
        rows.append(paired(get("E3", f"{arch}_rand_aux", col), get("E3", f"{arch}_pre_aux", col), f"{arch} aux: pretrained vs random", col))
        for ini in ("rand", "pre"):
            rows.append(paired(get("E3", f"{arch}_{ini}_base", col), get("E3", f"{arch}_{ini}_aux", col), f"{arch} {ini}: aux vs base", col))
        rows.append(paired(get("E2", f"base@{CMP}", col), get("E3", f"{arch}_pre_base", col), f"{arch} pretrained base vs SmallCNN base @{CMP}px", col))
        rows.append(paired(get("E2", f"base@{CMP}", col), get("E3", f"{arch}_rand_base", col), f"{arch} random base vs SmallCNN base @{CMP}px", col))
pd.DataFrame(rows).to_csv(f"{O}/extra_paired_tests.csv", index=False)

# TTA effect per configuration (paired by seed)
rows = []
for (exp, name), g in D.groupby(["exp", "name"], sort=False):
    g = g.sort_values("seed")
    for col in ("acc", "macro_f1"):
        a = g.set_index("seed")[col]; b = g.set_index("seed")["tta_" + col]
        rows.append(paired(a, b, f"{exp}/{name}: TTA vs no TTA", col))
pd.DataFrame(rows).to_csv(f"{O}/tta_effect.csv", index=False)

# Stability: compare each config with base on per-run curve statistics
S = pd.read_csv(f"{T}/stability_per_run.csv")
rows = []
for met in ("sd_f1_after_warm", "sd_ce_after_warm", "max_drop", "n_collapses", "epochs_to_95"):
    a = S[S.config == "base"].set_index("seed")[met]
    for cfg in S.config.unique():
        if cfg == "base": continue
        b = S[S.config == cfg].set_index("seed")[met]
        r = paired(a, b, f"{cfg} vs base", met); rows.append(r)
pd.DataFrame(rows).to_csv(f"{O}/stability_tests.csv", index=False)

# Same-split ensemble vs base
EN = pd.read_csv(f"{T}/ensemble_base_se_full_same_split.csv").set_index("seed")
rows = [paired(get("E1", "base", c), EN[c], "ensemble(base+se+full) vs base", c) for c in ("acc", "macro_f1")]
pd.DataFrame(rows).to_csv(f"{O}/ensemble_vs_base.csv", index=False)

# Per-class F1 mean+-SD across seeds for selected configs
cls = [c[3:] for c in D.columns if c.startswith("f1_")]
rows = []
for exp, nm in (("E1", "base"), ("E1", "full"), ("E1", "ls"), ("E1", "se")):
    g = D[(D.exp == exp) & (D.name == nm)]
    r = dict(config=nm)
    for c in cls:
        for k in ("prec", "rec", "f1"):
            r[f"{k}_{c}"] = f"{g[f'{k}_{c}'].mean():.3f} ± {g[f'{k}_{c}'].std(ddof=1):.3f}"
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{O}/per_class_mean_sd.csv", index=False)

# Hard-image coverage
Q = pd.read_csv(f"{T}/image_quality_and_miss_rate.csv")
cov = Q.times_in_test.ge(5).sum()
print("REF res", REF, "E2", E2R, "CMP", CMP)
print("images in >=1 test split (n_test_appearances>=5):", cov, "of", len(Q))
print("total unique images:", len(Q), Q.label.value_counts().to_dict())
hard = Q[(Q.times_in_test >= 5) & (Q.miss_rate >= 0.5)]; easy = Q[(Q.times_in_test >= 5) & (Q.miss_rate == 0)]
print("hard", len(hard), hard.label.value_counts().to_dict(), "easy", len(easy))
print("share of Cedar among covered images:", (Q[Q.times_in_test >= 5].label == "Cedar_apple_rust").mean())
