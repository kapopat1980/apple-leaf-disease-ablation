# Apple Leaf Disease Classification: An Empirical Ablation of Attention, Focal Loss, and Physically-Inspired Augmentation

This repository accompanies the manuscript *"Does Added Model Complexity Help on a
Near-Saturated Benchmark? An Empirical Ablation Study of Apple Leaf Disease
Classification on PlantVillage"* by Divyakant Meva and Kalpesh Popat.

## Summary

We benchmark a plain CNN baseline against progressively more sophisticated
variants — squeeze-and-excitation (SE) channel attention, focal loss, class
weighting, physically-inspired augmentation (CLAHE / illumination / sensor
noise), test-time augmentation, and a two-model ensemble — on the apple leaf
disease subset of the PlantVillage dataset (3,171 images; 4 classes: Apple
Scab, Black Rot, Cedar Apple Rust, healthy). All models are trained **from
scratch** (see `docs/manuscript.docx`, Section 3.5, for why pretrained
ImageNet weights were not used).

**Headline finding:** none of the added-complexity variants outperformed the
plain baseline CNN on this dataset. The baseline achieved 99.37% test
accuracy / 0.9928 macro-F1 (3 errors out of 479 test images); every other
variant performed at or slightly below that level. We report this as a
genuine ablation result rather than adjusting the narrative to fit a
"proposed method wins" template — see the manuscript for full discussion.

## Repository structure

```
.
├── src/
│   ├── make_splits.py     # stratified 70/15/15 train/val/test split
│   ├── common.py          # baseline + SE-attention model, focal loss, 64px augmentation
│   ├── common_v3.py       # v3 model: 96px input, lighter/targeted augmentation, label smoothing
│   ├── train.py           # trains baseline or v1 "proposed" model
│   ├── train_v3.py        # trains the v3 model
│   ├── evaluate.py        # test-set evaluation for baseline vs v1 proposed
│   ├── evaluate_v3.py     # test-set evaluation for v3 + TTA + ensemble with baseline
│   └── make_plots.py      # confusion matrices / training curves / per-class F1 bar chart
├── results/                # all raw metrics (JSON) and figures (PNG) reported in the manuscript
├── docs/
│   └── manuscript.docx    # full write-up
├── data/
│   └── README.md          # how to obtain the dataset (not redistributed here)
├── requirements.txt
└── LICENSE
```

## Reproducing the results

1. Download the apple leaf subset of PlantVillage (see `data/README.md`) into `data/apple_data/<class_name>/*.jpg`.
2. `pip install -r requirements.txt`
3. `cd src && python make_splits.py` — writes `splits.json`
4. `python train.py baseline 25` and `python train.py proposed 25` — trains the baseline and v1 attention/focal-loss/augmentation model
5. `python train_v3.py` — trains the v3 (higher-resolution, lighter-augmentation) model
6. `python evaluate.py` — test-set metrics for baseline vs. v1 proposed
7. `python evaluate_v3.py` — test-set metrics for v3 (with TTA) and the baseline+v3 ensemble
8. `python make_plots.py` — regenerates the figures in `results/`

All training was run on CPU only (single core, ~4 GB RAM); no GPU or
pretrained weights were used. Expect roughly 1–4 minutes per epoch depending
on model/resolution.

## Environment constraint (please read before comparing to the literature)

This work could not access pretrained ImageNet weights (no route to
`download.pytorch.org` / `huggingface.co` from the training environment), so
none of the models here use true transfer learning. Several works in
`docs/manuscript.docx`'s reference list report higher accuracy using
pretrained EfficientNet/ResNet/Inception backbones — that comparison is
apples-to-oranges (pun intended) with the from-scratch models in this repo.
If you have GPU/internet access, swapping `AttentiveNetV3`'s backbone for a
pretrained `torchvision` model is the natural next experiment (see Section 6
of the manuscript).

## Authors

- **Divyakant Meva** — Associate Professor, Faculty of Computer Applications, Marwadi University, Rajkot, Gujarat, India
- **Kalpesh Popat** — Faculty of Computer Applications, Marwadi University, Rajkot, Gujarat, India

See `docs/manuscript.docx` for the full CRediT author contribution statement.

## License

Code: MIT License (see `LICENSE`). Dataset: PlantVillage is subject to its
own license/terms — see `data/README.md`.
