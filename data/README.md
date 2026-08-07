# Data

This repository does not redistribute the PlantVillage images.

## Source

The apple leaf disease subset used in this study comes from the PlantVillage
dataset (Hughes & Salathé, 2015), specifically the `raw/color` split of the
public mirror maintained at:

https://github.com/spMohanty/PlantVillage-Dataset

## Classes used

| Class | Folder name in source repo | # images |
|---|---|---|
| Apple Scab | `Apple___Apple_scab` | 630 |
| Black Rot | `Apple___Black_rot` | 621 |
| Cedar Apple Rust | `Apple___Cedar_apple_rust` | 275 |
| Healthy | `Apple___healthy` | 1,645 |
| **Total** | | **3,171** |

## Setup

```bash
git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git /tmp/pv
mkdir -p data/apple_data
cp -r /tmp/pv/raw/color/Apple___Apple_scab       data/apple_data/Apple_scab
cp -r /tmp/pv/raw/color/Apple___Black_rot         data/apple_data/Black_rot
cp -r /tmp/pv/raw/color/Apple___Cedar_apple_rust  data/apple_data/Cedar_apple_rust
cp -r /tmp/pv/raw/color/Apple___healthy           data/apple_data/healthy
rm -rf /tmp/pv
```

`src/make_splits.py` expects this exact `data/apple_data/<class>/*.JPG` layout
(update the `ROOT` constant in that script if you place the data elsewhere).
