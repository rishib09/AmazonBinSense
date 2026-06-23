# BinSense AI — Project Instructions

## ⚠️ Canonical location — read first

**This D: drive repo is the one true working directory for BinSense.**
`D:\GitHub_Repo\Amazon BinSense\` — the git repo. Always open Claude Code here.

Code and data are deliberately split (a `.venv` inside Google Drive was causing
constant sync failures):

| What | Where |
|---|---|
| **Code** (notebooks, tools, utils, docs) — git-tracked | `D:\GitHub_Repo\Amazon BinSense\` (here) |
| **Data** (images, metadata, splits, labels) — NOT in git | `G:\My Drive\Interview Kickstart\Capstone Project\Amazon BinSense\data\` |
| **`.venv`** — git-ignored | here, on D: |

If you ever find yourself launched from the G: Drive path, stop — that folder
holds **data only**. Switch to this D: repo.

## How code finds data

`utils/env_utils.py` resolves two independent roots:
- **code root** (`base_dir`) — from `__file__` or `BINSENSE_DIR`
- **data root** (`data_dir`) — from `BINSENSE_DATA_DIR`, else `base_dir/data`

Notebooks set `BINSENSE_DATA_DIR` in their **Cell 1 bootstrap**:
- **Colab**: data root = the Google Drive mount; code root = this repo
  (currently the Drive path; switch to a `git clone` once the GitHub remote is set).
- **Local (here)**: set `BINSENSE_DATA_DIR` to the G: `data` path, or symlink
  `D:\GitHub_Repo\Amazon BinSense\data` → the Drive `data` folder.

Every notebook starts with:
```python
from utils.env_utils import setup_env
cfg = setup_env()          # cfg.images_dir, cfg.splits_dir, cfg.models_dir, ...
```

## Project shape

Order (ASINs + qty) + bin image → YOLO 1-class detect (count) → metric-learning
embedder → FAISS gallery → constrained match → verify presence + count.
Full plan: `PROJECT_PLAN.md`. Dataset: 3,875-bin curated subset, 5,285 ASINs,
anonymous S3 (`aft-vbi-pds`, no creds).

## Conventions

- **One notebook per step**: `01_download` → `02_eda_and_splits` → `03_labeling`
  → `04_train_yolo` → `05_embedder` → `06_gallery_selftrain` → …
- **Colab-first** for anything needing a GPU; notebooks must also run locally.
- `data/splits/*.csv` **is** git-tracked (small reproducibility artifacts);
  `data/images`, `data/metadata`, `data/labels`, `.venv`, `mlruns`, `models` are not.

## Status (as of 2026-06-23)

- M1 download ✅ · M2 EDA+splits notebook written, **run it in Colab to produce
  `seed.csv`/`extend.csv`/`eval.csv`** · M3 labeling started · M4+ not begun.
- Git: repo init'd locally; **GitHub remote not yet connected** — first commit pending.
