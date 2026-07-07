# Related Work — ABID Challenge (silverbottlep/abid_challenge)

> Comparative analysis vs BinSense. Same dataset, same high-level goal, different method
> and ambition. Use this as the "beat the published baseline" reference in the report.
> Anchors to [`00_PROJECT_ANCHOR.md`](00_PROJECT_ANCHOR.md).
> Repo: <https://github.com/silverbottlep/abid_challenge>

## 1. What ABID is
A 2017-era **research baseline** on the Amazon Bin Image Dataset. It reframes every task
as **whole-image CNN** work — no detection, no crops, no retrieval, no index.

| ABID task | Their method | Their result |
|---|---|---|
| Counting | ResNet-34 **classifier** (whole image → one of 6 classes, 0–5), trained from scratch | **55.67%** acc, RMSE **0.930** |
| Object verification | Siamese ResNet-34 on **image pairs** ("do these share a common object?") | **76.3%** acc |
| Quantity verification | Binary "is there N of X?" | — |

Dataset: full **535,234** images (481K train / 53K val). Moderate = bins ≤5 objects; Hard = all bins.

## 2. Where ABID excels (give it credit)
- **Zero manual labeling.** The metadata count *is* the label → they trained on all 535K
  images cheaply. This is its single biggest advantage over us.
- **Clean, reproducible baseline.** Simple scripts, 40 epochs, published pretrained weights,
  clear moderate/hard eval protocol. Easy to reproduce and cite.
- **Honest difficulty signal.** Their numbers (counting drops to **44.9%** at qty 5;
  164,255 categories appear once) are hard evidence that (a) counting-from-pixels breaks
  under occlusion for *everyone*, and (b) the long tail forbids closed-set classification —
  both of which **validate our design choices.**

## 3. Where ABID falls short (the problems we fix)
- **No attribution.** Whole-image "total = 0–5" can't say *which* ASIN the count belongs to.
  It structurally **cannot** answer our real question: verify an order of specific ASINs ×
  specific quantities.
- **Closed-set classification on an open-set problem.** 164K single-occurrence categories
  guarantee a low ceiling — hence 55.67%.
- **No localization / explainability.** A black-box "3" gives no crops, no per-item scores,
  no threshold — none of the UI our brief mandates.
- **Pairwise Siamese doesn't scale.** Verifying a crop against K order candidates needs K
  full forward passes (worse with their majority voting); there's no reusable index and no
  way to grow the catalog without re-inference. (Detailed rationale in the design notes.)
- **Counting is really a fullness prior.** The classifier "counts" occluded bins by
  memorizing that stuffed-looking bins hold ~N — not by perceiving items. Unreliable at high N.

## 4. What we reuse
- **Benchmark to beat.** Cite 55.67% / RMSE 0.930 (counting) and 76.3% (verification) as the
  published baseline; show our pipeline beating it. *Caveat:* their numbers are on the full
  535K set — for an honest apples-to-apples row, retrain their ResNet-34 counter on **our
  3,875-bin subset** as the naïve baseline.
- **Eval protocol.** Adopt their **moderate (≤5) vs hard (all)** split so our metrics are
  directly comparable.
- **Data-prep plumbing.** `make_metadata.py`, `random_split.py`, `resize_image.py` (224×224)
  parse the same JSON schema — reference implementations, not blind copies.
- **Backbone init (optional).** Their ResNet-34 saw the same visual domain (glare, mesh,
  packaging), so its **backbone** is a plausibly-better init than ImageNet for our crop
  embedder — discard the pairwise head, retrain with a metric loss on crops.
- **Fullness prior (optional, later).** Their whole-image count regressor as a *weak
  secondary signal* to disambiguate our `visible < expected` case (plausibly occluded vs
  actually missing). Enhancement, not core.

## 5. What we improve (BinSense deltas)
| Dimension | ABID | BinSense |
|---|---|---|
| Counting | whole-image classifier (no localization) | **YOLO** instance detection → count per SKU |
| Identity | pairwise Siamese, no index | **metric embedder → FAISS gallery → constrained** top-1 retrieval vs the order's ASINs |
| Quantity | binary "N of X?" | **visible count vs invoice**, per SKU, with occlusion-aware flag regimes |
| Open set / long tail | ignored (closed-set) | core driver → retrieval + **staged self-training / pseudo-labeling** |
| Explainability / UI | none | crops, per-item similarity scores, tunable threshold, downloadable report |
| Scope | train + eval scripts | Train → Productionize → Operate (MLOps) → **Deploy (EC2)** |

## Bottom line
ABID solves a **coarser, category-agnostic** version of our problem and pays for it with a
low ceiling and no attribution. We take on manual labeling (seeded, then SAM/pseudo-labeled)
to gain **localization + per-SKU identity + explainability** — everything order verification
actually requires — and accept that neither approach truly counts *occluded* items, which we
handle with honest flag/HITL regimes rather than a confident guess.
