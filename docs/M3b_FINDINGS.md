# M3b Findings — Auto-Labeling Is Not Viable on ABID Bins → Pivot to Siamese Similarity

*Dated 2026-07-09. This is a **negative result** and a **course-correction**, both
deliberate deliverable content (the objectives reward an "innovative mindset" — that
includes exploring an approach, measuring it, and setting it aside on evidence).*

---

## The question M3b asked

Can we scale detector training labels past the 120-bin manual seed **without** poisoning
the YOLO detector with noisy auto-labels — and prove which labels are trustworthy?
(GitHub issue #2.)

## What we built

A pure-CPU gating engine (`tools/labeling/autolabel.py`) and three notebooks that run
SAM + a second detector over the `extend` bins and compare label sets:

- **`03b` Track 1 — raw union (control):** SAM ∪ zero-shot, geometry filter + NMS only.
- **`03c` Track 2 — gated (treatment):** geometry → **cross-method agreement (SAM ∩ zero-shot)** → count-plausibility.
- **`03d` Track 2 variant — M4 as second opinion:** SAM ∩ **M4 detector** (item-trained), same gates.

## The evidence (all on the same ~200 `extend` bins, held out of eval)

| Experiment | Result | Read |
|---|---|---|
| **Raw union (control)** | **20.5 boxes/bin** vs expected ~6; scatter entirely above the diagonal | Massive over-detection — the noise floor, as designed |
| **SAM ∩ zero-shot** | **7 / 202 bins** accepted, **9 boxes**, survivors on **dividers/tape** | Consensus starved — zero-shot is near-blind on cluttered bins; the few agreements are *correlated errors* (both methods box the same non-item) |
| **SAM ∩ M4** | **27 / 60** accepted but only **~1.5 boxes/bin** (41 total), survivors on **striped texture / edges** | Slightly less starved, still wrong. Acceptance rose only because the count gate does not reject *under*-counting (1 box for a 15-item bin "passes") |

## Root cause (structural, not a tuning problem)

ABID bins are **cluttered, occluded, text-heavy, behind plastic**. Given that:

1. **SAM automatic mode segments *texture*, not items** — it boxes letters ("MORE THAN A COOKIE JAR"), stripes, dividers.
2. **A generic open-vocab detector (YOLO-World) is blind** — warehouse items don't match "product/box/bottle", so it returns almost nothing.
3. **An item-trained detector (M4) shares SAM's texture blind spot** — so where SAM and M4 "agree", they agree on the same non-items. Consensus cannot rescue this; it *amplifies* the correlated error.

Every auto-label path either produces noise (raw union), starves (consensus), or confirms
the wrong boxes (correlated errors). **Fully-automatic labeling does not yield trustworthy
boxes on this data.**

## The cross-check that reframed everything

Re-reading the evaluator template notebooks (`docs/Copy of Week 4 & 5.ipynb`) revealed the
**graded core model is a Siamese similarity network** — positive pairs (same item/quantity)
vs negative pairs (different), backbones **VGG16 vs ResNet50**, verified by **ROC + threshold
optimization + confusion matrix**. It runs on **whole bin images** (`root/images` + `metadata`).
There is **no detection, no bounding boxes, no counting** anywhere in the provided templates,
and the Objectives doc frames the task as *verification/similarity*, not detection.

**Implication:** the detection-first architecture (YOLO → embedder → FAISS → constrained
matching) was more ambitious than the task requires, and the detection **labeling** — the
part that failed — is an *optional, self-imposed* sub-problem. The core deliverable does not
depend on it.

## Decision

**Pivot to Siamese-first** (see decision log D9–D11 in `DATA_FLOW.md`):

- **Core (critical path):** build the Siamese similarity model per the template — train on
  single-ASIN / whole-bin pairs, compare VGG16 vs ResNet50, verify with ROC/threshold/
  confusion matrix. This directly answers "are the ordered items present?" and needs **no
  detection labels**.
- **Detection (optional add-on):** keep the M4 detector (honest baseline, mAP@50 0.255) and
  this M3b finding as the *counting / innovation* story for the video. Not on the critical path.
- **Dropped:** scaling detection labels (the manual 130→250 push and the Label Studio SAM-assist
  setup are no longer needed for the core deliverable).

## What carries forward

- `autolabel.py` + `03b/03c/03d` + this document stay in the repo as the **auditable record**
  of the explored-and-set-aside branch (Level-3 video material: "here's what I tried, measured,
  and why I changed course").
- The M4 baseline weights remain as the optional detector.
- Next milestone: **M5 — Siamese similarity network** (new critical path).
