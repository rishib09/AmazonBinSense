# BinSense AI — Project Anchor (source of truth)

> **Why this file exists.** This is the *hook* we return to before every milestone
> and every deliverable, to confirm we are building what the problem statement
> actually asks for — not what we drifted into. If a decision can't be traced back
> to something on this page, it needs an explicit, documented reason.
>
> Sources: mentor brief "BinSense AI" (Drive `1miJfMW4dAczw2nchFwwhpqsXZ8koJogr`),
> `problem_statement.md`, `Objective_and_deliverables.docx`,
> `metadata_and_solution.docx`, `Frontend.docx`, and a real metadata JSON.
> Full execution plan lives in [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md).

---

## 1. Problem statement

E-commerce fulfillment centers must, at scale, confirm that the **right items in the
right quantities** are physically in a bin before packing. Doing this manually is
slow (≈45 min/order cited in the brief) and error-prone (1–3% pick error). BinSense
automates it with computer vision.

**One line:** *Given an order (items + quantities) and a bin image, verify whether
those items — and the right counts — are actually in the bin.*

---

## 2. The dataset & metadata

**Amazon Bin Image Dataset** — bin photos from an operating Amazon Fulfillment
Center (robot-carried pods). Full set is 535K images; **we use a curated
subset of 3,875 bins** (list = Drive `1rSBsWpSUyZ1Wt-mGJZrZH1Vu0JegTnJY`).

- Bins hold **multiple product types and multiple instances**. A **tape** across the
  front keeps items in — and often makes them unclear. **Occlusion is expected and
  acknowledged by the brief**, not an anomaly.
- Bucket: `aft-vbi-pds`, prefix `/bin-images/`. ⚠️ The brief says "AWS credentials
  needed to download" — **this is wrong for our subset**; the bucket is anonymous/open.
  Credentials are only needed for the EC2 deploy.

**Metadata JSON (one per bin)** — this is the ground truth of bin contents:

```json
{
  "BIN_FCSKU_DATA": {
    "B0033UNIQC": {
      "asin": "B0033UNIQC",
      "name": "Fiskars 8 Inch ... Scissors",
      "quantity": 14,
      "height": {"unit": "IN", "value": 0.70},
      "length": {"unit": "IN", "value": 10.50},
      "width":  {"unit": "IN", "value": 4.00},
      "weight": {"unit": "pounds", "value": 0.20}
    },
    "B00E9J3MLM": { "...": "..." , "quantity": 1 }
  },
  "EXPECTED_QUANTITY": 15
}
```

- One entry per **ASIN** (distinct product); `quantity` = units of that ASIN.
- `EXPECTED_QUANTITY` = **total units** across all ASINs (here 14 + 1 = 15).
- Subset facts (measured): **5,285 unique ASINs**, mean 4.6 units/bin, ≈90% of bins
  are multi-product, 46% of ASINs appear in only one bin (near one-shot).

---

## 3. Objective (verbatim)

> **"Use the dataset (images + metadata) and develop a highly accurate and fast
> computer vision model to verify if the items with their respective quantities are
> present in the image of the bin."**

Worked example from the brief — an order comes in:

| Item in order | Quantity |
|---|---|
| Nerf Super Soaker Zipfire 3-Pack | 2 |
| Aurora Master Ocean Relax Projector | 1 |
| Samsung ML1650D8/XAA Toner/Drum | 1 |

> *"Now you get an image of the bin where items in the order are present. Your
> objective is to validate whether the items in the order are the ones in the bin."*

**Direction of validation:** order → checked against → image. Report, per line,
**present / absent** and **quantity match / mismatch**.

---

## 4. Solution

### 4.1 Locked architecture
```
Order (ASINs + qty) ─┐
                     │        Bin image
                     │            │
                     │   ┌────────▼────────┐
                     │   │ YOLO (1 class)  │  detect every item instance → COUNT
                     │   └────────┬────────┘
                     │       crops│
                     │   ┌────────▼────────┐
                     │   │ Embedder        │  metric-learning (Siamese-style)
                     │   └────────┬────────┘
                     │            │ cosine / NN
                     │   ┌────────▼────────┐
                     └──▶│ FAISS gallery   │  match each crop ONLY against the
       candidate ASINs   │ (SKU embeddings)│  order's candidate ASINs (constrained)
                         └────────┬────────┘
                         ┌────────▼────────┐
                         │ Verify present? │  presence + count vs order → report
                         └─────────────────┘
```
Identity grounding = **staged self-training**: seed embedder/gallery from
single-ASIN bins → pseudo-label multi-ASIN bins → iterate. See `PROJECT_PLAN.md`.

The brief's own "Solution Overview" frames the same shape:
*Input → Image Processing → Item Detection → Quantity Validation → Output*, and
explicitly says to **"utilize metadata for cross-validation."**

### 4.2 ⚠️ Governance note — how we treat the metadata (READ BEFORE DEVIATING)
Metadata plays two roles: it **is the order** we validate against, *and* it is the
**ground-truth label** we're evaluated on ("detecting if the item with its respective
quantity is present"). But the brief does **not** say "trust it blindly" — it says
**"utilize metadata for cross-validation"** and **"apply quality checks through
multiple validation layers"** (`metadata_and_solution.docx`). So metadata is a
**cross-check, not an unquestioned oracle.**

**Our stance (doc-aligned):**
- The model **predicts counts/identity from pixels independently** — it never reads
  the answer from metadata at inference.
- When model output **disagrees** with metadata, that is a **flag + log entry** — a
  validation layer (per the brief) and a **drift signal** (MLOps 15%).
- A disagreement usually means the **image** couldn't show everything (occlusion /
  tape) — an image/model limitation, **not** a metadata error. Do **not** frame this
  as "the metadata is wrong."
- The **human review + retrain** is the *documented MLOps response* to accumulated
  flags — **not** a core inference dependency. Keep it lightweight; do not build the
  full labeling flywheel before the core prototype (D1/D2) exists.
- The **tampered-order simulator** is an **evaluation tool** to measure validation
  accuracy — not a product feature. Tamper detection is only an *adjacent* use case
  in the brief.

---

## 5. Deliverables (brief → our mapping)

| # | Brief requires | Our vehicle |
|---|---|---|
| D1 | Working prototype: CV system for **item + quantity validation**, high accuracy | Streamlit/Gradio app (M8), containerized (M9) |
| D2 | UI: pick items+quantities → select matching image → run model → show result + downloadable report | Frontend flow in `Frontend.docx` (M8) |

**Required UI features (from `Frontend.docx` — not optional):**
- A **tunable similarity threshold** (ROC-tuned cosine distance) exposed as a UI slider.
- **Per-item similarity scores** shown alongside the present/absent + quantity status.
- A **downloadable verification report** as the output.
- Speed matters: the brief repeatedly asks for "fast" / "reduce processing time" →
  hold the <2–3 s per-bin latency target (`PROJECT_PLAN.md §6`).
- *Stretch (optional):* inventory-availability dashboard with synthetic stock (20–100).
| D3 | Saved model files / state dicts | YOLO + embedder + FAISS artifacts (M4/M5/M6) |
| D4 | Step-by-step **EDA** code + visualizations + decisions | `notebooks/02_eda_and_splits.ipynb` ✅ |
| D5 | Step-by-step **model** code: selection, tuning, evaluation | notebooks 04–07 |
| D6 | **MLOps** code: deploy, monitor, log, re-train (+ screen-recorded video) | MLflow/Airflow/Prometheus/Grafana (M10–M13), EC2 (M14) |
| D7 | Annotated data (optional) | YOLO labels (M3) |
| D8 | Comprehensive presentation + documentation | M15 |

**Frontend flow (from `Frontend.docx`):** startup (load dataset, init UI) →
model selection (pick model + threshold) → order creation (items + quantities) →
verification (per item: compare to bin, similarity score, status) →
results (verdict, package metrics, downloadable report).

---

## 6. Evaluation criteria (what we're graded on)

| Weight | Criterion | Implication for us |
|---|---|---|
| 20% | Understanding of objectives | Build for accurate + efficient + **productionized** item-and-quantity validation |
| 20% | Selection & application of CV models | EDA-driven, documented experimental plan |
| 20% | **Evaluation of CV models** | Justify metrics; align to dataset reality (occlusion, long tail) — see `PROJECT_PLAN.md §6` |
| 15% | MLOps | Automated deploy, monitoring, logging, re-training |
| 10% | UI/UX | Intuitive, fast, clear errors |
| 8%  | Innovation & creativity | Where our self-training + HITL enhancements earn credit |
| 7%  | Presentation & documentation | Clear, comprehensive |

---

## 7. The anchor checklist (run this before each milestone)

1. Does this work advance **item-and-quantity validation of a bin image against an
   order** — or have we drifted into a side quest?
2. Are we still **measuring counts from pixels**, not reading the answer from metadata?
3. If we're treating metadata as non-golden, is that **explicitly justified** (§4.2)?
4. Which **deliverable (D1–D8)** and **eval criterion** does this feed?
5. Is it **documented** (EDA/model/MLOps step, decision, and metric)?
