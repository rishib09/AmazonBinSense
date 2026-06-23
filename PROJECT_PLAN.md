# BinSense AI — Project Game Plan

**Owner:** Rishi (solo) · **Created:** 2026-06-05 · **Target window:** ~6 weeks (Mon 2026-06-08 → Fri 2026-07-17, with refinement buffer Jul 18–19)

> Learning goal: *experience what it takes to productionize an AI/ML application end to end* — train → productionize → operate (monitor + audit) → deploy.

---

## 1. The problem (one line)
Given an **order** (list of products + quantities) and a **bin image**, verify whether those products — and the right counts — are actually in the bin.

## 2. Solution architecture (locked)
```
Order (ASINs + qty)              Bin image
        │                            │
        │                   ┌────────▼────────┐
        │                   │ YOLO (1 class)  │  detect every item instance  → COUNT
        │                   │  "object"       │
        │                   └────────┬────────┘
        │                       crops │
        │                   ┌────────▼────────┐
        │                   │ Embedder        │  metric-learning (Siamese-style)
        │                   │ (per-crop emb.) │
        │                   └────────┬────────┘
        │                            │ cosine / NN
        │                   ┌────────▼────────┐
        └──── candidate ───▶│ FAISS gallery   │  match each crop ONLY against
              ASIN set      │ (SKU embeddings)│  the order's candidate ASINs
                            └────────┬────────┘  (constrained matching)
                                     │
                            ┌────────▼────────┐
                            │ Verify: present?│  presence + count vs order → report
                            └─────────────────┘
```

### Why this shape (data-driven)
Measured over the **3,875-bin** subset (fetched from S3 metadata, 0 errors):

| Distinct product types / bin | bins | % |
|---|---|---|
| 1 (single-ASIN) | 376 | 9.7% |
| 2 | 1,398 | 36.1% |
| 3 | 1,432 | 37.0% |
| 4 | 667 | 17.2% |
| 5 | 1 | ~0% |

- **5,285 unique ASINs** across the subset → average ASIN appears in ~1–2 bins ⇒ **near-one-shot, open-set retrieval, NOT classification.** (Hence metric learning + a gallery, never a 5,285-way classifier.)
- Single-ASIN bins cover only **341 ASINs (6.5%)** and ~**713 clean crops** ⇒ single-ASIN bootstrap is the right **seed** but is **not** sufficient; pseudo-labeling multi-ASIN bins is **mandatory** for coverage.
- **Constrained matching** (compare a crop only to the order's 2–4 candidate ASINs) is what makes identity tractable.

### Identity grounding = staged self-training (Decision: Option A)
1. **Seed** — train embedder + build FAISS gallery from the 341 single-ASIN ASINs (clean labels).
2. **Extend** — run YOLO + seed embedder on multi-ASIN bins; **Hungarian / constrained assignment** of detected boxes → the bin's known ASIN set (from metadata) to pseudo-label new crops; grow the gallery; iterate.
3. The **detector** handles **quantity** independently, so even tail ASINs get counted.
> Coverage is reported as a metric and grows gradually — we will not reach 100% of 5,285, and that's expected.

## 3. Locked decisions (the design tree)
| Decision | Choice |
|---|---|
| Scope / timeline | Full ambition, realistic ~6 weeks solo (2-week MVP = internal checkpoint) |
| Core approach | Detection (YOLO 1-class) → embedding → FAISS → constrained match → verify |
| Labeling | Demonstrate all three: manual seed (LabelImg) → SAM semi-supervised → zero-shot auto-label; train YOLO on result |
| Identity grounding | **A — staged self-training** (single-ASIN seed → pseudo-label multi-ASIN → iterate) |
| Phasing | (1) Train [Colab] → (2) Productionize [Docker app] → (3) Operate [MLflow/Airflow/Prometheus/Grafana] → (4) Deploy [AWS EC2, required] |
| Topology | Colab (GPU/ML) + GitHub (CI spine) + Google Drive (artifacts) + local Docker Compose (services/app) + AWS EC2 (final deploy) |
| Data | 3,875 bins; **anonymous S3** (`aft-vbi-pds`, no creds); fits Google Drive (<~1 GB) |

## 4. Open micro-decisions (resolved at the start of each milestone, not blocking)
- Embedder backbone: ResNet50 vs MobileNetV2 vs CLIP image encoder (decide in M5).
- YOLO version: YOLOv8 vs YOLOv11 (Ultralytics) (decide in M4).
- Zero-shot labeler: GroundingDINO vs YOLO-World vs SAM-assisted (decide in M3).
- App UI: Streamlit (course default) vs Gradio (decide in M8).
- Order simulation design for eval (decide in M7).

---

## 5. Time-bound milestones

### Week 0 — Setup (weekend, Jun 6–7) *(light)*
- GitHub repo + structure; Colab Pro confirmed; Google Drive folders; Docker Desktop installed.
- **Exit:** repo pushes, Colab mounts Drive, `docker run hello-world` works.

### Week 1 — Data foundation (Jun 8–14) · *maps to course Wk 2–3*
- **M1 Download tool** — anonymous S3 fetch of all 3,875 images + JSON to Drive; resume, progress, error handling; reusable module.
- **M2 EDA + splits** — JSON→DataFrame; EDA (item/qty/weight/items-per-bin) + image-quality pass; create splits: **seed** (single-ASIN), **extend** (multi-ASIN), **eval** (held-out).
- **Exit:** dataset in Drive + EDA notebook + committed split manifests.

### Week 2 — Detection (Jun 15–21) · *new (whiteboard route)*
- **M3 Labeling pipeline** — manual seed (~50–100 bins, LabelImg) → SAM-assisted semi-supervised → zero-shot auto-label remainder; produce YOLO-format labels.
- **M4 Train YOLO (1 class)** — train + tune; evaluate detection (mAP@50, total-count accuracy ±1).
- **Gold set:** hand-verify ~100-bin detection eval set (held out).
- **Exit:** trained detector + detection/count metrics report.

### Week 3 — Identity + end-to-end (Jun 22–28) · *maps to course Wk 4–5*
- **M5 Embedder + gallery** — metric-learning embedder (transfer from pretrained backbone) on seed crops; build FAISS gallery (341 ASINs).
- **M6 Self-training** — constrained Hungarian matching → pseudo-label multi-ASIN bins → grow gallery → iterate ≥1 round; report catalog coverage.
- **M7 End-to-end verification + eval** — order→bin pipeline; order simulator; metrics (below). **This is the "desired results" gate.**
- **Gold set:** hand-label box→ASIN on ~30–50 multi-ASIN bins for identity eval. **Baseline all metrics here, then commit final target bands.**
- **Exit:** end-to-end pipeline + measured metrics vs targets.

### Week 4 — Productionize (Jun 29 – Jul 5) · *maps to course Wk 6–7*
- **M8 App** — Streamlit/Gradio: order entry → image select → verify → downloadable report.
- **M9 Containerize** — Dockerfile for app; `docker-compose` foundation; wire MLflow as model source.
- **Exit:** `docker compose up` serves the app locally on :8501 using a registered model.

### Week 5 — Operate / MLOps (Jul 6–12) · *whiteboard stretch, now in-scope*
- **M10 MLflow** — experiment tracking + model registry (retrofit training runs; audit trail).
- **M11 Airflow** — DAG orchestrating the retrain / pseudo-label / re-index loop.
- **M12 Prometheus + Grafana** — app + model metrics dashboards; basic drift signal.
- **M13 GitHub Actions** — CI: lint/tests/build image.
- **Exit:** full local stack via compose; dashboards live; CI green.

### Week 6 — Deploy + finalize (Jul 13–17)
- **M14 AWS EC2** — ECR push → EC2 run (app on :8501); security group; run instructions.
- **M15 Docs + presentation + video** — architecture, decisions/ADRs, MLOps strategy, user guide, recorded demo.
- **Exit:** live on EC2 + complete submission package.

### Refinement buffer (Jul 18–19+)
- Optimize accuracy/coverage, retrain, polish UI/docs.

---

## 6. "Desired results" — metrics (bands are proposed; baseline first at M7, then commit)

**Gold eval sets (hand-verified, held out from all training/gallery):**
- ~100 bins with hand-checked boxes → detection metrics.
- ~30–50 multi-ASIN bins with hand-labeled box→ASIN → identity metrics.
(Required because training labels are auto-generated; evaluating against them would only measure agreement with the labeler.)

### Tier 1 — Business gates (= "desired results")
| Metric | Band | Definition / how measured |
|---|---|---|
| End-to-end order-verification accuracy | ≥ 75% | Order simulator makes true + tampered orders (drop line / add absent product / wrong qty) over gold bins; judge each line match/mismatch. Report **per-line** and **per-order**. |
| ↳ Recall on tampered lines | ≥ 80% | Catching wrong orders matters more than raw accuracy (warehouse context). |
| Total-count accuracy (±1 / bin) | ≥ 75% | `#detected` vs `EXPECTED_QUANTITY` within ±1. Also report exact-match % and RMSE. |

### Tier 2 — Component diagnostics
| Metric | Band | Definition |
|---|---|---|
| Detection mAP@50 (1-class) | ≥ 0.60 | AP of `object` at IoU 0.50 on the 100-bin gold set (Ultralytics `val`). |
| Identity top-1, single-ASIN crops | ≥ 80% | Clean known-ASIN crops → nearest gallery neighbor correct. Isolates embedder quality. |
| Constrained within-bin top-1 (multi-ASIN) | ≥ 70% | Hungarian assign boxes → bin's candidate set; correctness on the 30–50-bin gold set. |

### Tier 3 — Progress signals (not pass/fail)
| Metric | Definition |
|---|---|
| Catalog coverage after self-training | % of 5,285 ASINs with ≥1 gallery entry; seed = 6.5% (341), report growth per round + gallery-size distribution. |
| Inference latency | detect→embed→match per bin, target < 2–3 s on Colab GPU (objective values speed). |

**Chaining:** end-to-end accuracy is a product of detection + counting + identity; the Tier-2 diagnostics localize the bottleneck. Identity decisions use a cosine-distance threshold tuned via ROC (course Wk 4–5 step; mirrored by the UI similarity-threshold slider).

## 7. Top risks & mitigations
- **Occlusion / tape** wrecks detection → rely on count-tolerance metrics; zero-shot labeler reduces manual cost.
- **Long-tail catalog** limits identity coverage → constrained matching + report coverage honestly; demo on seeded ASINs if needed.
- **Docker/MLOps learning curve** → front-load a Docker spike in Week 4; bring services up one at a time.
- **Colab disconnects on long training** → Colab Pro; checkpoint to Drive; keep runs short/resumable.
- **Scope creep** → phases are independently demo-able; if time slips, ship through M9 (working containerized app) and treat M10–M14 as stretch.

## 8. Deliverables (per mentor brief)
Codebase · documentation (architecture, approaches, decisions, MLOps) · run instructions · recorded video.
