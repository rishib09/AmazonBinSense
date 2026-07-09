# BinSense — Data Flow & Decision Map

*Living document. Snapshot: 2026-07-09 (M3b in progress). Renders on GitHub via Mermaid.*

This maps **what data flows where**, **which decisions are locked**, and **how the
current M3b labeling A/B branches into the rest of the project**. Read it top-to-bottom:
(1) the end-to-end pipeline, (2) the M3b labeling detail, (3) how M3b's *result* changes
future work, (4) what is build-time tooling vs what actually ships.

Legend: ✅ done · 🟡 in progress · ⬜ future · 🔒 locked decision

---

## 1. End-to-end pipeline (data → model → product → deploy)

```mermaid
flowchart TD
    S3["Anonymous S3 bucket<br/>3874 bin images + metadata JSON"]:::done
    DL["M1 · Download<br/>tools/download/s3_downloader.py<br/>nb 01"]:::done
    EDA["M2 · EDA + splits<br/>nb 02 → seed / extend / eval CSVs"]:::done

    S3 --> DL --> EDA
    EDA --> SEED["seed (346)"]:::done
    EDA --> EXT["extend (3397)"]:::done
    EDA --> EVAL["eval-gold (130) 🔒 never trained on"]:::done

    SEED --> MAN["M3 · Manual seed labels<br/>120 bins · Label Studio · nb 03"]:::done
    MAN --> M4base["M4 · Detector baseline<br/>YOLOv8s · mAP@50 0.255 ❗honest baseline"]:::done

    EXT --> M3b{{"M3b · Scale labels (A/B)<br/>nb 03b / 03c · issue #2"}}:::wip
    MAN --> M3b
    M3b --> M4next["M4' · Retrained detector<br/>seed + accepted auto-labels"]:::future

    M4next --> M5["M5 · Metric-learning embedder<br/>on detector crops"]:::future
    M5 --> FAISS["M6 · FAISS SKU gallery<br/>staged self-training"]:::future
    FAISS --> MATCH["M7 · Constrained matching<br/>crop vs order's candidate ASINs"]:::future
    MATCH --> VERIFY["Verify presence + count<br/>vs order + EXPECTED_QUANTITY cross-check"]:::future

    VERIFY --> APP["Productionize · Streamlit/Gradio + Docker"]:::future
    APP --> OPS["Operate · MLflow + Airflow + Prometheus + Grafana"]:::future
    OPS --> DEPLOY["Deploy · AWS EC2 🔒 required"]:::future

    classDef done fill:#d4edda,stroke:#28a745,color:#000
    classDef wip fill:#fff3cd,stroke:#ffc107,color:#000
    classDef future fill:#e2e3e5,stroke:#6c757d,color:#000
```

**Where we are:** the detector exists but is data-starved (both gates missed). Everything
downstream (M5→deploy) waits on a detector good enough to produce clean crops. M3b is the
unblock.

---

## 2. M3b in detail — the labeling A/B (current work)

The question M3b answers: **can we scale labels past the 120-bin seed *without* poisoning
the detector with noisy auto-labels — and can we prove it?**

```mermaid
flowchart TD
    subgraph GPU["Colab GPU (inference only)"]
        SAM["SAM auto-mask → boxes<br/>data/labels_auto/sam/"]:::gpu
        ZS["Zero-shot (YOLO-World) → boxes<br/>data/labels_auto/zeroshot/"]:::gpu
    end

    EXT["extend bins"]:::data --> SAM
    EXT --> ZS

    SAM --> ENGINE["tools/labeling/autolabel.py<br/>(pure-CPU gating engine)"]:::engine
    ZS --> ENGINE

    ENGINE --> T1["Track 1 · RAW UNION (control)<br/>union + speck filter + NMS<br/>data/labels_track1/"]:::t1
    ENGINE --> T2["Track 2 · GATED (treatment)"]:::t2

    subgraph GATES["Track 2 gates (a box/bin must survive all)"]
        GC["Gate C · geometry<br/>drop specks / full-frame / slivers"]:::gate
        GB["Gate B · cross-method agreement<br/>keep SAM ∩ zero-shot, IoU ≥ 0.5"]:::gate
        GA["Gate A · count plausibility<br/>reject bin if N≈0 or N ≫ EXPECTED_QUANTITY"]:::gate
        GD["Gate D · seed-model agreement<br/>(optional, off until detector improves)"]:::gate
        GC --> GB --> GA -.-> GD
    end
    T2 --> GATES
    GATES --> T2out["data/labels_track2/ + provenance manifest"]:::t2

    MAN["120 manual seed labels"]:::data --> TRAIN1
    MAN --> TRAIN2
    T1 --> TRAIN1["train M4 on seed + Track 1"]:::future
    T2out --> TRAIN2["train M4 on seed + Track 2"]:::future

    GOLD["Track 0 · gold set<br/>~50 hand-boxed eval bins 🔒<br/>(owner: Rishi, manual)"]:::gold
    TRAIN1 --> SCORE{{"Score both on gold set<br/>mAP@50 + count-within-1"}}:::wip
    TRAIN2 --> SCORE
    M4base["120-seed baseline (mAP 0.255)"]:::data --> SCORE
    GOLD --> SCORE

    classDef gpu fill:#cce5ff,stroke:#0d6efd,color:#000
    classDef data fill:#d4edda,stroke:#28a745,color:#000
    classDef engine fill:#f8d7da,stroke:#dc3545,color:#000
    classDef gate fill:#fff3cd,stroke:#ffc107,color:#000
    classDef t1 fill:#e2e3e5,stroke:#6c757d,color:#000
    classDef t2 fill:#d1ecf1,stroke:#17a2b8,color:#000
    classDef gold fill:#e7d4f5,stroke:#8b5cf6,color:#000
    classDef wip fill:#fff3cd,stroke:#ffc107,color:#000
    classDef future fill:#e2e3e5,stroke:#6c757d,color:#000
```

**Key design choices baked in here:**
- GPU does *only* inference; **all quality logic is CPU + testable** (`autolabel.py`).
- Track 1 exists purely as the **control** — it exposes the noise floor so Track 2 has
  something to beat. It is not a production label set.
- **Gate A operates per-bin** (accept/reject the whole bin); Gates B/C operate per-box.
- Track 0 gold set is the **only trustworthy ruler** — without it, mAP is measured on 18
  noisy val images and can't be trusted.

---

## 3. How the M3b *result* branches future work (impact)

The A/B outcome is a fork that changes what M4→M5 look like:

```mermaid
flowchart TD
    Q{{"Does Track 2 (gated) beat<br/>Track 1 and the 0.255 baseline<br/>on the gold set?"}}:::wip

    Q -->|"gated wins,<br/>mAP ≥ 0.60"| WIN["Adopt gated labels<br/>→ retrain M4 → passes Tier-2<br/>→ unblock M5 crops"]:::good
    Q -->|"gated wins but<br/>mAP still < 0.60"| PARTIAL["Labels help but not enough:<br/>add labels / bigger model / more epochs<br/>OR relax count target with metadata cross-check"]:::warn
    Q -->|"gated ≈ raw union"| NEUTRAL["Consensus gating didn't pay:<br/>use raw union, spend effort elsewhere<br/>(honest Level-3 finding)"]:::warn
    Q -->|"both hurt vs baseline"| BAD["Auto-labels are net noise:<br/>fall back to hand-labeling +<br/>lean on metadata count, not detection"]:::bad

    WIN --> M5["M5 embedder gets clean crops"]:::future
    PARTIAL --> RISK["Watch selection bias:<br/>gates may have skewed training EASY"]:::warn
    NEUTRAL --> M5
    BAD --> RETHINK["Re-scope: is single-photo counting<br/>even achievable? (occlusion ceiling)"]:::bad

    classDef wip fill:#fff3cd,stroke:#ffc107,color:#000
    classDef good fill:#d4edda,stroke:#28a745,color:#000
    classDef warn fill:#fff3cd,stroke:#ffc107,color:#000
    classDef bad fill:#f8d7da,stroke:#dc3545,color:#000
    classDef future fill:#e2e3e5,stroke:#6c757d,color:#000
```

**Why this matters for scheduling (8 days to deadline):** whichever branch we land on,
the required deliverable is the **EC2 deployment**. So M3b is time-boxed — a "good enough"
detector plus an honest report beats a perfect detector with no deployed app.

---

## 4. Build-time tooling vs the shipped product

Not everything in the repo runs in production. This separation drives the final structure:

```mermaid
flowchart LR
    subgraph BUILD["Build-time tooling — runs offline a few times, then dormant"]
        DLt["s3_downloader.py"]:::tool
        LSt["make_label_studio_tasks.py"]:::tool
        ALt["autolabel.py (M3b gating)"]:::tool
        OVt["overlay_check.py / count_eval.py"]:::tool
    end

    subgraph ARTIFACTS["Frozen artifacts (carry forward)"]
        LBL["training labels"]:::art
        WTS["detector best.pt + embedder + FAISS index"]:::art
    end

    subgraph PROD["Shipped product — runs on every request, on EC2"]
        DET["detector"]:::prod
        EMB["embedder"]:::prod
        FA["FAISS match"]:::prod
        UI["Streamlit/Gradio UI + Docker"]:::prod
    end

    BUILD --> ARTIFACTS
    WTS --> PROD
    ARTIFACTS -.->|"provenance / reproducibility only"| PROD

    classDef tool fill:#f8d7da,stroke:#dc3545,color:#000
    classDef art fill:#e7d4f5,stroke:#8b5cf6,color:#000
    classDef prod fill:#cce5ff,stroke:#0d6efd,color:#000
```

`autolabel.py` lives in the **left box**: it produces labels, then goes quiet. It never
ships to EC2. It stays in the repo as the auditable record of *how* labels were made (a
graded deliverable), but the running app only needs the trained weights.

---

## 5. Locked-decision log (the "why", for the video)

| # | Decision | Rationale |
|---|---|---|
| D1 | YOLO single-class detection (count instances) → embedder → FAISS → **constrained** matching | Compare each crop only to the *order's* candidate ASINs, not all 5285 — makes the problem tractable |
| D2 | Staged self-training (Option A) for identity | Seed from single-ASIN bins, pseudo-label multi-ASIN bins; grows catalog coverage without full manual labels |
| D3 | Demonstrate all 3 labeling methods (manual → SAM → zero-shot) | Brief requires it; also lets us A/B raw vs gated auto-labels |
| D4 | Ship M4 as an **honest baseline** (mAP 0.255) before M3b | The whole M3b A/B is framed as "beat this baseline"; needs it in master |
| D5 | Track 0 gold set is a prerequisite | Only trustworthy mAP ruler; 18-image val split is too noisy to tune against |
| D6 | Gating logic on CPU (`autolabel.py`), GPU does inference only | Testable without a GPU; the intellectual core is auditable |
| D7 | Notebooks are the deliverable; `tools/` is the engine | Mentor brief wants narrative notebooks; engine keeps them thin |
| D8 | EC2 deployment is non-negotiable → M3b is time-boxed | Required deliverable; don't let labeling perfection eat the deploy |
| D9 | **Auto-labeling abandoned** — raw union noisy, SAM∩zero-shot starved, SAM∩M4 correlated-errors | Measured on ~200 bins; fully-automatic labeling can't box items on cluttered/occluded ABID bins. See `M3b_FINDINGS.md` |
| D10 | **Pivot to Siamese-first (2026-07-09)** — Siamese similarity model is the graded core; detection demoted to optional | Evaluator template (`Copy of Week 4 & 5.ipynb`) = Siamese + ROC/threshold/confusion (VGG16 vs ResNet50), **no detection**. Core needs no detection labels |
| D11 | Keep M4 baseline + M3b engine as an **optional counting/innovation add-on**, not critical path | Objectives reward innovation; the explored-and-set-aside branch is legitimate video content |

> **Status note (2026-07-09):** the pivot demotes the detection column of the §1 pipeline.
> New critical path: **M5 Siamese similarity** (whole-bin pairs) → verify (ROC/threshold/
> confusion) → productionize → EC2. YOLO/M4/M3b remain in-repo as an optional enhancement.

---

*Update this file as milestones land — it's the map the final video's three levels
(business / decisions / library deep-dive) narrate against.*
