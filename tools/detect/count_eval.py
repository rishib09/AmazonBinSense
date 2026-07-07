"""
tools/detect/count_eval.py
==========================
Count-accuracy evaluation for the M4 detector (the Tier-1 "total-count" metric).

Detection mAP (via Ultralytics `model.val()`) needs boxed labels, so it runs on the
held-out **val split of the labeled bins**. But the **count** metric only needs a
number per bin, so it runs on the unlabeled **eval-gold** bins by comparing
`#detections` to `EXPECTED_QUANTITY` from metadata — no boxes required.

    from tools.detect.count_eval import count_predictions, summarize_counts
    rows = count_predictions(model, eval_ids, cfg.images_dir, cfg.metadata_dir, conf=0.25)
    print(summarize_counts(rows))
"""
from __future__ import annotations

import json
from pathlib import Path


def expected_quantity(meta_dir: Path, bin_id: str):
    p = Path(meta_dir) / f"{bin_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("EXPECTED_QUANTITY")
    except Exception:
        return None


def count_predictions(model, bin_ids, images_dir, meta_dir,
                      conf: float = 0.25, imgsz: int = 640) -> list[dict]:
    """
    Run the detector on each bin and record #detections vs EXPECTED_QUANTITY.
    `model` is an Ultralytics YOLO. Returns one dict per bin.
    """
    images_dir, meta_dir = Path(images_dir), Path(meta_dir)
    rows = []
    for bid in bin_ids:
        img = images_dir / f"{bid}.jpg"
        if not img.exists():
            continue
        res = model.predict(str(img), conf=conf, imgsz=imgsz, verbose=False)[0]
        n_det = int(len(res.boxes))
        eq = expected_quantity(meta_dir, bid)
        rows.append({
            "bin_id": bid,
            "n_det": n_det,
            "expected": eq,
            "diff": (n_det - eq) if eq is not None else None,
            "within1": (eq is not None and abs(n_det - eq) <= 1),
            "exact": (eq is not None and n_det == eq),
        })
    return rows


def summarize_counts(rows: list[dict]) -> dict:
    """Tier-1 count metrics over rows that have an EXPECTED_QUANTITY."""
    import numpy as np

    have = [r for r in rows if r["expected"] is not None]
    if not have:
        return {"n_bins": 0}
    n = len(have)
    diffs = np.array([r["n_det"] - r["expected"] for r in have], dtype=float)
    return {
        "n_bins": n,
        "within1_pct": round(100 * sum(r["within1"] for r in have) / n, 1),
        "exact_pct": round(100 * sum(r["exact"] for r in have) / n, 1),
        "rmse": round(float(np.sqrt((diffs ** 2).mean())), 2),
        "mae": round(float(np.abs(diffs).mean()), 2),
        "mean_signed_diff": round(float(diffs.mean()), 2),  # <0 => undercount (occlusion)
    }
