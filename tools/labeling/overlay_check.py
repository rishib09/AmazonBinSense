"""
tools/labeling/overlay_check.py
===============================
Sanity-check exported YOLO labels by drawing them back onto the bin images.

The #1 labeling-export bug is coordinate misalignment (percent vs normalized,
wrong image dims, xywh vs xyxy). This renders each label file's boxes over its
image so you can eyeball correctness BEFORE training (M4), and prints a summary:
box counts, empty/negative images, coordinate-range warnings, and the
visible-vs-EXPECTED_QUANTITY delta (a healthy gap is expected — we label visible
instances; see docs/03_LABELING_GUIDE.md).

Usage
-----
    python tools/labeling/overlay_check.py            # render a 40-image sample
    python tools/labeling/overlay_check.py --all      # render everything
    python tools/labeling/overlay_check.py --n 100 --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# ── code/data split aware (mirror the notebook bootstrap) ─────────────────────
LOCAL_DATA = r"G:\My Drive\Interview Kickstart\Capstone Project\Amazon BinSense\data"
if not os.getenv("BINSENSE_DATA_DIR") and Path(LOCAL_DATA).exists():
    os.environ["BINSENSE_DATA_DIR"] = LOCAL_DATA
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.env_utils import setup_env  # noqa: E402


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return [(cls, cx, cy, w, h), ...] from a YOLO .txt (empty file → [])."""
    boxes = []
    if not path.exists():
        return boxes
    for ln in path.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cls, cx, cy, w, h))
    return boxes


def box_warnings(boxes) -> list[str]:
    """Flag coordinates that don't look like valid normalized YOLO boxes."""
    out = []
    for i, (_cls, cx, cy, w, h) in enumerate(boxes):
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            out.append(f"box {i}: coord outside [0,1] {(cx, cy, w, h)} — percent, not normalized?")
        if w <= 0 or h <= 0:
            out.append(f"box {i}: non-positive w/h {(w, h)}")
        elif cx - w / 2 < -0.01 or cx + w / 2 > 1.01 or cy - h / 2 < -0.01 or cy + h / 2 > 1.01:
            out.append(f"box {i}: extends outside image bounds")
    return out


def draw_overlay(image_path: Path, boxes, out_path: Path) -> bool:
    """Draw boxes over the image and save. Returns False if the image is unreadable."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return False
    H, W = img.shape[:2]
    for (_cls, cx, cy, w, h) in boxes:
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
    cv2.putText(img, f"{len(boxes)} boxes", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def _expected_qty(meta_dir: Path | None, bin_id: str):
    if meta_dir is None:
        return None
    p = meta_dir / f"{bin_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("EXPECTED_QUANTITY")
    except Exception:
        return None


def summarize(labels_dir: Path, images_dir: Path, meta_dir: Path | None = None) -> list[dict]:
    """Print a coverage/quality summary; return one dict per label file."""
    label_files = sorted(labels_dir.glob("*.txt"))
    rows, total_boxes, n_empty, n_warn = [], 0, 0, 0
    for lf in label_files:
        bid = lf.stem
        boxes = parse_label(lf)
        warns = box_warnings(boxes)
        n_warn += bool(warns)
        n_empty += (len(boxes) == 0)
        total_boxes += len(boxes)
        eq = _expected_qty(meta_dir, bid)
        rows.append({
            "bin_id": bid,
            "n_boxes": len(boxes),
            "expected": eq,
            "delta": (len(boxes) - eq) if eq is not None else None,
            "img_exists": (images_dir / f"{bid}.jpg").exists(),
            "warnings": warns,
        })
    n = len(label_files)
    print(f"Label files             : {n}")
    print(f"Total boxes             : {total_boxes}")
    if n:
        counts = [r["n_boxes"] for r in rows]
        print(f"Boxes/image             : min {min(counts)}, mean {total_boxes / n:.1f}, max {max(counts)}")
    print(f"Empty (negative) images : {n_empty}")
    print(f"Files with coord WARN   : {n_warn}")
    missing_img = [r["bin_id"] for r in rows if not r["img_exists"]]
    if missing_img:
        print(f"Labels with NO image    : {len(missing_img)} → {missing_img[:10]}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay YOLO labels on images for QA.")
    ap.add_argument("--labels-dir", type=Path, default=None)
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=40, help="sample size to render (default 40)")
    ap.add_argument("--all", action="store_true", help="render every labeled image")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = setup_env(verbose=False)
    labels_dir = args.labels_dir or cfg.labels_dir
    images_dir = args.images_dir or cfg.images_dir
    meta_dir = cfg.metadata_dir
    out = args.out or (cfg.base_dir / "reports" / "label_overlays")

    if not labels_dir.exists() or not any(labels_dir.glob("*.txt")):
        print(f"No YOLO labels found in {labels_dir}\n"
              f"Export from Label Studio (YOLO) into data/labels/ first "
              f"(see docs/03_LABELING_GUIDE.md).")
        return

    print(f"Labels : {labels_dir}")
    print(f"Images : {images_dir}")
    print(f"Out    : {out}\n")
    rows = summarize(labels_dir, images_dir, meta_dir)

    all_warn = [(r["bin_id"], w) for r in rows for w in r["warnings"]]
    if all_warn:
        print(f"\n⚠️  {len(all_warn)} coordinate warning(s) — first 10:")
        for bid, w in all_warn[:10]:
            print(f"   {bid}: {w}")

    labeled = [r for r in rows if r["img_exists"]]
    if not args.all:
        random.Random(args.seed).shuffle(labeled)
        labeled = labeled[:args.n]

    print(f"\nRendering {len(labeled)} overlay(s) ...")
    ok = sum(
        draw_overlay(images_dir / f"{r['bin_id']}.jpg",
                     parse_label(labels_dir / f"{r['bin_id']}.txt"),
                     out / f"{r['bin_id']}.jpg")
        for r in labeled
    )
    print(f"  wrote {ok} overlay image(s) to {out}")
    print("\nEyeball a few: boxes should hug items. If shifted/scaled, the YOLO "
          "export mapping is wrong — fix before M4.")


if __name__ == "__main__":
    main()
