"""
tools/labeling/make_label_studio_tasks.py
==========================================
Generate a Label Studio JSON import file for a stratified sample of bin images.

Each task carries the image URL (public S3 HTTPS or local file path) plus
metadata fields visible in the labeling UI: bin_id, asin_count,
expected_quantity, and asin_list.  The annotator draws one bounding box per
visible item — single class "item" — which is later exported as YOLO format.

Sampling strategy
-----------------
  --single N   single-ASIN bins (easiest: box count == expected_quantity)
  --multi  N   2–3 ASIN bins    (realistic occlusion)
  --hard   N   4+ ASIN bins     (dense / challenging)

Default: 50 single + 50 multi + 20 hard = 120 images.

Output
------
  data/splits/label_studio_sample.csv   IDs selected (for audit / reproducibility)
  data/label_studio_tasks.json          import this file into Label Studio

Usage
-----
  # S3 URLs (works anywhere, no local images needed):
  python tools/labeling/make_label_studio_tasks.py

  # Local file paths (faster in Label Studio if images are already downloaded):
  python tools/labeling/make_label_studio_tasks.py --local

  # Custom sample sizes:
  python tools/labeling/make_label_studio_tasks.py --single 30 --multi 30 --hard 10

  # Reproducible run:
  python tools/labeling/make_label_studio_tasks.py --seed 99
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

# ── Path config (code/data split aware) ───────────────────────────────────────
# Data (images/metadata) may live apart from code on Google Drive. Resolve the
# data root via utils.env_utils / BINSENSE_DATA_DIR (mirrors the notebook Cell 1
# bootstrap). Reproducibility artifacts (split CSVs, the sample manifest) live
# with the CODE in the git repo so they stay version-controlled.
LOCAL_DATA = r"G:\My Drive\Interview Kickstart\Capstone Project\Amazon BinSense\data"
if not os.getenv("BINSENSE_DATA_DIR") and Path(LOCAL_DATA).exists():
    os.environ["BINSENSE_DATA_DIR"] = LOCAL_DATA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.env_utils import setup_env

cfg = setup_env(verbose=False)

BASE_DIR      = cfg.base_dir                     # code root (git repo)
CSV_PATH      = BASE_DIR / "analysis" / "unique_image_names.csv"
META_DIR      = cfg.metadata_dir                 # data root (Drive on this machine)
IMAGES_DIR    = cfg.images_dir
CODE_SPLITS   = BASE_DIR / "data" / "splits"     # git-tracked splits
EVAL_CSV      = CODE_SPLITS / "eval.csv"         # held-out gold — NEVER label/train on it
OUTPUT_JSON   = BASE_DIR / "data" / "label_studio_tasks.json"
OUTPUT_CSV    = CODE_SPLITS / "label_studio_sample.csv"

# ── S3 public URL template ────────────────────────────────────────────────────
S3_IMAGE_URL  = "https://aft-vbi-pds.s3.amazonaws.com/bin-images/{bin_id}.jpg"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ids(csv_path: Path) -> list[str]:
    ids: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row["image_metadata_filename"].strip().zfill(5))
    return ids


def load_eval_ids(path: Path) -> set[str]:
    """Bin IDs held out for eval — must NEVER enter the labeling/training pool."""
    if not path.exists():
        print(
            f"  WARNING: {path} not found; NOT excluding any eval bins "
            f"(run notebook 02 to produce it). Risk of train/eval leakage.",
            file=sys.stderr,
        )
        return set()
    ids: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bid = (row.get("bin_id") or "").strip()
            if bid:
                ids.add(bid.zfill(5))
    return ids


def load_meta(bin_id: str) -> dict | None:
    """Return parsed metadata for bin_id, or None if file missing."""
    path = META_DIR / f"{bin_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def asin_count(meta: dict | None) -> int:
    if meta is None:
        return -1
    return len(meta.get("BIN_FCSKU_DATA") or {})


def expected_quantity(meta: dict | None) -> int | None:
    if meta is None:
        return None
    return meta.get("EXPECTED_QUANTITY")


def asin_list(meta: dict | None) -> list[str]:
    if meta is None:
        return []
    return list((meta.get("BIN_FCSKU_DATA") or {}).keys())


def image_url(bin_id: str, local: bool) -> str:
    if local:
        path = IMAGES_DIR / f"{bin_id}.jpg"
        if not path.exists():
            # fall back to S3 if local file missing
            return S3_IMAGE_URL.format(bin_id=bin_id)
        # Label Studio on localhost accepts absolute paths via /data/local-files/
        # when storage is configured; for simplicity use file:// URI
        return path.as_uri()
    return S3_IMAGE_URL.format(bin_id=bin_id)


# ── Sampling ──────────────────────────────────────────────────────────────────

def stratified_sample(
    ids: list[str],
    n_single: int,
    n_multi: int,
    n_hard: int,
    seed: int,
    exclude: set[str] = frozenset(),
) -> list[str]:
    """
    Partition IDs by ASIN count bucket and draw without replacement.
    IDs whose metadata is missing are put in a fallback pool.
    Bins in `exclude` (the eval holdout) are dropped before sampling.
    """
    rng = random.Random(seed)

    single_pool, multi_pool, hard_pool, unknown_pool = [], [], [], []
    n_excluded = 0
    for bid in ids:
        if bid in exclude:
            n_excluded += 1
            continue
        meta = load_meta(bid)
        n = asin_count(meta)
        if n == 1:
            single_pool.append(bid)
        elif n in (2, 3):
            multi_pool.append(bid)
        elif n >= 4:
            hard_pool.append(bid)
        else:
            unknown_pool.append(bid)

    def draw(pool: list[str], n: int, label: str) -> list[str]:
        if len(pool) < n:
            print(
                f"  WARNING: requested {n} {label} bins but only {len(pool)} available;"
                f" using all {len(pool)}.",
                file=sys.stderr,
            )
            n = len(pool)
        return rng.sample(pool, n)

    sampled = (
        draw(single_pool,  n_single, "single-ASIN")
        + draw(multi_pool,  n_multi,  "2–3 ASIN")
        + draw(hard_pool,   n_hard,   "4+ ASIN")
    )

    if n_excluded:
        print(f"  Excluded {n_excluded} eval/held-out bin(s) from the labeling pool.")
    print(
        f"  Pools available — single: {len(single_pool)}, "
        f"multi: {len(multi_pool)}, hard: {len(hard_pool)}, "
        f"unknown (no metadata): {len(unknown_pool)}"
    )
    return sampled


# ── Task builder ──────────────────────────────────────────────────────────────

def build_task(bin_id: str, local: bool) -> dict:
    meta = load_meta(bin_id)
    n    = asin_count(meta)
    qty  = expected_quantity(meta)
    asins = asin_list(meta)

    return {
        "data": {
            "image":             image_url(bin_id, local),
            "bin_id":            bin_id,
            "asin_count":        n if n >= 0 else "unknown",
            "expected_quantity": qty if qty is not None else "unknown",
            "asin_list":         asins,
            # hint shown in labeling UI: how many boxes the annotator should draw
            "hint":              (
                f"Draw a bounding box around every visible item. "
                f"Expected {qty} item(s) total."
                if qty is not None
                else "Draw a bounding box around every visible item."
            ),
        }
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Label Studio task JSON for bin image labeling."
    )
    parser.add_argument("--single", type=int, default=50,
                        help="Number of single-ASIN bins (default 50)")
    parser.add_argument("--multi",  type=int, default=50,
                        help="Number of 2–3 ASIN bins (default 50)")
    parser.add_argument("--hard",   type=int, default=20,
                        help="Number of 4+ ASIN bins (default 20)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for reproducibility (default 42)")
    parser.add_argument("--local",  action="store_true",
                        help="Use local file:// URIs instead of S3 HTTPS URLs")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON,
                        help=f"Output JSON path (default: {OUTPUT_JSON})")
    args = parser.parse_args()

    print(f"Loading IDs from {CSV_PATH} ...")
    ids = load_ids(CSV_PATH)
    print(f"  {len(ids)} total IDs in subset")

    exclude = load_eval_ids(EVAL_CSV)
    print(f"  {len(exclude)} eval bins loaded for exclusion (from {EVAL_CSV.name})")

    print("Sampling ...")
    sampled = stratified_sample(
        ids, args.single, args.multi, args.hard, args.seed, exclude=exclude
    )
    print(f"  Selected {len(sampled)} bins for labeling")

    # Safety net: assert no eval bin slipped through
    leaked = sorted(set(sampled) & exclude)
    assert not leaked, f"Eval leakage — these eval bins are in the sample: {leaked}"

    print("Building Label Studio tasks ...")
    tasks = [build_task(bid, args.local) for bid in sampled]

    # ── Write outputs ──────────────────────────────────────────────────────────
    CODE_SPLITS.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)
    print(f"  Written: {args.output}  ({len(tasks)} tasks)")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_id", "asin_count", "expected_quantity", "asin_list"])
        for t in tasks:
            d = t["data"]
            writer.writerow([
                d["bin_id"],
                d["asin_count"],
                d["expected_quantity"],
                "|".join(d["asin_list"]),
            ])
    print(f"  Written: {OUTPUT_CSV}")

    print("\nNext steps:")
    print("  1. Start Label Studio:  label-studio start")
    print("  2. Create project -> use the RectangleLabels XML config (see README)")
    print(f"  3. Import tasks:        {args.output}")
    print("  4. Label all items with bounding boxes (single class: 'item')")
    print("  5. Export -> YOLO format -> data/labels/")


if __name__ == "__main__":
    main()
