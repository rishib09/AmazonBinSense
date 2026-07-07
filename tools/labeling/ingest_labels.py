"""
tools/labeling/ingest_labels.py
===============================
Ingest a Label Studio "YOLO with Images" export into data/labels/.

Label Studio prefixes exported filenames with an 8-char hash
(`<hash>__<bin_id>.txt`). Our pipeline needs `data/labels/<bin_id>.txt` to pair
with `data/images/<bin_id>.jpg`, so this strips the prefix, validates each bin,
refuses to ingest eval-gold bins, and copies the .txt files in.

Usage
-----
    uv run python tools/labeling/ingest_labels.py --export-dir "<...>/label_studio_output"
    uv run python tools/labeling/ingest_labels.py --export-dir "<...>" --dry-run
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

LOCAL_DATA = r"G:\My Drive\Interview Kickstart\Capstone Project\Amazon BinSense\data"
if not os.getenv("BINSENSE_DATA_DIR") and Path(LOCAL_DATA).exists():
    os.environ["BINSENSE_DATA_DIR"] = LOCAL_DATA
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.env_utils import setup_env  # noqa: E402


def bin_id_from_name(stem: str) -> str:
    """`026da21b__113235` -> `113235`; leaves already-clean stems unchanged."""
    return stem.rsplit("__", 1)[-1]


def load_eval_ids(splits_dir: Path) -> set[str]:
    p = splits_dir / "eval.csv"
    if not p.exists():
        return set()
    with open(p, newline="", encoding="utf-8") as f:
        return {r["bin_id"].strip().zfill(5) for r in csv.DictReader(f) if r.get("bin_id")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a Label Studio YOLO export into data/labels/.")
    ap.add_argument("--export-dir", type=Path, required=True,
                    help="The unzipped export folder (contains labels/, images/, classes.txt)")
    ap.add_argument("--dry-run", action="store_true", help="Report only; copy nothing")
    args = ap.parse_args()

    cfg = setup_env(verbose=False)
    src = args.export_dir / "labels"
    if not src.exists():
        print(f"No labels/ folder in {args.export_dir}")
        return

    # classes.txt sanity
    classes = (args.export_dir / "classes.txt")
    if classes.exists():
        names = [c.strip() for c in classes.read_text(encoding="utf-8").splitlines() if c.strip()]
        if names != ["item"]:
            print(f"⚠️  classes.txt = {names} (expected exactly ['item']) — check your LS config.")

    eval_ids = load_eval_ids(cfg.splits_dir)
    cfg.labels_dir.mkdir(parents=True, exist_ok=True)

    copied, empties, leaked, no_image, seen = 0, [], [], [], {}
    for lf in sorted(src.glob("*.txt")):
        bid = bin_id_from_name(lf.stem).zfill(5)
        if bid in eval_ids:
            leaked.append(bid)
            continue
        if bid in seen:
            print(f"⚠️  duplicate bin {bid}: {lf.name} and {seen[bid]}")
        seen[bid] = lf.name
        if not (cfg.images_dir / f"{bid}.jpg").exists():
            no_image.append(bid)
        if not lf.read_text(encoding="utf-8").strip():
            empties.append(bid)
        if not args.dry_run:
            shutil.copyfile(lf, cfg.labels_dir / f"{bid}.txt")
        copied += 1

    assert not leaked, f"EVAL LEAKAGE — export contains eval bins, refusing: {sorted(leaked)}"

    verb = "Would copy" if args.dry_run else "Copied"
    print(f"{verb} {copied} label file(s) -> {cfg.labels_dir}")
    print(f"  empty (0-box) labels : {len(empties)}" + (f" {empties}" if empties else ""))
    if no_image:
        print(f"  ⚠️  labels with NO matching image: {len(no_image)} -> {no_image[:10]}")
    if args.dry_run:
        print("  (dry run — nothing written)")
    else:
        print("\nNext: overlay QA —")
        print("  uv run python tools/labeling/overlay_check.py")


if __name__ == "__main__":
    main()
