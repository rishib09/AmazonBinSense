"""
tools/download/s3_downloader.py
================================
Standalone, reusable downloader for the Amazon Bin Image Dataset.

Public S3 bucket  : aft-vbi-pds  (anonymous — no AWS credentials needed)
Metadata URL shape: s3://aft-vbi-pds/metadata/{id_5digits}.json
Image URL shape   : s3://aft-vbi-pds/bin-images/{id_5digits}.jpg

Features
--------
- Resume support   : skips files that already exist and are non-empty
- Retry with backoff: up to MAX_RETRIES attempts per file
- Parallel I/O     : ThreadPoolExecutor (configurable worker count)
- Progress bar     : tqdm (works in Colab and VS Code notebooks)
- Verification     : reports exactly which IDs are missing from disk

Intentionally has NO imports from model training code or output artifacts.
The only project-level dependency allowed here is the path config (passed in,
never imported directly from utils).

Usage
-----
    from tools.download import BinS3Downloader

    dl = BinS3Downloader(meta_dir=cfg.metadata_dir, images_dir=cfg.images_dir)
    ids = BinS3Downloader.load_ids(cfg.base_dir / "analysis" / "unique_image_names.csv")

    meta_stats  = dl.download_metadata(ids)
    image_stats = dl.download_images(ids)
    verify      = dl.verify(ids)
"""

from __future__ import annotations

import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm.auto import tqdm

# ── S3 constants ──────────────────────────────────────────────────────────────
BUCKET       = "aft-vbi-pds"
META_PREFIX  = "metadata"
IMAGE_PREFIX = "bin-images"

# ── Retry policy ──────────────────────────────────────────────────────────────
MAX_RETRIES    = 3
RETRY_BACKOFF  = 2.0   # seconds; multiplied by attempt number (1×, 2×, 3×)

log = logging.getLogger(__name__)


class BinS3Downloader:
    """
    Downloads metadata JSONs and bin images from the Amazon Bin Image Dataset.

    Parameters
    ----------
    meta_dir   : Path where metadata/*.json files will be saved.
    images_dir : Path where bin-images/*.jpg files will be saved.
    workers    : Thread-pool size.  32 works well for metadata (small files).
                 Lower to 16 for images to avoid memory pressure on Colab.
    """

    def __init__(
        self,
        meta_dir: Path,
        images_dir: Path,
        workers: int = 32,
    ) -> None:
        self.meta_dir   = Path(meta_dir)
        self.images_dir = Path(images_dir)
        self.workers    = workers
        self._client    = None

        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # ── boto3 client (lazy, anonymous) ────────────────────────────────────────

    @property
    def client(self):
        """Lazy-init anonymous boto3 S3 client (no credentials needed)."""
        if self._client is None:
            self._client = boto3.client(
                "s3",
                region_name="us-east-1",
                config=Config(signature_version=UNSIGNED),
            )
        return self._client

    # ── ID helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def load_ids(csv_path: Path) -> list[str]:
        """
        Read bin IDs from unique_image_names.csv.

        IDs are returned as strings with natural zero-padding preserved.
        Both short IDs ('00964' → kept as-is) and long IDs ('113526' → kept)
        are handled correctly; zfill(5) is a safe no-op for ≥5-digit IDs.

        Returns
        -------
        List of bin ID strings in CSV order, e.g. ['113526', '00964', ...]
        """
        ids: list[str] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row["image_metadata_filename"].strip()
                ids.append(raw.zfill(5))   # normalise; noop for len≥5
        return ids

    # ── S3 key + local destination ────────────────────────────────────────────

    @staticmethod
    def _meta_key(bin_id: str) -> str:
        return f"{META_PREFIX}/{bin_id}.json"

    @staticmethod
    def _image_key(bin_id: str) -> str:
        return f"{IMAGE_PREFIX}/{bin_id}.jpg"

    def _meta_dest(self, bin_id: str) -> Path:
        return self.meta_dir / f"{bin_id}.json"

    def _image_dest(self, bin_id: str) -> Path:
        return self.images_dir / f"{bin_id}.jpg"

    # ── Single-file download (with resume + retry) ────────────────────────────

    def _download_one(
        self,
        s3_key: str,
        dest: Path,
        force: bool = False,
    ) -> dict:
        """
        Download one S3 object to *dest*.

        Resume logic: if the file exists and is non-empty, skip it unless
        *force=True*.  A zero-byte file is treated as a failed previous attempt
        and is overwritten.

        Returns
        -------
        dict with keys:
            ok      – True if the file is now present on disk
            skipped – True if file was already there (resume hit)
            error   – error string on failure, else None
        """
        if not force and dest.exists() and dest.stat().st_size > 0:
            return {"ok": True, "skipped": True, "error": None}

        last_err: str | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.client.download_file(BUCKET, s3_key, str(dest))
                return {"ok": True, "skipped": False, "error": None}
            except Exception as exc:
                last_err = str(exc)
                log.debug("Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, s3_key, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)

        # All retries exhausted — remove any zero-byte file left behind
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)

        log.warning("FAILED after %d attempts: %s — %s", MAX_RETRIES, s3_key, last_err)
        return {"ok": False, "skipped": False, "error": last_err}

    # ── Generic parallel batch download ───────────────────────────────────────

    def _batch_download(
        self,
        ids: list[str],
        key_fn: Callable[[str], str],
        dest_fn: Callable[[str], Path],
        label: str,
        force: bool = False,
    ) -> dict:
        """
        Fan out *_download_one* across *ids* using a thread pool.

        Returns
        -------
        dict:
            total      – number of IDs submitted
            downloaded – newly downloaded
            skipped    – already-present (resume)
            failed     – could not download after all retries
            errors     – list of (bin_id, error_string) for failures
        """
        stats: dict = {
            "total":      len(ids),
            "downloaded": 0,
            "skipped":    0,
            "failed":     0,
            "errors":     [],
        }

        futures = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for bid in ids:
                fut = pool.submit(self._download_one, key_fn(bid), dest_fn(bid), force)
                futures[fut] = bid

            with tqdm(total=len(ids), desc=label, unit="file") as pbar:
                for fut in as_completed(futures):
                    bid    = futures[fut]
                    result = fut.result()

                    if result["ok"] and result["skipped"]:
                        stats["skipped"] += 1
                    elif result["ok"]:
                        stats["downloaded"] += 1
                    else:
                        stats["failed"] += 1
                        stats["errors"].append((bid, result["error"]))

                    pbar.update(1)
                    pbar.set_postfix(
                        dl=stats["downloaded"],
                        skip=stats["skipped"],
                        fail=stats["failed"],
                        refresh=False,
                    )

        return stats

    # ── Public download methods ───────────────────────────────────────────────

    def download_metadata(
        self,
        ids: list[str],
        force: bool = False,
    ) -> dict:
        """
        Download metadata/*.json for every bin ID in *ids*.

        Parameters
        ----------
        ids   : list of bin ID strings from load_ids()
        force : if True, re-download even if file already exists

        Returns
        -------
        Stats dict (total / downloaded / skipped / failed / errors)
        """
        return self._batch_download(
            ids,
            key_fn=self._meta_key,
            dest_fn=self._meta_dest,
            label="Metadata JSONs",
            force=force,
        )

    def download_images(
        self,
        ids: list[str],
        force: bool = False,
        workers: int | None = None,
    ) -> dict:
        """
        Download bin-images/*.jpg for every bin ID in *ids*.

        Parameters
        ----------
        ids     : list of bin ID strings from load_ids()
        force   : if True, re-download even if file already exists
        workers : override thread-pool size just for images (images are larger
                  than metadata; 16 is a safe default on Colab T4)

        Returns
        -------
        Stats dict (total / downloaded / skipped / failed / errors)
        """
        saved_workers = self.workers
        if workers is not None:
            self.workers = workers

        try:
            return self._batch_download(
                ids,
                key_fn=self._image_key,
                dest_fn=self._image_dest,
                label="Bin images  ",
                force=force,
            )
        finally:
            self.workers = saved_workers

    # ── Verification ──────────────────────────────────────────────────────────

    def verify(self, ids: list[str]) -> dict:
        """
        Check which files are present on disk for every ID in *ids*.

        Returns
        -------
        dict:
            total           – total IDs checked
            meta_present    – metadata files found and non-empty
            meta_missing    – count of missing / zero-byte metadata files
            meta_missing_ids– list of those IDs
            images_present  – image files found and non-empty
            images_missing  – count of missing / zero-byte image files
            images_missing_ids – list of those IDs
        """
        missing_meta:   list[str] = []
        missing_images: list[str] = []

        for bid in ids:
            mp = self._meta_dest(bid)
            ip = self._image_dest(bid)
            if not mp.exists() or mp.stat().st_size == 0:
                missing_meta.append(bid)
            if not ip.exists() or ip.stat().st_size == 0:
                missing_images.append(bid)

        total = len(ids)
        return {
            "total":             total,
            "meta_present":      total - len(missing_meta),
            "meta_missing":      len(missing_meta),
            "meta_missing_ids":  missing_meta,
            "images_present":    total - len(missing_images),
            "images_missing":    len(missing_images),
            "images_missing_ids": missing_images,
        }

    # ── Pretty printer ────────────────────────────────────────────────────────

    @staticmethod
    def print_stats(stats: dict, label: str = "") -> None:
        """Print a one-line download summary."""
        prefix = f"[{label}]  " if label else ""
        print(
            f"{prefix}"
            f"total={stats['total']}  "
            f"downloaded={stats['downloaded']}  "
            f"skipped(resume)={stats['skipped']}  "
            f"failed={stats['failed']}"
        )
        if stats["errors"]:
            print(f"  ↳ First 5 failures:")
            for bid, err in stats["errors"][:5]:
                print(f"      {bid}: {err}")

    @staticmethod
    def print_verify(v: dict) -> None:
        """Print a verification summary."""
        print(f"Verification  ({v['total']} IDs checked)")
        print(f"  Metadata : {v['meta_present']:>5} present  |  {v['meta_missing']:>5} missing")
        print(f"  Images   : {v['images_present']:>5} present  |  {v['images_missing']:>5} missing")
        if v["meta_missing_ids"]:
            print(f"  Missing meta (first 10): {v['meta_missing_ids'][:10]}")
        if v["images_missing_ids"]:
            print(f"  Missing imgs (first 10): {v['images_missing_ids'][:10]}")
