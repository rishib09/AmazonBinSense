"""
tools/eda/image_metrics.py
==========================
Per-image quality + structure metrics for the BinSense EDA (M2 completeness).

Covers the metrics the evaluator Week-2&3 template scaffolds beyond the basic
blur/brightness/contrast we already had: **SNR, entropy, edge density, and
contour analysis** (Canny/Sobel). Pure OpenCV + NumPy so it runs anywhere.

Used by notebooks/02b_eda_image_analysis.ipynb (narrative) — the notebook imports
these functions rather than re-implementing them.

    from tools.eda.image_metrics import analyze_image, analyze_dir
    row = analyze_image(path)                 # one image -> dict of metrics
    df  = analyze_dir(images_dir, sample=200) # many images -> list[dict]
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ── Single-metric functions (all take an 8-bit grayscale array) ───────────────

def blur_laplacian(gray: np.ndarray) -> float:
    """Focus measure: variance of the Laplacian. Low = blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def contrast(gray: np.ndarray) -> float:
    return float(gray.std())


def snr(gray: np.ndarray) -> float:
    """
    Signal-to-noise ratio (dB), a simple global estimate: 20*log10(mean/std).
    Higher = cleaner. std==0 (flat image) -> 0 dB.
    """
    mu, sigma = float(gray.mean()), float(gray.std())
    if sigma <= 1e-6:
        return 0.0
    return float(20.0 * np.log10(mu / sigma)) if mu > 0 else 0.0


def entropy(gray: np.ndarray) -> float:
    """Shannon entropy (bits) of the 256-bin intensity histogram. Higher = more detail."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum() if hist.sum() > 0 else hist
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_density(gray: np.ndarray, lo: int = 100, hi: int = 200) -> float:
    """Fraction of pixels that are Canny edges. Proxy for how much structure/clutter."""
    edges = cv2.Canny(gray, lo, hi)
    return float((edges > 0).mean())


def sobel_energy(gray: np.ndarray) -> float:
    """Mean gradient magnitude (Sobel). Complements edge_density with a continuous value."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.hypot(gx, gy).mean())


def contour_stats(gray: np.ndarray, lo: int = 100, hi: int = 200) -> tuple[int, float]:
    """(#contours, mean contour area) from Canny edges — a rough 'how many blobs' signal."""
    edges = cv2.Canny(gray, lo, hi)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0.0
    areas = [cv2.contourArea(c) for c in contours]
    return len(contours), float(np.mean(areas))


# ── Whole-image + batch ───────────────────────────────────────────────────────

def analyze_image(path: Path | str) -> dict:
    """Compute all metrics for one image. `readable=False` if it can't be loaded."""
    path = Path(path)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"bin_id": path.stem, "readable": False}
    h, w = gray.shape
    n_contours, mean_area = contour_stats(gray)
    return {
        "bin_id": path.stem, "readable": True, "width": w, "height": h,
        "brightness": brightness(gray), "contrast": contrast(gray),
        "blur_var": blur_laplacian(gray), "snr_db": snr(gray),
        "entropy": entropy(gray), "edge_density": edge_density(gray),
        "sobel_energy": sobel_energy(gray),
        "contour_count": n_contours, "mean_contour_area": mean_area,
    }


def analyze_dir(images_dir: Path | str, sample: int | None = None,
                seed: int = 42) -> list[dict]:
    """Analyze every *.jpg in a directory (or a random `sample` of them)."""
    import random
    files = sorted(Path(images_dir).glob("*.jpg"))
    if sample and sample < len(files):
        files = random.Random(seed).sample(files, sample)
    return [analyze_image(p) for p in files]


def flag_poor_quality(df, blur_pct: float = 0.05):
    """
    Multi-criterion poor-quality flag (mirrors the template's percentile approach):
    very blurry OR too dark/bright OR low contrast OR low entropy OR poor SNR.
    `df` is a pandas DataFrame from analyze_dir; returns a boolean Series.
    """
    ok = df["readable"] == True  # noqa: E712
    blur_thr = df.loc[ok, "blur_var"].quantile(blur_pct)
    ent_thr = df.loc[ok, "entropy"].quantile(blur_pct)
    snr_thr = df.loc[ok, "snr_db"].quantile(blur_pct)
    return (
        (~ok)
        | (df["blur_var"] < blur_thr)
        | (df["brightness"] < 30) | (df["brightness"] > 225)
        | (df["contrast"] < 20)
        | (df["entropy"] < ent_thr)
        | (df["snr_db"] < snr_thr)
    )
