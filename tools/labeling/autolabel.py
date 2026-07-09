"""
tools/labeling/autolabel.py
===========================
The M3b auto-labeling **gating engine** (pure CPU, testable).

Two label tracks are built from the raw per-method box files produced on Colab
(SAM auto-mask -> boxes, and a zero-shot open-vocabulary detector). GPU inference
lives in notebook 03; **all quality logic lives here** so it can be unit-tested
without a GPU and so the Track-1-vs-Track-2 A/B (GitHub issue #2) turns on code we
can reason about.

Track 1 — RAW UNION (control): concat SAM + zero-shot boxes, drop only obvious
    junk (specks / full-frame / slivers), dedup with NMS. Exposes the *noise
    floor*; NOT a production label set.

Track 2 — GATED (treatment): the same boxes must survive four gates before they
    are trusted (each output box carries provenance=auto):
      Gate C — geometry sanity  (drop specks, bin-sized, frame, extreme aspect)
      Gate B — cross-method agreement (keep boxes SAM and zero-shot BOTH found,
                                       IoU >= thresh)  -> kills correlated-with-
                                       nothing single-method hallucinations
      Gate A — count plausibility (reject the whole bin if #boxes is implausible
                                   vs EXPECTED_QUANTITY: 0 => fail, >> E => over-seg)
      Gate D — seed-model agreement (OPTIONAL self-training; only if seed boxes
                                     are supplied — weak until the detector improves)

A bin that fails Gate A is dropped from Track 2 (optionally routed to human
review) rather than injected as noise.

Box representation (internal): dict with normalized YOLO coords
    {"cx","cy","w","h","score","source"}  (class is always 0 = 'item')

On disk we use YOLO txt with an optional 6th score column:
    0 cx cy w h [score]

    from tools.labeling.autolabel import build_track, GateParams
    stats = build_track("track2", sam_dir, zs_dir, out_dir, meta_dir,
                        bin_ids, eval_ids, GateParams())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

# ── box I/O ───────────────────────────────────────────────────────────────────

def read_boxes(path: Path, source: str = "") -> list[dict]:
    """Read a YOLO txt (`cls cx cy w h [score]`) into a list of box dicts."""
    path = Path(path)
    out: list[dict] = []
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        p = ln.strip().split()
        if len(p) < 5:
            continue
        cx, cy, w, h = map(float, p[1:5])
        score = float(p[5]) if len(p) >= 6 else 1.0
        out.append({"cx": cx, "cy": cy, "w": w, "h": h, "score": score, "source": source})
    return out


def write_boxes(path: Path, boxes: list[dict], with_score: bool = False) -> None:
    """Write boxes to a YOLO txt. Class is always 0. Empty list => empty file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for b in boxes:
        row = f"0 {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        if with_score:
            row += f" {b.get('score', 1.0):.4f}"
        lines.append(row)
    path.write_text("\n".join(lines), encoding="utf-8")


def expected_quantity(meta_dir: Path | None, bin_id: str):
    """EXPECTED_QUANTITY from a bin's metadata JSON, or None if unavailable."""
    if meta_dir is None:
        return None
    p = Path(meta_dir) / f"{bin_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("EXPECTED_QUANTITY")
    except Exception:
        return None


# ── geometry ──────────────────────────────────────────────────────────────────

def _xyxy(b: dict) -> tuple[float, float, float, float]:
    return (b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2,
            b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2)


def iou(a: dict, b: dict) -> float:
    """Intersection-over-Union of two normalized boxes. 0.0 if disjoint/degenerate."""
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── gate parameters ───────────────────────────────────────────────────────────

@dataclass
class GateParams:
    # Gate C — geometry
    min_area: float = 0.0015     # drop specks below this fraction of the frame
    max_area: float = 0.45       # drop bin-/frame-sized boxes above this fraction
    max_aspect: float = 12.0     # drop slivers with w/h (or h/w) beyond this
    min_score: float = 0.0       # drop boxes below this confidence (0 = keep all)
    # Gate B — cross-method agreement
    agree_iou: float = 0.5       # SAM & zero-shot boxes must overlap at least this
    # Gate A — count plausibility (n = surviving box count for the bin)
    count_over_frac: float = 1.5  # reject bin if n > E * this (over-segmentation)
    count_abs_slack: int = 2      # ... plus this absolute slack on the upper bound
    min_boxes: int = 1            # reject bin if fewer than this survive
    # dedup
    nms_iou: float = 0.6         # merge boxes overlapping more than this
    # Gate D — seed-model agreement (only applied if seed boxes supplied)
    seed_iou: float = 0.5


# ── gates ─────────────────────────────────────────────────────────────────────

def gate_geometry(boxes: list[dict], p: GateParams) -> list[dict]:
    """Gate C — keep only geometrically plausible item boxes."""
    kept = []
    for b in boxes:
        w, h = b["w"], b["h"]
        if w <= 0 or h <= 0:
            continue
        area = w * h
        if area < p.min_area or area > p.max_area:
            continue
        ar = max(w / h, h / w)
        if ar > p.max_aspect:
            continue
        if b.get("score", 1.0) < p.min_score:
            continue
        # frame-hugging: spans almost the whole width AND height (the bin itself)
        if w >= 0.92 and h >= 0.92:
            continue
        kept.append(b)
    return kept


def nms(boxes: list[dict], iou_thresh: float) -> list[dict]:
    """Greedy non-max suppression; ranks by score then area (larger wins ties)."""
    order = sorted(boxes, key=lambda b: (b.get("score", 1.0), b["w"] * b["h"]), reverse=True)
    kept: list[dict] = []
    for b in order:
        if all(iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def cross_method_agreement(a: list[dict], b: list[dict], iou_thresh: float) -> list[dict]:
    """
    Gate B — keep boxes that BOTH methods found. For each box in `a` with a
    match in `b` (IoU >= thresh), emit the geometric average of the pair (the
    consensus box), with score = min of the two. Each `b` box matches at most once.
    """
    used_b: set[int] = set()
    consensus: list[dict] = []
    for ba in a:
        best_j, best_iou = -1, iou_thresh
        for j, bb in enumerate(b):
            if j in used_b:
                continue
            v = iou(ba, bb)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used_b.add(best_j)
            bb = b[best_j]
            consensus.append({
                "cx": (ba["cx"] + bb["cx"]) / 2, "cy": (ba["cy"] + bb["cy"]) / 2,
                "w": (ba["w"] + bb["w"]) / 2, "h": (ba["h"] + bb["h"]) / 2,
                "score": min(ba.get("score", 1.0), bb.get("score", 1.0)),
                "source": "consensus",
            })
    return consensus


def gate_count(n: int, expected, p: GateParams) -> tuple[bool, str]:
    """
    Gate A — is the surviving box count plausible for this bin?

    We label *visible* instances, so n <= EXPECTED_QUANTITY is normal (occlusion).
    The gate therefore rejects only the two implausible extremes: nothing found,
    or gross over-segmentation (many more boxes than items could exist).
    Returns (passed, reason).
    """
    if n < p.min_boxes:
        return False, "count_zero"
    if expected is None:
        return False, "no_expected"
    upper = expected * p.count_over_frac + p.count_abs_slack
    if n > upper:
        return False, "over_segmented"
    return True, "ok"


def gate_seed_agreement(boxes: list[dict], seed: list[dict], iou_thresh: float) -> list[dict]:
    """Gate D (optional) — keep consensus boxes the seed detector also supports."""
    return [b for b in boxes if any(iou(b, s) >= iou_thresh for s in seed)]


# ── track builders (per bin) ──────────────────────────────────────────────────

def raw_union(sam: list[dict], zs: list[dict], p: GateParams) -> list[dict]:
    """Track 1 — union of both methods, junk-filtered + deduped. The noise floor."""
    boxes = gate_geometry(sam + zs, p)
    return nms(boxes, p.nms_iou)


def gated_bin(sam: list[dict], zs: list[dict], expected, p: GateParams,
              seed: list[dict] | None = None) -> dict:
    """
    Track 2 — run one bin through all gates. Returns a provenance record:
        {passed, reason, boxes, n_sam, n_zs, n_geom_*, n_consensus, n_final, expected}
    `boxes` is the accepted label set (empty if the bin is rejected).
    """
    sam_g = gate_geometry(sam, p)                         # Gate C
    zs_g = gate_geometry(zs, p)
    consensus = cross_method_agreement(sam_g, zs_g, p.agree_iou)  # Gate B
    consensus = nms(consensus, p.nms_iou)
    if seed is not None:                                  # Gate D (optional)
        consensus = gate_seed_agreement(consensus, seed, p.seed_iou)
    passed, reason = gate_count(len(consensus), expected, p)      # Gate A
    return {
        "passed": passed, "reason": reason,
        "boxes": consensus if passed else [],
        "n_sam": len(sam), "n_zs": len(zs),
        "n_sam_geom": len(sam_g), "n_zs_geom": len(zs_g),
        "n_consensus": len(consensus), "n_final": len(consensus) if passed else 0,
        "expected": expected,
    }


# ── orchestration (over many bins) ────────────────────────────────────────────

@dataclass
class TrackStats:
    track: str
    n_bins_in: int = 0
    n_bins_out: int = 0        # bins with >=1 accepted box written
    n_boxes_out: int = 0
    n_rejected: int = 0
    reject_reasons: dict = field(default_factory=dict)
    manifest: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"track": self.track, "n_bins_in": self.n_bins_in,
                "n_bins_out": self.n_bins_out, "n_boxes_out": self.n_boxes_out,
                "n_rejected": self.n_rejected, "reject_reasons": self.reject_reasons}


def build_track(track: str, sam_dir, zs_dir, out_dir, meta_dir,
                bin_ids, eval_ids, p: GateParams,
                seed_dir=None) -> TrackStats:
    """
    Build a label track for `bin_ids`, writing YOLO txts to `out_dir` and
    returning stats + a per-bin manifest (provenance).

    track      : "track1" (raw union) or "track2" (gated)
    sam_dir    : dir of per-bin SAM boxes (YOLO txt, optional score col)
    zs_dir     : dir of per-bin zero-shot boxes
    seed_dir   : optional dir of seed-detector boxes (enables Gate D on track2)
    eval_ids   : bins to NEVER label (the frozen eval-gold set) — asserted out
    """
    sam_dir, zs_dir, out_dir = Path(sam_dir), Path(zs_dir), Path(out_dir)
    seed_dir = Path(seed_dir) if seed_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_ids = set(eval_ids)
    st = TrackStats(track=track)

    for bid in bin_ids:
        if bid in eval_ids:            # the one wall we never cross
            continue
        st.n_bins_in += 1
        sam = read_boxes(sam_dir / f"{bid}.txt", "sam")
        zs = read_boxes(zs_dir / f"{bid}.txt", "zeroshot")

        if track == "track1":
            boxes = raw_union(sam, zs, p)
            rec = {"bin_id": bid, "passed": bool(boxes), "reason": "raw_union",
                   "n_sam": len(sam), "n_zs": len(zs), "n_final": len(boxes),
                   "expected": expected_quantity(meta_dir, bid), "provenance": "auto"}
        elif track == "track2":
            seed = read_boxes(seed_dir / f"{bid}.txt", "seed") if seed_dir else None
            r = gated_bin(sam, zs, expected_quantity(meta_dir, bid), p, seed)
            boxes = r["boxes"]
            rec = {"bin_id": bid, **{k: r[k] for k in
                   ("passed", "reason", "n_sam", "n_zs", "n_sam_geom",
                    "n_zs_geom", "n_consensus", "n_final", "expected")},
                   "provenance": "auto"}
        else:
            raise ValueError(f"unknown track: {track!r} (use 'track1' or 'track2')")

        st.manifest.append(rec)
        if boxes:
            write_boxes(out_dir / f"{bid}.txt", boxes)
            st.n_bins_out += 1
            st.n_boxes_out += len(boxes)
        else:
            st.n_rejected += 1
            reason = rec["reason"]
            st.reject_reasons[reason] = st.reject_reasons.get(reason, 0) + 1

    return st
