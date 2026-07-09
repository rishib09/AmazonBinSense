"""
tools/similarity/pairs.py
=========================
Siamese pair generation for the M5 similarity model (the graded core — verify
whether two bin images contain the **same item**).

The evaluator template (docs/Copy of Week 4 & 5.ipynb) trains a Siamese network on
**positive pairs (same item)** vs **negative pairs (different item)** and scores it
with ROC / threshold / confusion. This module builds that pair index from the
**single-ASIN bins** (clean identity: one bin = one product), pure-pandas and
GPU-free so it is testable locally. Image loading + augmentation live in the
notebook's tf.data pipeline.

Data reality that shapes the design (measured on the seed split):
- 346 single-ASIN seed bins, but only **28 ASINs have >=2 bins** (57 bins).
- So most positives are **augmented** (same image, two random augmentations); the
  57 cross-bin same-ASIN pairs add real intra-class variation. Negatives (different
  ASIN) are abundant.

Train/val split is **by ASIN** (identity-disjoint) so val measures generalization,
not memorized pairs.

    from tools.similarity.pairs import build_pairs
    train_df, val_df, bins_df = build_pairs(cfg.splits_dir, split='seed')
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_identity_bins(splits_dir, split: str | None = "seed") -> pd.DataFrame:
    """Single-ASIN bins as (bin_id, asin). `split=None` uses all single bins."""
    df = pd.read_csv(Path(splits_dir) / "bins_master.csv", dtype={"bin_id": str})
    df = df[df["bucket"] == "single"]
    if split:
        df = df[df["split"] == split]
    out = df[["bin_id", "asin_list"]].rename(columns={"asin_list": "asin"}).copy()
    out["bin_id"] = out["bin_id"].str.zfill(5)
    return out.reset_index(drop=True)


def split_by_asin(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Assign each ASIN (not each bin) to train/val → identity-disjoint folds."""
    asins = sorted(df["asin"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(asins)
    n_val = max(1, int(val_frac * len(asins)))
    val_asins = set(asins[:n_val])
    df = df.copy()
    df["fold"] = np.where(df["asin"].isin(val_asins), "val", "train")
    return df


def make_pairs(df: pd.DataFrame, neg_per_pos: int = 1, seed: int = 42) -> pd.DataFrame:
    """
    One positive per bin (same-ASIN partner if one exists → kind='cross', else the
    bin itself → kind='aug' for an augmented positive) plus `neg_per_pos` different-
    ASIN negatives. Returns columns: bin_a, bin_b, label(1/0), kind.
    """
    rng = np.random.RandomState(seed)
    by_asin = df.groupby("asin")["bin_id"].apply(list).to_dict()
    bins = df["bin_id"].tolist()
    bin2asin = dict(zip(df["bin_id"], df["asin"]))
    rows = []
    for bid in bins:
        asin = bin2asin[bid]
        mates = [b for b in by_asin[asin] if b != bid]
        if mates:
            rows.append({"bin_a": bid, "bin_b": mates[rng.randint(len(mates))],
                         "label": 1, "kind": "cross"})
        else:
            rows.append({"bin_a": bid, "bin_b": bid, "label": 1, "kind": "aug"})
        for _ in range(neg_per_pos):
            nb = bins[rng.randint(len(bins))]
            while bin2asin[nb] == asin:
                nb = bins[rng.randint(len(bins))]
            rows.append({"bin_a": bid, "bin_b": nb, "label": 0, "kind": "neg"})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_pairs(splits_dir, split: str | None = "seed", val_frac: float = 0.2,
                neg_per_pos: int = 1, seed: int = 42):
    """Return (train_pairs, val_pairs, bins_df) with an ASIN-disjoint train/val split."""
    bins_df = split_by_asin(load_identity_bins(splits_dir, split), val_frac, seed)
    train = make_pairs(bins_df[bins_df["fold"] == "train"], neg_per_pos, seed)
    val = make_pairs(bins_df[bins_df["fold"] == "val"], neg_per_pos, seed + 1)
    return train, val, bins_df


def summarize(train: pd.DataFrame, val: pd.DataFrame, bins_df: pd.DataFrame) -> dict:
    """Sanity numbers + an assert that no ASIN leaks across the train/val folds."""
    tr_asins = set(bins_df[bins_df["fold"] == "train"]["asin"])
    va_asins = set(bins_df[bins_df["fold"] == "val"]["asin"])
    leak = tr_asins & va_asins
    assert not leak, f"ASIN leakage across folds: {sorted(leak)[:5]}"
    def _kinds(d):
        return {k: int(v) for k, v in d["kind"].value_counts().items()}
    return {
        "train_pairs": len(train), "val_pairs": len(val),
        "train_asins": len(tr_asins), "val_asins": len(va_asins),
        "train_kinds": _kinds(train), "val_kinds": _kinds(val),
        "train_pos_frac": round(float(train["label"].mean()), 3),
        "val_pos_frac": round(float(val["label"].mean()), 3),
    }
