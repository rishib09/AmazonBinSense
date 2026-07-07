# M3 — Labeling Cheat Sheet (one screen)

> Quick reference while annotating. Full rules + edge cases: [`03_LABELING_GUIDE.md`](03_LABELING_GUIDE.md).
> Why we label this way: [`00_PROJECT_ANCHOR.md`](00_PROJECT_ANCHOR.md) §4.2.

## The golden rule
**Box what you can SEE, never what the metadata SAYS.** Single class: `item`.

- `EXPECTED_QUANTITY` is a **sanity check, not a target.** Meta says 2, you see 1 → label **1**.
- **Never invent a box** over glare/occlusion to match the count. Over-labeling teaches
  the detector to hallucinate objects (false positives everywhere). Under-labeling only
  costs a little recall. **When unsure, box less, not more.**
- One sellable ASIN = one box. A "3-pack" is **one box**, not three.

## Decision in 5 seconds
| You see… | Draw |
|---|---|
| A clearly separable unit | 1 tight box |
| Two touching same-color items, both visible | 2 boxes |
| A pile — some units distinguishable | 1 box per distinguishable top/edge; ignore the rest |
| A sliver of a clearly-separate item behind another | 1 box on the sliver |
| Can't tell an item is even there | **skip it** |
| Nothing visible | submit **0 boxes** (valid negative) |
| Dividers, tape, barcodes, shadows, glare | **never box** |

## Same image, two jobs
- **Detection training** → label every visible instance honestly (above).
- **Gallery seed** (embedder reference crop for that ASIN) → only **clean, clearly-visible**
  crops. Heavily occluded single-ASIN bins are low value for the gallery — use them for
  detection only, or skip. Don't perfect boxes on mud.

## What happens to the count gap (downstream — not your problem while labeling)
The verifier compares **visible count** vs **expected**, in three regimes:
- `visible > expected` → **flag**: extra / wrong items (real anomaly).
- `visible == expected` → **pass**.
- `visible < expected` → **inconclusive (likely occlusion)** → log / route to HITL. **Not a failure.**

So your visible-only labels are correct even when they disagree with metadata — the
disagreement is expected and handled here, not in your annotation.
