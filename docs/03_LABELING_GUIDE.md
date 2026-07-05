# M3 — Labeling Guide (single-class object detection)

> **Goal:** produce YOLO-format boxes (`data/labels/`) for the detector (M4).
> **Scope of this pass:** the 120-bin manual seed in `data/label_studio_tasks.json`
> (50 single + 50 multi + 20 hard; **eval-gold bins already excluded** — no leakage).
> Later passes (SAM-assisted, zero-shot) reuse the exact same convention below.
>
> Read [`00_PROJECT_ANCHOR.md`](00_PROJECT_ANCHOR.md) §4.2 first — it's why we count
> from pixels and treat metadata as a cross-check, not the answer.

---

## 1. The one rule that governs everything

**Draw one tight box around every _visible_ physical unit. Single class: `item`.**

- **Visible** — we label what the camera can see. The tape and piling hide items;
  we do **not** guess hidden ones. Our count metric measures *visible-count*
  accuracy, and disagreement with `EXPECTED_QUANTITY` due to occlusion is **expected
  and acceptable** (documented in the anchor).
- **Physical unit** — one sellable item = one box. A "3-pack" is sold as **one**
  ASIN, so it is **one box**, not three (the order's *quantity* counts packs, not
  the things inside a pack).
- **Single class** — everything is `item`. Identity (which ASIN) is solved later by
  the embedder; the detector only localizes and counts.

The metadata shown in the UI (`expected_quantity`, `asin_count`, `asin_list`) is a
**sanity check**, not a target. If you can only see 4 items in a bin whose metadata
says 6, label **4** — do not invent 2 boxes to match.

---

## 2. Edge-case playbook

| Situation | What to do |
|---|---|
| **Occlusion** — item mostly hidden behind another | If you're confident it's a *separate* item, draw one box over the visible sliver. If you genuinely can't tell it's there, skip it. |
| **Dense identical pile** (e.g., 14 scissors stacked) | Box each unit you can individually distinguish (a distinct top/edge/handle). Don't force the count to match metadata; box what's separable. |
| **Multi-pack / bundle** (blister of 2, "3-pack") | **One box** — it's one sellable unit (one ASIN). |
| **Ambiguous boundary** between two touching same-color items | Split into two boxes only if you can see two distinct items; otherwise one. Be consistent across the batch. |
| **Partially cut off at frame edge** | Box it if it's clearly an item; include only the visible part. |
| **Transparent/bagged item** | Box the bag/package outline as one item. |
| **NOT items — never box these** | Bin dividers, the tape, barcodes/labels on the bin, shadows, glare/reflections, the empty bin wall. |
| **Empty-looking bin** (nothing clearly visible) | Submit with **zero** boxes — a valid negative example. |

**Tightness:** boxes should hug the visible extent of the item — not loose, not
clipping. Loose boxes produce dirty crops that later pollute the embedder gallery.

---

## 3. Setup (Windows + uv)

Label Studio runs via **uvx** (nothing to install into the project).

> **Why not the S3 URLs?** The public `aft-vbi-pds` bucket sends **no CORS header**,
> and Label Studio renders images on a *cross-origin canvas* (so it can zoom / draw
> boxes) — so S3 URLs fail to load in LS even though they open fine in a browser.
> Fix: serve the images we already have **on disk** with CORS enabled.

### Recommended — local CORS static server (no LS storage, no env vars)

**Terminal 1 — image server** (leave it running the whole time you label):
```powershell
uv run python tools/labeling/serve_images.py
# if `uv run` won't stay up, use the venv directly:
# .\.venv\Scripts\python.exe tools\labeling\serve_images.py
```
It prints `Serving … http://127.0.0.1:8081/<id>.jpg` and **stays running** (does not
return to the prompt). Sanity-check by opening
<http://127.0.0.1:8081/00015.jpg> in a browser — you should see the bin photo.

**Terminal 2 — Label Studio**:
```powershell
uvx label-studio
```

In the browser:
1. **Create a project** (e.g., "Amazon Bin Sense").
2. **Settings → Labeling Interface → Code** → paste
   [`tools/labeling/label_config.xml`](../tools/labeling/label_config.xml).
3. **Import** `data/label_studio_tasks_http.json` (generated with `--serve http`).
4. Label with **hotkey `1`**, drag to draw, scroll to **zoom** (essential for dense bins).

**Do NOT** add a Cloud Storage → Local files source and click *Sync* — that's a
different mechanism; Sync scans the folder as if each `.jpg` were a JSON task and
errors with `… is not a JSON file`. The static server needs none of that.

Regenerate the http task set (uv):
```powershell
uv run python tools/labeling/make_label_studio_tasks.py --serve http --output data/label_studio_tasks_http.json
```
Different port? run the server with `--port <p>` and add `--http-base http://127.0.0.1:<p>`.

### Alternative — Label Studio local-files serving
Set `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` and
`LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<data dir>` in the shell **before**
`uvx label-studio`, register a *Local files* source storage at the data dir (Save,
**do not Sync**), and import `data/label_studio_tasks_localfiles.json`
(`--serve local-files`). More moving parts than the static server — prefer the
server above unless you have a reason not to.

---

## 4. Export to YOLO

1. **Export → YOLO** (a zip: `images/`, `labels/`, `classes.txt`, `notes.json`).
2. Copy the `labels/*.txt` into **`data/labels/`** (on the Drive data root —
   `BINSENSE_DATA_DIR`). One `.txt` per image; each line is
   `0 cx cy w h` (normalized 0–1; class **0 = item**).
3. `classes.txt` should contain exactly one line: `item`.

**Bin without any item** → an **empty** `.txt` file (or no file) is correct; that's a
valid negative for YOLO.

---

## 5. Before training — sanity check (don't skip)

The single most common labeling bug is coordinate misalignment (percent vs
normalized, wrong image dims). **Overlay a handful of exported labels back onto their
images and eyeball them** before M4. If boxes are shifted/scaled, the export
mapping is wrong — fix it now, not after a training run. (Ask and I'll drop in a
small overlay-check script.)

---

## 6. Guardrails (from the anchor)

- These 120 bins are **all non-eval** — safe to train on. The **eval gold set** (the
  130 held-out bins) is hand-labeled **separately** and is **never** exported into
  `data/labels/` / trained on. Keeping that wall intact is what makes the M4/M7
  metrics trustworthy.
- Keep the convention **identical** across manual, SAM, and zero-shot passes, or the
  detector learns three different definitions of "an item."
