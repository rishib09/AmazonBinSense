import base64, json, sys, io, csv
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

B64_PATH = "analysis/ids_b64.txt"
META_URL = "https://aft-vbi-pds.s3.amazonaws.com/metadata/{}.json"

# 1. decode the CSV of ids
raw = open(B64_PATH, "r").read().strip()
csv_text = base64.b64decode(raw).decode("utf-8", "replace")
rows = [r.strip() for r in csv_text.splitlines()]
header = rows[0]
ids = [r for r in rows[1:] if r]
print(f"Header: {header!r}")
print(f"Total ids in subset: {len(ids)}")

def fetch(i):
    # pad to 5 digits like 00015.json
    key = i.zfill(5)
    url = META_URL.format(key)
    try:
        with urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        bfd = data.get("BIN_FCSKU_DATA", {}) or {}
        n_asin = len(bfd)
        exp_qty = data.get("EXPECTED_QUANTITY", None)
        total_units = sum(int(v.get("quantity", 0)) for v in bfd.values())
        asins = list(bfd.keys())
        return (i, n_asin, exp_qty, total_units, asins, None)
    except (URLError, HTTPError, Exception) as e:
        return (i, None, None, None, None, str(e))

results = []
errors = []
with ThreadPoolExecutor(max_workers=48) as ex:
    for idx, res in enumerate(ex.map(fetch, ids), 1):
        if res[1] is None:
            errors.append((res[0], res[5]))
        else:
            results.append(res)
        if idx % 500 == 0:
            print(f"  fetched {idx}/{len(ids)} ...", flush=True)

print(f"\nFetched OK: {len(results)}  Errors: {len(errors)}")
if errors[:5]:
    print("Sample errors:", errors[:5])

dist = Counter(r[1] for r in results)
total = len(results)
single = dist.get(1, 0)
print("\n--- Distinct-ASIN-types per bin ---")
for k in sorted(dist):
    print(f"  {k:>2} product type(s): {dist[k]:>5} bins  ({100*dist[k]/total:5.1f}%)")

print(f"\nSINGLE-ASIN bins: {single} / {total} = {100*single/total:.1f}%")

# how many of the single-asin bins have quantity > 1 (multiple identical units = useful crops)
single_multi_unit = sum(1 for r in results if r[1] == 1 and (r[3] or 0) > 1)
single_total_units = sum((r[3] or 0) for r in results if r[1] == 1)
print(f"  of which qty>1 (multiple identical units): {single_multi_unit}")
print(f"  total item units inside single-ASIN bins: {single_total_units}")

# unique ASIN universe
all_asins = set()
single_asins = set()
for r in results:
    for a in (r[4] or []):
        all_asins.add(a)
    if r[1] == 1:
        single_asins.update(r[4] or [])
print(f"\nUnique ASINs across ALL bins: {len(all_asins)}")
print(f"Unique ASINs that appear as a SINGLE-ASIN bin (gallery-able for free): {len(single_asins)}")

# empty bins
empty = dist.get(0, 0)
print(f"\nEmpty bins (0 products): {empty}")
