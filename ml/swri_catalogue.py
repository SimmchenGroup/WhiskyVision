"""Catalogue every SWRI whisky-web image and map it to the SWRI key (19 Nov 2026).

    python swri_catalogue.py "<SWRI_Samples>" "<SWRI_Key.csv>" "<out dir>"

One row per DROPLET (sample number × droplet volume × folder): its tiles, whether a
stitched image exists, and the product it maps to. Handles the naming variants found:
  S25-0227_4X_0.5uL_01.tiff        (folder/tile index)
  S18_1901_4X_0.5uL_646.tiff       (underscore in the number)
  JL_S19-2283__1uL_001.tiff        (prefix, double underscore, "mono" variants)
  S19-2283_4X_1uL_001.tif          (Thor)
  S25-0227_4X_0.5uL_stitched.tif   (in Stitched/)
Writes swri_droplets.csv, swri_unmapped.csv and swri_summary.md.
"""
import csv
import os
import re
import sys
from collections import defaultdict

root, keyf, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
key = {r["sample_number"]: r for r in csv.DictReader(open(keyf, encoding="utf-8"))}

NUM = re.compile(r"S(\d{2})[-_](\d{4})")
VOL = re.compile(r"(?<![\d.])(0\.5|1)\s*uL", re.I)


def parse(fname):
    m = NUM.search(fname)
    if not m:
        return None
    num = f"S{m.group(1)}-{m.group(2)}"
    v = VOL.search(fname)
    vol = (v.group(1) + "uL") if v else "0.5uL?"          # Old/: files without a volume tag
    variant = "mono" if "mono" in fname.lower() else ""
    stitched = "_stitched" in fname.lower()
    return num, vol, variant, stitched


drops = defaultdict(lambda: dict(tiles=0, stitched="", files=[]))
skipped = []
for dirpath, _, files in os.walk(root):
    rel = os.path.relpath(dirpath, root).replace("\\", "/")
    if rel.startswith(("Normal/Thor", "Normal/Old")):
        continue                                            # DFK only (Josh, 19 Nov 2026)
    group = rel[:-len("/Stitched")] if rel.endswith("/Stitched") else rel
    for f in files:
        if not f.lower().endswith((".tif", ".tiff")) or f.startswith("."):
            continue
        p = parse(f)
        if p is None:
            skipped.append(os.path.join(rel, f))
            continue
        num, vol, variant, st = p
        k = (group, num, vol, variant)
        if st:
            drops[k]["stitched"] = os.path.join(dirpath, f)
        else:
            drops[k]["tiles"] += 1
            drops[k]["files"].append(f)

# Probable typos in file names (19 Nov 2026, to confirm with Josh): numbers that are not in
# any key but match exactly one key entry with a different year prefix, or (Purple) one
# digit off a product that is one droplet short.
ALIAS_NOTE = {}
def resolve(num, group):
    if num in key:
        return key[num], ""
    if num.startswith("S20-") and ("S18-" + num[4:]) in key and "/Orange" in group or "/Pink" in group and             ("S18-" + num[4:]) in key:
        return key["S18-" + num[4:]], f"file says {num}; is S18-{num[4:]} (typo confirmed by Josh 19 Nov 2026)"
    manual = {"S22-1934": "S22-1954", "S22-2139": "S22-2149"}
    if num in manual:
        return key[manual[num]], f"file says {num}; is {manual[num]} (typo confirmed by Josh 19 Nov 2026)"
    return None, ""


rows, unmapped = [], []
for (group, num, vol, variant), d in sorted(drops.items()):
    k, alias = resolve(num, group)
    if k is None and group.startswith("Counterfeit"):
        k = dict(product="counterfeit (no key)", product_id="counterfeit", category="Counterfeit", set="counterfeit")
    parts = group.split("/")
    cls = parts[0]                                          # Counterfeit / Different_Years / Normal
    sub = "/".join(parts[1:])                              # colour folder (+ scope for Normal)
    scope = parts[1] if cls == "Normal" and len(parts) > 2 else ("DFK?" if cls != "Normal" else "")
    row = dict(group=group, set_folder=cls, subfolder=sub, scope=scope, sample_number=num, volume=vol,
               variant=variant, tiles=d["tiles"], stitched=bool(d["stitched"]), stitched_path=d["stitched"],
               product=k["product"] if k else "", product_id=k["product_id"] if k else "",
               category=k["category"] if k else "", key_set=k["set"] if k else "", mapping_note=alias)
    rows.append(row)
    if not k:
        unmapped.append(row)

cols = list(rows[0].keys())
for name, rs in (("swri_droplets.csv", rows), ("swri_unmapped.csv", unmapped)):
    with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rs)

# summary
by_set = defaultdict(lambda: [0, 0, set()])
for r in rows:
    b = by_set[(r["set_folder"], r["volume"])]
    b[0] += 1
    b[1] += r["stitched"]
    b[2].add(r["sample_number"])
prod = defaultdict(lambda: defaultdict(int))
for r in rows:
    if r["product_id"]:
        prod[r["product_id"]][(r["set_folder"], r["key_set"])] += 1
lines = ["# SWRI image catalogue", "",
         f"{len(rows)} droplets ({sum(r['stitched'] for r in rows)} stitched) from "
         f"{len({r['sample_number'] for r in rows})} sample numbers; "
         f"{len({r['sample_number'] for r in unmapped})} sample numbers not in the key; "
         f"{len(skipped)} image files without a sample number.", "",
         "| Folder | Volume | Droplets | Stitched | Sample numbers |", "|---|---|---|---|---|"]
for (s, v), (n, st, nums) in sorted(by_set.items()):
    lines.append(f"| {s} | {v} | {n} | {st} | {len(nums)} |")
lines += ["", "## Products with droplets in more than one set or year (bottle-to-bottle tests)", "",
          "| Product | Droplets by (folder, key set) |", "|---|---|"]
for p, d in sorted(prod.items(), key=lambda t: -len(t[1])):
    if len(d) > 1:
        lines.append(f"| {p} | " + "; ".join(f"{a}/{b}: {n}" for (a, b), n in sorted(d.items())) + " |")
lines += ["", "## Sample numbers not in the key", ""]
lines += sorted({f"- {r['sample_number']} ({r['group']})" for r in unmapped})
if skipped:
    lines += ["", "## Files without a sample number", ""] + [f"- {s}" for s in skipped[:50]]
open(os.path.join(out, "swri_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines[:12]))
