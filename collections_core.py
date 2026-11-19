"""
collections_core.py — the image collections the app can show (backend for a
"collection" drop-down in the Gallery, Identify and ML tabs).

    Khalifa samples           <Whisky Webs>/Khalifa Samples/<letter>/Stitched/        (Whisky_Key.csv)
    SWRI · Normal (DFK)       <Whisky Webs>/SWRI_Samples/Normal/DFK/<colour>/Stitched/ (SWRI_Key.csv)
    SWRI · Different years    <Whisky Webs>/SWRI_Samples/Different_Years/<colour>/Stitched/
    SWRI · Counterfeit        <Whisky Webs>/SWRI_Samples/Counterfeit/<colour>/Stitched/  (no key)
    SWRI · all genuine        Normal + Different years, grouped by product

Every collection is scanned into the SAME structure identify_core.scan_classes
returns, so existing tab code can switch collection without other changes:

    classes, unstitched = scan_collection(cid, base)   # [(label, [stitched paths])], [labels]
    names = collection_names(cid, base)                # {label: display name}
    info = image_info(path)                            # sample number, volume, product, year...

SWRI images are grouped by PRODUCT (label = product id, e.g. "laphroaig10"), not by the
colour folders, so the same whisky from different bottles / years sits together.
Only DFK images are used (Josh, 19 Nov 2026: Thor and Old excluded). File-name typos
confirmed by Josh (19 Nov 2026) are corrected through SWRI_ALIASES.
"""
import csv
import os
import re
import threading
import time

import identify_core as idc

HERE = os.path.dirname(os.path.abspath(__file__))
SWRI_KEY = os.path.join(HERE, "SWRI_Key.csv")
STITCHED = "Stitched"

COLLECTIONS = [
    # id, menu title, path below the "4. Whisky Webs" folder, kind
    ("khalifa", "Khalifa samples", "Khalifa Samples", "letters"),
    ("swri_normal", "SWRI · Normal (DFK)", os.path.join("SWRI_Samples", "Normal", "DFK"), "swri"),
    ("swri_years", "SWRI · Different years", os.path.join("SWRI_Samples", "Different_Years"), "swri"),
    ("swri_counterfeit", "SWRI · Counterfeit", os.path.join("SWRI_Samples", "Counterfeit"), "swri"),
    ("swri_genuine", "SWRI · all genuine (by product)", "SWRI_Samples", "swri"),
]
_SKIP = ("Thor", "Old")                     # not DFK: excluded from every SWRI collection

# typos in file names -> real sample number (confirmed by Josh, 19 Nov 2026)
SWRI_ALIASES = {"S22-1934": "S22-1954", "S22-2139": "S22-2149"}
# Normal/DFK/Orange and /Pink files say S20-xxxx for 2018 samples S18-xxxx
_S18_AS_S20_FOLDERS = ("Orange", "Pink")

_NUM = re.compile(r"S(\d{2})[-_](\d{4})")
_VOL = re.compile(r"(?<![\d.])(0\.5|1)\s*uL", re.I)
_key_cache = {}
_scan_cache = {}
_lock = threading.Lock()
SCAN_CACHE_SECONDS = 600


# --------------------------------------------------------------------------- #
#  Locations
# --------------------------------------------------------------------------- #
def whisky_webs_root(folder):
    """The "4. Whisky Webs" folder, given it or any folder inside it (e.g. the Khalifa
    Samples folder the app already remembers)."""
    f = os.path.normpath(folder)
    for _ in range(6):
        if os.path.isdir(os.path.join(f, "SWRI_Samples")) or os.path.isdir(os.path.join(f, "Khalifa Samples")):
            return f
        parent = os.path.dirname(f)
        if parent == f:
            break
        f = parent
    return os.path.normpath(folder)


def available_collections(folder):
    """[(id, title, path)] for the collections that exist under the Whisky Webs folder."""
    root = whisky_webs_root(folder)
    return [(cid, title, os.path.join(root, sub)) for cid, title, sub, _ in COLLECTIONS
            if os.path.isdir(os.path.join(root, sub))]


def collection_path(cid, folder):
    root = whisky_webs_root(folder)
    sub = next(s for c, _, s, _ in COLLECTIONS if c == cid)
    return os.path.join(root, sub)


# --------------------------------------------------------------------------- #
#  SWRI key
# --------------------------------------------------------------------------- #
def load_swri_key(path=SWRI_KEY):
    """{sample number: row} with product, product_id, category, set (2018 standard /
    2020 standard / different years)."""
    if path not in _key_cache:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            _key_cache[path] = {r["sample_number"]: r for r in csv.DictReader(fh)}
    return _key_cache[path]


def swri_products(path=SWRI_KEY):
    """{product_id: display name}: the shortest product name seen for each id."""
    out = {}
    for r in load_swri_key(path).values():
        p, n = r["product_id"], r["product"]
        if p not in out or len(n) < len(out[p]):
            out[p] = n
    return out


def parse_swri_name(fname, folder=""):
    """(sample number as written, corrected sample number, volume, note) or None."""
    m = _NUM.search(fname)
    if not m:
        return None
    written = f"S{m.group(1)}-{m.group(2)}"
    v = _VOL.search(fname)
    vol = (v.group(1) + " µL") if v else ""
    num, note = written, ""
    if written in SWRI_ALIASES:
        num = SWRI_ALIASES[written]
        note = f"file says {written}"
    elif written.startswith("S20-") and os.path.basename(folder) in _S18_AS_S20_FOLDERS \
            and "S18-" + written[4:] in load_swri_key() and written not in load_swri_key():
        num = "S18-" + written[4:]
        note = f"file says {written}"
    return written, num, vol, note


def image_info(path):
    """What is known about one stitched image: collection, sample number, volume,
    product, category, set/year, source folder. Works for Khalifa images too."""
    p = os.path.normpath(path)
    parts = p.split(os.sep)
    folder = os.path.dirname(p)
    if os.path.basename(folder) == STITCHED:
        folder = os.path.dirname(folder)
    info = dict(path=path, file=os.path.basename(p), source_folder=os.path.basename(folder))
    if "SWRI_Samples" in parts:
        i = parts.index("SWRI_Samples")
        info["collection"] = {"Normal": "swri_normal", "Different_Years": "swri_years",
                              "Counterfeit": "swri_counterfeit"}.get(parts[i + 1], "swri")
        info["scope"] = parts[i + 2] if parts[i + 1] == "Normal" else "DFK"
        parsed = parse_swri_name(os.path.basename(p), folder)
        if parsed:
            written, num, vol, note = parsed
            k = load_swri_key().get(num)
            info.update(sample_number=num, written_as=written, volume=vol, note=note,
                        year="20" + num[1:3])
            if info["collection"] == "swri_counterfeit":
                info.update(product="Counterfeit (no key)", product_id="counterfeit:" + num,
                            category="Counterfeit", set="counterfeit")
            elif k:
                info.update(product=k["product"], product_id=k["product_id"], category=k["category"],
                            set=k["set"])
            else:
                info.update(product="", product_id="", category="", set="", note=(note + "; " if note else "")
                            + "not in the SWRI key")
    else:
        info["collection"] = "khalifa"
        info["sample_number"] = os.path.basename(folder)
    return info


# --------------------------------------------------------------------------- #
#  Scanning
# --------------------------------------------------------------------------- #
def _stitched_in(folder):
    """Stitched images directly in folder/Stitched (not _replaced, _layouts...)."""
    d = os.path.join(folder, STITCHED)
    out = []
    try:
        with os.scandir(d) as it:
            for e in sorted(it, key=lambda e: e.name):
                if e.is_file() and not e.name.startswith((".", "_")) and \
                        e.name.lower().endswith((".tif", ".tiff")) and "_stitched" in e.name.lower():
                    st = e.stat()
                    idc._file_stats[e.path] = (st.st_mtime, st.st_size)   # fast thumbnails / features
                    out.append(e.path)
    except OSError:
        pass
    return out


def _has_tiles(folder):
    try:
        return any(f.lower().endswith((".tif", ".tiff")) and not f.startswith(".") for f in os.listdir(folder))
    except OSError:
        return False


def _swri_folders(root):
    """Every image folder below root (DFK only), i.e. folders holding tiles or a Stitched/."""
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP and not d.startswith((".", "_"))
                             and d != STITCHED)
        if os.path.isdir(os.path.join(dirpath, STITCHED)) or _has_tiles(dirpath):
            found.append(dirpath)
    return found


def _scan_swri(cid, root, include_unmapped):
    groups, unstitched = {}, []
    roots = [root]
    if cid == "swri_genuine":
        roots = [os.path.join(root, "Normal", "DFK"), os.path.join(root, "Different_Years")]
    for r in roots:
        for f in _swri_folders(r):
            paths = _stitched_in(f)
            if not paths and _has_tiles(f):
                unstitched.append(os.path.relpath(f, r))
            for p in paths:
                inf = image_info(p)
                label = inf.get("product_id") or ("unmapped" if include_unmapped else "")
                if label:
                    groups.setdefault(label, []).append(p)
    # products in A-Z order of their display name; unmapped last
    names = collection_names(cid)
    order = sorted(groups, key=lambda l: (l == "unmapped", label_name(l, names).lower()))
    return [(l, groups[l]) for l in order], unstitched


def scan_collection(cid, folder, fresh=False, include_unmapped=False):
    """([(label, [stitched images])], [folders not stitched yet]) for one collection, in
    the same form as identify_core.scan_classes. folder: the Whisky Webs folder or any
    folder inside it. Cached for SCAN_CACHE_SECONDS (fresh=True to rescan)."""
    path = collection_path(cid, folder)
    kind = next(k for c, _, _, k in COLLECTIONS if c == cid)
    if kind == "letters":
        return idc.scan_classes(path, fresh=fresh)
    key = (cid, os.path.normcase(path), include_unmapped)
    with _lock:
        hit = _scan_cache.get(key)
        if hit and not fresh and time.monotonic() - hit[0] < SCAN_CACHE_SECONDS:
            res = hit[1]
        else:
            res = _scan_swri(cid, path, include_unmapped)
            _scan_cache[key] = (time.monotonic(), res)
    return [(l, list(p)) for l, p in res[0]], list(res[1])


def collection_names(cid, folder=None):
    """{label: display name} for a collection's classes."""
    if cid == "khalifa":
        path = collection_path(cid, folder) if folder else ""
        k = idc.find_key(path) if path else idc.BUILTIN_KEY
        return idc.load_key(k) if k else {}
    names = swri_products()
    names["unmapped"] = "Not in the SWRI key"
    return names


def label_name(label, names):
    """Display name for a label, including counterfeit labels ("counterfeit:S25-0227")."""
    if label.startswith("counterfeit:"):
        return "Counterfeit " + label.split(":", 1)[1]
    return names.get(label, label)


def image_caption(path):
    """Short caption for a thumbnail, e.g. "S20-0027 · 1 µL · 2020" or "001" (Khalifa)."""
    i = image_info(path)
    if i["collection"] == "khalifa":
        return os.path.basename(path).split("_stitched")[0]
    bits = [i.get("sample_number", "?"), i.get("volume", "")]
    if i.get("collection") in ("swri_years", "swri_genuine") or i.get("set") == "different years":
        bits.append(i.get("year", ""))
    return " · ".join(b for b in bits if b)
