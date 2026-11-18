"""
identify_core.py — "which whisky made this web?" by image similarity.

Adapted from Khalifa Mohamed's "Whisky Web Identification.py" (installation notes, Jan 2026):
every image is turned grey, shrunk to 128 × 128 and described by OpenCV's HOG
(Histogram of Oriented Gradients) features. A test image is compared with every
reference image by cosine similarity, and the closest reference wins
(nearest-neighbour matching).

Reference folder layout (one sub-folder per whisky):

    Learning_Database/
        A/  image1.tif  image2.tif …
        B/  …
    Whisky_Key.xlsx      optional: column 1 = folder name (A, B, …), column 2 = whisky name

Without a key the folder names are shown as they are, so folders can simply be
named after the whisky.

Test images named like the original test set ("O_4.tif" = class O, picture 4),
or kept in a folder named after their class, are scored as right or wrong.

Changes from the original script:
  * the result shows the whisky of the *matched* reference. The script looked up
    the test file's own prefix, so it printed the expected answer, not the match.
  * scikit-learn / pandas are replaced by the equivalent numpy / openpyxl code,
    and images are read with stitcher_core.read_rgb (16-bit and compressed TIFFs,
    non-ASCII paths), so the scores are the same but fewer packages are needed.
"""
import csv
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

import stitcher_core as core

FEATURE_EDGE = 128
KEY_EXTS = (".xlsx", ".xlsm", ".csv")
BUILTIN_KEY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Whisky_Key.csv")

FEATURE_DIR = os.path.join(os.path.expanduser("~"), ".whisky_stitcher_features")
FEATURE_VERSION = "hog128-v1"   # change if extract_features changes, so old files aren't reused
READERS = 6                     # images read at once (reading over the network is the slow part)

_feature_cache = {}      # (path, mtime, size) -> feature vector
_file_stats = {}         # path -> (mtime, size), noted while scanning: asking a network
                         # drive for each file separately takes ~0.1 s per file


# --------------------------------------------------------------------------- #
#  Features
# --------------------------------------------------------------------------- #
def opencv_problem():
    """Why matching can't run with the installed OpenCV, or None if it's fine."""
    fix = (f'Close the app and double-click "Start Stitcher" again to fix it, or run:\n'
           f'"{sys.executable}" -m pip install --user "opencv-python-headless<5" openpyxl')
    try:
        import cv2
    except ImportError:
        return "The image-matching part (OpenCV) isn't installed.\n\n" + fix
    if not hasattr(cv2, "HOGDescriptor"):
        return (f"This computer has OpenCV {getattr(cv2, '__version__', '?')}, which no longer includes "
                f"the HOG features Identify uses (they were removed in OpenCV 5). "
                f"OpenCV 4 is needed.\n\n" + fix)
    return None


def extract_features(gray):
    """HOG features of a greyscale uint8 image (same settings as the original script)."""
    import cv2
    img = cv2.resize(gray, (FEATURE_EDGE, FEATURE_EDGE))
    return cv2.HOGDescriptor().compute(img).flatten()


def read_gray(path):
    import cv2
    return cv2.cvtColor(core.read_rgb(path), cv2.COLOR_RGB2GRAY)


def image_features(path):
    """Features of one image file. Kept in memory and in ~/.whisky_stitcher_features
    until the file changes, so each image is only read once."""
    if path in _file_stats:
        mtime, size = _file_stats[path]
    else:
        st = os.stat(path)
        mtime, size = st.st_mtime, st.st_size
    key = (path, mtime, size)
    if key in _feature_cache:
        return _feature_cache[key]
    name = hashlib.sha1(f"{FEATURE_VERSION}|{path}|{mtime}|{size}".encode("utf-8")).hexdigest()
    disk = os.path.join(FEATURE_DIR, name + ".npy")
    try:
        f = np.load(disk)
    except (OSError, ValueError):
        f = extract_features(read_gray(path))
        try:
            os.makedirs(FEATURE_DIR, exist_ok=True)
            tmp = disk + f".{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, f)
            os.replace(tmp, disk)
        except OSError:
            pass
    _feature_cache[key] = f
    return f


def _unit_rows(X):
    """Scale rows to length 1, so a dot product is the cosine similarity."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# --------------------------------------------------------------------------- #
#  Reference database
# --------------------------------------------------------------------------- #
def _is_image(fname):
    return fname.lower().endswith(core.IMG_EXTS) and not fname.startswith(".")


def is_stitched(path):
    """A stitched result: named "…_stitched" or kept in a "Stitched" folder."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    return stem.endswith("_stitched") or os.path.basename(os.path.dirname(path)) == core.OUT_DIR_NAME


def code_order(label):
    """Key order: A…Z, AA…AF, and 1, 2 … 10 numerically; other names alphabetically."""
    short = len(label) <= 3 and label.isalnum()
    return (0, len(label), label.upper()) if short else (1, 0, label.lower())


def class_images(sub):
    """Every image in one whisky's folder, searched recursively. Each file's
    date and size come with the folder listing and are kept in _file_stats."""
    found, dirs = [], []
    with os.scandir(sub) as it:
        entries = sorted(it, key=lambda e: e.name)
    for e in entries:
        if e.name.startswith((".", "_")):
            continue
        if e.is_dir():
            dirs.append(e.path)
        elif _is_image(e.name):
            st = e.stat()
            _file_stats[e.path] = (st.st_mtime, st.st_size)
            found.append(e.path)
    for d in dirs:
        found += class_images(d)
    return found


_scan_cache = {}           # normcase(folder) -> (time, result)
_scan_locks = {}           # normcase(folder) -> Lock
SCAN_CACHE_SECONDS = 600   # the Identify and Gallery tabs share one scan for this long
                           # (Refresh / changing folder / finishing a stitch scan again)


def scan_classes(folder, fresh=False):
    """([(label, [reference images])], [labels left out because they're not stitched yet]).

    If any whisky folder has stitched images ("…_stitched" or in a "Stitched"
    folder), only stitched images are used, so raw tiles are never compared, and
    whisky folders without any are left out. If none do (a plain set of finished
    images, like the original Learning_Database), every image is used.

    The result is cached for SCAN_CACHE_SECONDS (a scan of a big folder on a network
    drive takes seconds); fresh=True scans again."""
    key = os.path.normcase(os.path.normpath(folder))
    started = time.monotonic()
    with _scan_locks.setdefault(key, threading.Lock()):   # a scan already running is waited for
        hit = _scan_cache.get(key)
        if hit and (hit[0] >= started or (not fresh and started - hit[0] < SCAN_CACHE_SECONDS)):
            result = hit[1]
        else:
            result = _scan_classes(folder)
            _scan_cache[key] = (time.monotonic(), result)
    return [(label, list(paths)) for label, paths in result[0]], list(result[1])


def prefetch_scan(folder):
    """Scan `folder` in a background thread, so the tabs find it cached."""
    if folder and os.path.isdir(folder):
        threading.Thread(target=lambda: scan_classes(folder), daemon=True).start()


def forget_images(folder, paths):
    """Drop images that were moved away from the cached scan of `folder`, so the
    tabs can update without scanning the whole folder again."""
    hit = _scan_cache.get(os.path.normcase(os.path.normpath(folder)))
    if not hit:
        return
    gone = {os.path.normcase(p) for p in paths}
    found, unstitched = hit[1]
    found = [(label, [p for p in ps if os.path.normcase(p) not in gone]) for label, ps in found]
    unstitched = list(unstitched) + [label for label, ps in found if not ps]
    found = [(label, ps) for label, ps in found if ps]
    _scan_cache[os.path.normcase(os.path.normpath(folder))] = (hit[0], (found, unstitched))


def _scan_classes(folder):
    found = []
    for label in sorted(os.listdir(folder), key=code_order):
        sub = os.path.join(folder, label)
        if os.path.isdir(sub) and not label.startswith((".", "_")):
            images = class_images(sub)
            if images:
                found.append((label, images))
    stitched = [(label, [p for p in images if is_stitched(p)]) for label, images in found]
    if not any(paths for _, paths in stitched):
        return found, []
    return ([(label, paths) for label, paths in stitched if paths],
            [label for label, paths in stitched if not paths])


def scan_database(folder):
    """[(label, path)] for every reference image in the class sub-folders of `folder`."""
    return [(label, p) for label, paths in scan_classes(folder)[0] for p in paths]


@dataclass
class Database:
    folder: str
    labels: np.ndarray                     # class of each reference image
    paths: list                            # file of each reference image
    X: np.ndarray                          # unit-length feature rows
    skipped: list = field(default_factory=list)   # [(path, reason)]

    @property
    def classes(self):
        return sorted(set(self.labels.tolist()))

    def count(self, label):
        return int(np.sum(self.labels == label))


def build_database(folder, log=print, progress=None, should_stop=lambda: False):
    """Features of every reference image. progress(done, total) is called as it goes."""
    items = scan_database(folder)
    if not items:
        raise ValueError("no images found in sub-folders of this folder.\n"
                         "Put each whisky's reference images in its own sub-folder.")
    results, skipped = [None] * len(items), []
    with ThreadPoolExecutor(READERS) as pool:
        futures = {pool.submit(image_features, path): i for i, (_, path) in enumerate(items)}
        for done, fut in enumerate(as_completed(futures), 1):
            if should_stop():
                for f in futures:
                    f.cancel()
                raise InterruptedError("stopped")
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                path = items[i][1]
                skipped.append((path, str(e)))
                log(f"  skipped {os.path.relpath(path, folder)}: {e}")
            if progress:
                progress(done, len(items))
    keep = [i for i, r in enumerate(results) if r is not None]
    feats = [results[i] for i in keep]
    labels = [items[i][0] for i in keep]
    paths = [items[i][1] for i in keep]
    if not feats:
        raise ValueError("none of the reference images could be opened.")
    return Database(folder, np.array(labels), paths, _unit_rows(feats), skipped)


# --------------------------------------------------------------------------- #
#  Whisky key
# --------------------------------------------------------------------------- #
def find_key(db_folder):
    """A whisky key spreadsheet inside or next to the reference folder,
    else the lab's key that comes with the app (Whisky_Key.csv)."""
    for where in (db_folder, os.path.dirname(os.path.normpath(db_folder))):
        try:
            names = sorted(os.listdir(where))
        except OSError:
            continue
        for fname in names:
            if "key" in fname.lower() and fname.lower().endswith(KEY_EXTS) and not fname.startswith("~$"):
                return os.path.join(where, fname)
    return BUILTIN_KEY if os.path.isfile(BUILTIN_KEY) else ""


def load_key(path, labels=()):
    """{class code: whisky name} from an .xlsx or .csv key (see load_key_details)."""
    return load_key_details(path, labels)[0]


def load_key_details(path, labels=()):
    """({class code: whisky name}, {class code: (letter, sample number)}).

    A key with a "Name" column is read by its headings (see _key_from_table).
    Otherwise column 1 is the code and column 2 the name, and the first row is
    a header unless its first cell is one of `labels`."""
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    else:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            rows = [list(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
        finally:
            wb.close()

    def cell(v):
        if isinstance(v, float) and v.is_integer():
            v = int(v)                     # 1.0 -> "1", so numeric codes match folder names
        return "" if v is None else str(v).strip()
    rows = [[cell(v) for v in r] for r in rows]
    rows = [r for r in rows if any(r)]
    head = [h.lower() for h in rows[0]] if rows else []
    if "name" in head:
        return _key_from_table(head, rows[1:])
    rows = [r[:2] for r in rows if len(r) >= 2]
    if rows and rows[0][0] not in set(labels):
        rows = rows[1:]
    return {code: name for code, name in rows if code and name}, {}


def _is_letter_col(h):
    return h.startswith("letter") or h in ("code", "class", "folder")


def _is_number_col(h):
    """"Sample Number", "Sample No", "Number" – but not "Sample (50mL) price"."""
    return ("price" not in h and "cost" not in h
            and (h in ("number", "no", "no.", "#") or ("sample" in h and ("number" in h or " no" in h))))


def _key_from_table(head, rows):
    """Key with headed columns, like the lab's table:
    Letter Designation | Sample Number | Name | Age | Strength (abv) | (prices…).
    Folders may be named by letter or by number. Repeated names get their age
    (or strength) added, e.g. "Glengoyne 15 YO", "Royal Lochnagar (60% abv)"."""
    def col(test):
        return next((i for i, h in enumerate(head) if test(h)), None)

    def get(r, i):
        return r[i] if i is not None and i < len(r) else ""
    name_i = head.index("name")
    age_i = col(lambda h: h == "age" or h.startswith("age "))
    abv_i = col(lambda h: "strength" in h or "abv" in h)
    letter_i = col(_is_letter_col)
    number_i = col(_is_number_col)
    if letter_i is None and number_i is None:
        letter_i = 0 if name_i != 0 else None
    repeats = {}
    for r in rows:
        n = get(r, name_i).lower()
        repeats[n] = repeats.get(n, 0) + 1
    names, ids = {}, {}
    for r in rows:
        name = get(r, name_i)
        if not name:
            continue
        if repeats[name.lower()] > 1:
            age, abv = get(r, age_i), get(r, abv_i)
            if age.replace(".", "", 1).isdigit():
                name = f"{name} {age} YO"
            elif abv:
                name = f"{name} ({abv}% abv)"
        letter, number = get(r, letter_i), get(r, number_i)
        for code in (letter, number):
            if code:
                names.setdefault(code, name)
                ids.setdefault(code, (letter, number))
    return names, ids


# --------------------------------------------------------------------------- #
#  Matching
# --------------------------------------------------------------------------- #
@dataclass
class Match:
    path: str
    label: str = ""                # class of the closest reference image
    score: float = 0.0             # its cosine similarity
    ref_path: str = ""             # the closest reference image
    ranking: list = field(default_factory=list)   # [(label, best score, best path)], best first
    expected: str = None           # class from the file / folder name, if it has one
    error: str = ""

    @property
    def correct(self):
        return None if self.expected is None or self.error else self.expected == self.label


def expected_label(path, classes):
    """The class a test image should be, from "O_4.tif" (prefix before "_") or its folder name."""
    lookup = {c.lower(): c for c in classes}
    stem = os.path.splitext(os.path.basename(path))[0]
    for guess in (stem.split("_")[0], os.path.basename(os.path.dirname(path))):
        if guess.lower() in lookup:
            return lookup[guess.lower()]
    return None


def identify(path, db):
    """Compare one image with the database. Errors are returned in Match.error."""
    m = Match(path=path, expected=expected_label(path, db.classes))
    try:
        sims = db.X @ _unit_rows(image_features(path))[0]
    except Exception as e:
        m.error = str(e)
        return m
    i = int(np.argmax(sims))
    m.label, m.score, m.ref_path = str(db.labels[i]), float(sims[i]), db.paths[i]
    best = {}
    for j in np.argsort(-sims):
        lab = str(db.labels[j])
        if lab not in best:
            best[lab] = (lab, float(sims[j]), db.paths[j])
    m.ranking = list(best.values())
    return m


def leave_one_out(db):
    """Match every reference image against all the others.
    Returns ({class: (correct, total)}, [(path, true class, matched class, score)] of misses)."""
    S = db.X @ db.X.T
    np.fill_diagonal(S, -np.inf)
    nearest = np.argmax(S, axis=1)
    per_class, misses = {}, []
    for i, j in enumerate(nearest):
        true, got = str(db.labels[i]), str(db.labels[j])
        ok, total = per_class.get(true, (0, 0))
        per_class[true] = (ok + (true == got), total + 1)
        if true != got:
            misses.append((db.paths[i], true, got, float(S[i, j])))
    return per_class, misses


def list_test_images(paths):
    """Image files from a mix of files and folders (folders are searched recursively)."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
                out += [os.path.join(root, f) for f in sorted(files) if _is_image(f)]
        elif os.path.isfile(p) and _is_image(os.path.basename(p)):
            out.append(p)
    seen = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def write_csv(matches, path, names=None):
    names = names or {}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "match_code", "match_whisky", "similarity", "closest_reference",
                    "second_code", "second_similarity", "expected_code", "correct", "error"])
        for m in matches:
            second = m.ranking[1] if len(m.ranking) > 1 else ("", "", "")
            w.writerow([m.path, m.label, names.get(m.label, m.label) if m.label else "",
                        f"{m.score:.4f}" if not m.error else "", m.ref_path,
                        second[0], f"{second[1]:.4f}" if second[1] != "" else "",
                        m.expected or "", "" if m.correct is None else ("yes" if m.correct else "no"),
                        m.error])


if __name__ == "__main__":
    # Command line: python identify_core.py <reference folder> <test image or folder> … [--key Whisky_Key.xlsx]
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("database")
    ap.add_argument("tests", nargs="+")
    ap.add_argument("--key", default=None)
    a = ap.parse_args()
    db = build_database(a.database)
    key_path = a.key if a.key is not None else find_key(a.database)
    names = load_key(key_path, db.classes) if key_path else {}
    for m in (identify(p, db) for p in list_test_images(a.tests)):
        if m.error:
            print(f"{os.path.basename(m.path)}: error: {m.error}")
            continue
        verdict = "" if m.correct is None else ("  right" if m.correct else f"  WRONG (expected {m.expected})")
        print(f"{os.path.basename(m.path)}: nearest match {names.get(m.label, m.label)} "
              f"(similarity {m.score:.3f}){verdict}")
