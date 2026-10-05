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
import os
from dataclasses import dataclass, field

import numpy as np

import stitcher_core as core

FEATURE_EDGE = 128
KEY_EXTS = (".xlsx", ".xlsm", ".csv")

_feature_cache = {}      # (path, mtime, size) -> feature vector


# --------------------------------------------------------------------------- #
#  Features
# --------------------------------------------------------------------------- #
def extract_features(gray):
    """HOG features of a greyscale uint8 image (same settings as the original script)."""
    import cv2
    img = cv2.resize(gray, (FEATURE_EDGE, FEATURE_EDGE))
    return cv2.HOGDescriptor().compute(img).flatten()


def read_gray(path):
    import cv2
    return cv2.cvtColor(core.read_rgb(path), cv2.COLOR_RGB2GRAY)


def image_features(path):
    """Features of one image file, cached until the file changes."""
    st = os.stat(path)
    key = (path, st.st_mtime, st.st_size)
    if key not in _feature_cache:
        _feature_cache[key] = extract_features(read_gray(path))
    return _feature_cache[key]


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


def scan_database(folder):
    """[(label, path)] for every image in the class sub-folders of `folder`."""
    items = []
    for label in sorted(os.listdir(folder)):
        sub = os.path.join(folder, label)
        if not os.path.isdir(sub) or label.startswith((".", "_")):
            continue
        for fname in sorted(os.listdir(sub)):
            path = os.path.join(sub, fname)
            if os.path.isfile(path) and _is_image(fname):
                items.append((label, path))
    return items


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
    feats, labels, paths, skipped = [], [], [], []
    for i, (label, path) in enumerate(items):
        if should_stop():
            raise InterruptedError("stopped")
        try:
            feats.append(image_features(path))
            labels.append(label)
            paths.append(path)
        except Exception as e:
            skipped.append((path, str(e)))
            log(f"  skipped {os.path.relpath(path, folder)}: {e}")
        if progress:
            progress(i + 1, len(items))
    if not feats:
        raise ValueError("none of the reference images could be opened.")
    return Database(folder, np.array(labels), paths, _unit_rows(feats), skipped)


# --------------------------------------------------------------------------- #
#  Whisky key
# --------------------------------------------------------------------------- #
def find_key(db_folder):
    """A whisky key spreadsheet inside or next to the reference folder, or ''."""
    for where in (db_folder, os.path.dirname(os.path.normpath(db_folder))):
        try:
            names = sorted(os.listdir(where))
        except OSError:
            continue
        for fname in names:
            if "key" in fname.lower() and fname.lower().endswith(KEY_EXTS) and not fname.startswith("~$"):
                return os.path.join(where, fname)
    return ""


def load_key(path, labels=()):
    """{class code: whisky name} from the first two columns of an .xlsx or .csv.
    The first row is a header, unless its first cell is one of `labels`."""
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
    rows = [(cell(r[0]), cell(r[1])) for r in rows if len(r) >= 2]
    if rows and rows[0][0] not in set(labels):
        rows = rows[1:]
    return {code: name for code, name in rows if code and name}


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
