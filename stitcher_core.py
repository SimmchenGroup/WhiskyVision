"""
stitcher_core.py — shared logic for the Whisky Webs stitcher (GUI and command line).

  * scan_folder()   finds image tiles and groups them into samples
  * guess_grid()    suggests rows x columns from the tile count
  * tile_positions() lays tiles out in acquisition order (snake / row-by-row ...)
  * run_imagej()    stitches with Fiji's Grid/Collection stitching, run headless
  * run_native()    stitches with the pure-Python engine in stitch_native.py

Naming rule: the number after the LAST "_" (or space) is the tile number,
everything before it is the sample name.

    S24-1273_4X_0.5uL_01.tiff  ->  sample "S24-1273_4X_0.5uL", tile 1
    001_0.tiff                 ->  sample "001",               tile 0

Output for every sample: <folder>/Stitched/<sample>_stitched.tif
The original tiles are never renamed, moved or modified.
"""
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

IMG_EXTS = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg")
NAME_RE = re.compile(r"^(?P<sample>.+?)\s*[_\s](?P<index>\d+)\s*\.(?P<ext>[A-Za-z]+)$")
OUT_DIR_NAME = "Stitched"
LAYOUT_DIR_NAME = "_layouts"

# Order in which the microscope visited the tiles, always starting top-left.
ORDERS = {
    "Snake by rows":       "Row 1 left→right, row 2 right→left, …",
    "Row by row":          "Every row left→right",
    "Snake by columns":    "Column 1 top→bottom, column 2 bottom→top, …",
    "Column by column":    "Every column top→bottom",
}
DEFAULT_ORDER = "Snake by rows"

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".whisky_stitcher.json")

# tifffile logs a warning for every broken file it meets; we report those ourselves.
logging.getLogger("tifffile").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
#  Scanning
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    name: str
    tiles: list                      # [(index, full_path)] sorted by index
    rows: int = 1
    cols: int = 1
    order: str = DEFAULT_ORDER
    warnings: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.tiles)

    @property
    def paths(self):
        return [p for _, p in self.tiles]

    def output_path(self, folder):
        return os.path.join(folder, OUT_DIR_NAME, f"{self.name}_stitched.tif")

    def is_done(self, folder):
        """True if a complete, readable result already exists."""
        return output_ok(self.output_path(folder))

    def problem(self):
        """Return a message if this sample cannot be stitched as configured, else None."""
        if self.rows < 1 or self.cols < 1:
            return "Rows and columns must be at least 1"
        if self.rows * self.cols < self.n:
            return f"{self.n} tiles don't fit in {self.rows}×{self.cols}"
        if self.rows * self.cols - self.n >= max(self.rows, self.cols) and self.n > 1:
            return f"{self.rows}×{self.cols} has a whole empty row/column for {self.n} tiles"
        return None


def parse_name(fname):
    """('sample', index) for a tile filename, or None if it doesn't look like a tile."""
    if not fname.lower().endswith(IMG_EXTS):
        return None
    if fname.lower().endswith("_stitched.tif"):
        return None
    m = NAME_RE.match(fname)
    if not m:
        return None
    return m.group("sample").strip(), int(m.group("index"))


def guess_grid(n):
    """Suggested (rows, cols) for n tiles. Square grids first, then wide rectangles."""
    presets = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 6: (2, 3), 8: (2, 4),
               9: (3, 3), 12: (3, 4), 16: (4, 4), 20: (4, 5), 25: (5, 5), 36: (6, 6)}
    if n in presets:
        return presets[n]
    cols = 1
    while cols * cols < n:
        cols += 1
    rows = -(-n // cols)
    return rows, cols


def scan_folder(folder):
    """Return (samples, ignored_filenames). Only the top level of `folder` is scanned."""
    groups, ignored = {}, []
    for fname in sorted(os.listdir(folder)):
        full = os.path.join(folder, fname)
        if not os.path.isfile(full):
            continue
        parsed = parse_name(fname)
        if parsed is None:
            if fname.lower().endswith(IMG_EXTS):
                ignored.append(fname)
            continue
        sample, index = parsed
        groups.setdefault(sample, []).append((index, full))

    samples = []
    for name in sorted(groups):
        tiles = sorted(groups[name], key=lambda t: t[0])
        s = Sample(name=name, tiles=tiles)
        s.rows, s.cols = guess_grid(s.n)
        idx = [i for i, _ in tiles]
        if len(set(idx)) != len(idx):
            s.warnings.append("Two files have the same tile number")
        elif idx != list(range(idx[0], idx[0] + len(idx))):
            s.warnings.append(f"Tile numbers have gaps ({idx[0]}…{idx[-1]})")
        exts = {os.path.splitext(p)[1].lower() for _, p in tiles}
        if len(exts) > 1:
            s.warnings.append("Mixed file types: " + ", ".join(sorted(exts)))
        samples.append(s)
    return samples, ignored


# --------------------------------------------------------------------------- #
#  Layout
# --------------------------------------------------------------------------- #
def tile_positions(n, rows, cols, order=DEFAULT_ORDER):
    """(row, col) of each tile, in file order."""
    pos = []
    for i in range(n):
        if order in ("Snake by rows", "Row by row"):
            r, c = divmod(i, cols)
            if order == "Snake by rows" and r % 2 == 1:
                c = cols - 1 - c
        else:
            c, r = divmod(i, rows)
            if order == "Snake by columns" and c % 2 == 1:
                r = rows - 1 - r
        pos.append((r, c))
    return pos


def output_ok(path):
    """True if `path` exists and is a complete image file (an interrupted save
    leaves a TIFF with no image data, which should be redone, not skipped)."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        image_size(path)
        return True
    except Exception:
        return False


def _is_tiff(path):
    return path.lower().endswith((".tif", ".tiff"))


def image_size(path):
    """(width, height) without decoding the pixels."""
    if _is_tiff(path):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tf:
                if not tf.pages:
                    raise ValueError("the file is incomplete (an earlier save was interrupted)")
                page = tf.pages[0]
                return page.imagewidth, page.imagelength
        except ImportError:
            pass
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def read_rgb(path):
    """Load any tile as an HxWx3 uint8 array.
    TIFFs are read with tifffile when installed, because Pillow's libtiff
    decoder crashes Python outright with some Anaconda builds."""
    import numpy as np
    if _is_tiff(path):
        try:
            import tifffile
        except ImportError:
            tifffile = None
        if tifffile is not None:
            try:
                a = tifffile.imread(path)
            except ValueError as e:
                if "imagecodecs" in str(e):
                    raise ValueError("this TIFF is compressed; install the 'imagecodecs' "
                                     "package to read it (pip install imagecodecs)") from e
                raise
            if a.size == 0:
                raise ValueError("the file is incomplete (an earlier save was interrupted)")
            a = np.squeeze(a)
            if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
                a = np.moveaxis(a, 0, -1)          # planar CxHxW -> HxWxC
            if a.ndim == 3 and a.shape[-1] not in (3, 4):
                a = a[0]                           # stack: first plane only
            if a.ndim == 2:
                a = np.stack([a] * 3, axis=-1)
            a = a[..., :3]
            if a.dtype != np.uint8:
                maxval = 65535.0 if a.dtype == np.uint16 else float(a.max() or 1)
                a = np.clip(a.astype(np.float32) * (255.0 / maxval), 0, 255)
            return np.ascontiguousarray(a, dtype=np.uint8)
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def write_tile_configuration(sample, folder, overlap):
    """Write an ImageJ TileConfiguration with each tile's approximate position.
    Returns the path relative to `folder` (Fiji resolves it against the directory)."""
    w, h = image_size(sample.paths[0])
    layout_dir = os.path.join(folder, OUT_DIR_NAME, LAYOUT_DIR_NAME)
    os.makedirs(layout_dir, exist_ok=True)
    lines = ["# Generated by Whisky Webs stitcher", "dim = 2", ""]
    for (r, c), path in zip(tile_positions(sample.n, sample.rows, sample.cols, sample.order),
                            sample.paths):
        x = c * (1.0 - overlap) * w
        y = r * (1.0 - overlap) * h
        lines.append(f"{os.path.basename(path)}; ; ({x:.1f}, {y:.1f})")
    cfg = os.path.join(layout_dir, f"{sample.name}.txt")
    with open(cfg, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return f"{OUT_DIR_NAME}/{LAYOUT_DIR_NAME}/{sample.name}.txt"


# --------------------------------------------------------------------------- #
#  ImageJ / Fiji engine
# --------------------------------------------------------------------------- #
def _fiji_launcher_score(name):
    """How well a file name matches this computer's Fiji launcher (0 = not a launcher)."""
    import platform
    low = name.lower()
    if not low.startswith(("fiji-", "imagej-")) or low.endswith((".sh", ".txt", ".cfg", ".toml", ".bat")):
        return 0
    if sys.platform.startswith("win"):
        plat_ok = low.endswith(".exe")
    elif sys.platform == "darwin":
        plat_ok = "macos" in low
    else:
        plat_ok = "linux" in low
    if not plat_ok:
        return 0
    arm = platform.machine().lower() in ("arm64", "aarch64")
    score = 2
    if ("arm64" in low) == arm or "universal" in low:
        score += 1                         # matching CPU type
    if low.startswith("fiji-"):
        score += 1                         # new launcher preferred over the old ImageJ-* one
    return score


def resolve_fiji(path):
    """Turn whatever the user picked (the launcher itself, the Fiji folder, or the
    macOS Fiji.app bundle) into the launcher program's path. Returns '' if none."""
    if not path:
        return ""
    if os.path.isfile(path):
        return path if _fiji_launcher_score(os.path.basename(path)) else ""
    best, best_score = "", 0
    for d in (path, os.path.join(path, "Fiji"), os.path.join(path, "Fiji.app"),
              os.path.join(path, "Contents", "MacOS"),
              os.path.join(path, "Fiji.app", "Contents", "MacOS"),
              os.path.join(path, "Fiji", "Contents", "MacOS")):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            full = os.path.join(d, name)
            score = _fiji_launcher_score(name)
            if not score and d.endswith(os.path.join("Contents", "MacOS")) and os.access(full, os.X_OK):
                score = 1                  # inside a macOS .app: any program is the launcher
            if score > best_score and os.path.isfile(full):
                best, best_score = full, score
    return best


def find_fiji():
    """Best-effort search for a Fiji launcher (Windows, macOS or Linux). Returns a path or ''."""
    home = os.path.expanduser("~")
    roots = [home, os.path.join(home, "Desktop"), os.path.join(home, "Downloads"),
             os.path.join(home, "OneDrive", "Desktop"), os.path.join(home, "Documents"),
             os.path.join(home, "Applications"), "/Applications", "/opt",
             "C:\\", "C:\\Program Files", "C:\\Program Files (x86)"]
    for root in roots:
        for pattern in ("Fiji*", "fiji*", "*/Fiji*", "*/fiji*"):
            for d in glob.glob(os.path.join(root, pattern)):
                if os.path.isdir(d):
                    exe = resolve_fiji(d)
                    if exe:
                        return exe
    return ""


def _macro_str(s):
    return s.replace("\\", "/").replace('"', '\\"')


def build_macro(samples, folder, layouts, refine, progress_file):
    """ImageJ macro stitching every sample. `layouts` maps sample name ->
    TileConfiguration path (from write_tile_configuration) for samples of 3+ tiles."""
    folder_fwd = _macro_str(os.path.abspath(folder))
    prog = _macro_str(progress_file)
    lines = ['setBatchMode(true);']
    for s in samples:
        out_fwd = _macro_str(os.path.abspath(s.output_path(folder)))
        name = _macro_str(s.name)
        lines.append(f'File.append("START|{name}", "{prog}");')
        if s.n == 1:
            lines.append(f'open("{_macro_str(s.paths[0])}");')
        elif s.n == 2:
            # Pairwise stitching finds the direction itself (side-by-side or stacked).
            a, b = (_macro_str(os.path.basename(p)) for p in s.paths)
            lines += [
                f'open("{_macro_str(s.paths[0])}");',
                f'open("{_macro_str(s.paths[1])}");',
                f'run("Pairwise stitching", "first_image=[{a}] second_image=[{b}] '
                'fusion_method=[Linear Blending] fused_image=WW_fused check_peaks=5 '
                + ('compute_overlap ' if refine else '') +
                'x=0.0000 y=0.0000 registration_channel_image_1=[Average all channels] '
                'registration_channel_image_2=[Average all channels]");',
                'selectImage("WW_fused");',
            ]
        else:
            layout = layouts[s.name]
            lines.append(
                'run("Grid/Collection stitching", "type=[Positions from file] '
                'order=[Defined by TileConfiguration] '
                f'directory=[{folder_fwd}] layout_file=[{_macro_str(layout)}] '
                'fusion_method=[Linear Blending] regression_threshold=0.30 '
                'max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 '
                + ('compute_overlap subpixel_accuracy ' if refine else '') +
                'computation_parameters=[Save computation time (but use more RAM)] '
                'image_output=[Fuse and display]");')
        lines += [
            'if (bitDepth() != 24) run("RGB Color");',
            f'saveAs("Tiff", "{out_fwd}");',
            'close("*");',
            f'File.append("DONE|{name}", "{prog}");',
            '',
        ]
    return "\n".join(lines)


def run_imagej(samples, folder, fiji_path, overlap=0.20, refine=True,
               log=print,
               on_start=None, on_done=None, should_stop=lambda: False):
    """Stitch `samples` with Fiji in headless mode.

    The macro reports progress by appending to a small text file (Fiji's console
    output isn't reliably visible to other programs). If Fiji fails on one
    sample, that sample is marked failed and Fiji is restarted for the rest, so
    one bad sample doesn't stop the whole batch.
    on_start(sample_name) / on_done(sample_name, ok: bool) are called around each sample."""
    fiji_path = resolve_fiji(fiji_path)
    if not fiji_path:
        raise FileNotFoundError('Fiji was not found. Click "Find..." next to "Fiji program" and choose it.')
    os.makedirs(os.path.join(folder, OUT_DIR_NAME), exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    # Check every sample up front so one unreadable tile only fails its own sample.
    remaining, layouts = [], {}
    for s in samples:
        try:
            missing = [p for p in s.paths if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(f"missing tile {os.path.basename(missing[0])}")
            if s.n > 2:
                layouts[s.name] = write_tile_configuration(s, folder, overlap)
            remaining.append(s)
        except Exception as e:
            log(f"  FAILED    {s.name}: {e}")
            if on_done:
                on_done(s.name, False)

    workdir = tempfile.mkdtemp(prefix="whisky_stitch_")

    try:
        while remaining and not should_stop():
            progress = os.path.join(workdir, "progress.txt")
            macro_path = os.path.join(workdir, "stitch.ijm")
            if os.path.exists(progress):
                os.remove(progress)
            with open(macro_path, "w", encoding="utf-8") as f:
                f.write(build_macro(remaining, folder, layouts, refine, progress))
            log(f"Starting Fiji for {len(remaining)} sample(s) (takes ~10 s)...")
            proc = subprocess.Popen(
                [fiji_path, "--headless", "-macro", macro_path],
                cwd=os.path.dirname(fiji_path), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)

            current, done, seen = None, set(), 0
            while True:
                finished = proc.poll() is not None
                if os.path.exists(progress):
                    with open(progress, encoding="utf-8", errors="replace") as f:
                        events = f.read().splitlines()
                    for ev in events[seen:]:
                        kind, _, name = ev.partition("|")
                        if kind == "START":
                            current = name
                            log(f"  stitching {name}...")
                            if on_start:
                                on_start(name)
                        elif kind == "DONE":
                            done.add(name)
                            current = None
                            log(f"  done      {name}")
                            if on_done:
                                on_done(name, True)
                    seen = len(events)
                if finished:
                    break
                if should_stop():
                    proc.kill()
                    proc.wait()
                    log("Stopped.")
                    return
                time.sleep(0.3)

            remaining = [s for s in remaining if s.name not in done]
            if current is not None:
                log(f"  FAILED    {current}  (Fiji could not stitch it - check the "
                    "rows/columns and order, or try the Python engine)")
                if on_done:
                    on_done(current, False)
                remaining = [s for s in remaining if s.name != current]
            elif remaining:
                # Fiji exited before starting anything: don't loop forever.
                log("  FAILED    Fiji closed before stitching started. Is the Fiji "
                    "path correct, and does it open normally?")
                for s in remaining:
                    if on_done:
                        on_done(s.name, False)
                break
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def save_tiff(rgb, path):
    """Save an HxWx3 uint8 array as a lossless TIFF.
    Uses tifffile (zlib) when installed; otherwise uncompressed via Pillow.
    (Pillow's own LZW writer crashes outright with some Anaconda builds.)"""
    try:
        import tifffile
        tifffile.imwrite(path, rgb, photometric="rgb", compression="zlib")
    except ImportError:
        from PIL import Image
        Image.fromarray(rgb).save(path)


# --------------------------------------------------------------------------- #
#  Native Python engine
# --------------------------------------------------------------------------- #
def run_native(samples, folder, overlap=0.20, refine=True,
               log=print,
               on_start=None, on_done=None, should_stop=lambda: False):
    import stitch_native
    os.makedirs(os.path.join(folder, OUT_DIR_NAME), exist_ok=True)
    for s in samples:
        if should_stop():
            break
        layout = "pair" if s.n == 2 else f"{s.rows}x{s.cols}"
        log(f"  stitching {s.name} ({layout})...")
        if on_start:
            on_start(s.name)
        try:
            out = s.output_path(folder)
            if s.n == 1:
                save_tiff(read_rgb(s.paths[0]), out)
            else:
                rows, cols = s.rows, s.cols
                if s.n == 2:   # side-by-side or stacked: measure it, like ImageJ does
                    rows, cols = stitch_native.auto_orient_pair(*map(stitch_native.load_rgb, s.paths))
                pos = tile_positions(s.n, rows, cols, s.order)
                mosaic = stitch_native.stitch_tiles(s.paths, pos, overlap, refine=refine,
                                                    verbose=False)
                save_tiff(mosaic, out)
            log(f"  done      {s.name}")
            ok = True
        except Exception as e:  # keep going with the next sample
            log(f"  FAILED    {s.name}: {e}")
            ok = False
        if on_done:
            on_done(s.name, ok)


# --------------------------------------------------------------------------- #
#  Settings (remembered between sessions)
# --------------------------------------------------------------------------- #
def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(data):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def clear_layouts(folder):
    shutil.rmtree(os.path.join(folder, OUT_DIR_NAME, LAYOUT_DIR_NAME), ignore_errors=True)
