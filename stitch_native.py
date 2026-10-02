#!/usr/bin/env python3
"""
stitch_native.py — native (no ImageJ) mosaic stitcher for tiled microscopy images.

Pure-Python replacement for the ImageJ Pairwise/Grid-stitching macro:
full-frame phase-correlation registration + feather (linear) blending.
Depends only on numpy + Pillow (no scipy, no scikit-image, no ImageJ).

Naming convention (shared with the GUI, see stitcher_core.py):
    {sample}_{index}.{ext}     e.g.  S24-1273_4X_1uL_01.tiff, 001_0.tiff
  - Everything before the LAST "_" is the sample name.
  - The trailing {index} orders the tiles of that sample.
  - All tiles sharing a sample name are stitched into one mosaic.

Layout is chosen from the tile count (override with --rows/--cols/--order):
    1  -> copy      2  -> auto (vertical or horizontal, decided from the data)
    4  -> 2x2       9  -> 3x3       16 -> 4x4
  other counts fall back to the nearest rectangle (cols = ceil(sqrt(n))).
Default order is "snake by rows": row 0 left->right, row 1 right->left, ...
For a 2x2 that is top-left -> top-right -> bottom-right -> bottom-left, which
matches the acquisition path.

Output: {folder}/Stitched/{group}_stitched.tif   (lossless TIFF, 8-bit RGB)
By default a group whose output already exists is skipped, so an interrupted
run resumes where it stopped. Pass --overwrite to redo them.

Usage:
    python stitch_native.py "C:/path/to/folder"
    python stitch_native.py "C:/path/to/folder" --group S24-1273_4X_1uL
    python stitch_native.py "C:/path/to/folder" --rows 2 --cols 3 --order "Row by row"
    python stitch_native.py "C:/path/to/folder" --overwrite
    python stitch_native.py "C:/path/to/folder" --overlap 0.20 --ext .tiff
"""
import os
import argparse

import numpy as np
from PIL import Image

import stitcher_core as core

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- #
#  Grid geometry (snake by rows)
# --------------------------------------------------------------------------- #
def choose_grid(n, rows=None, cols=None):
    if rows and cols:
        return rows, cols
    return core.guess_grid(n)


# --------------------------------------------------------------------------- #
#  Image IO
# --------------------------------------------------------------------------- #
def load_rgb(path):
    """Load as float32 RGB HxWx3 in [0,255]."""
    return core.read_rgb(path).astype(np.float32)


def to_gray(a):
    return a[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Registration — full-frame phase correlation (numpy FFT only)
# --------------------------------------------------------------------------- #
def _ncc_at(a, b, shift):
    """Normalised cross-correlation of the overlap after integer shift (dy,dx),
    where b[i,j] is compared with a[dy+i, dx+j]."""
    dy, dx = int(round(shift[0])), int(round(shift[1]))
    H, W = a.shape
    ay0, ay1 = max(0, dy), min(H, H + dy)
    ax0, ax1 = max(0, dx), min(W, W + dx)
    by0, by1 = max(0, -dy), min(H, H - dy)
    bx0, bx1 = max(0, -dx), min(W, W - dx)
    A = a[ay0:ay1, ax0:ax1].ravel()
    B = b[by0:by1, bx0:bx1].ravel()
    if A.size < 4096:
        return -1.0
    A = A - A.mean(); B = B - B.mean()
    denom = np.sqrt((A * A).sum() * (B * B).sum())
    return float((A * B).sum() / denom) if denom > 0 else -1.0


def _edge_taper(n, frac=0.05):
    """1-D Tukey-style taper: flat in the middle, cosine roll-off over `frac` of
    each edge. Unlike a full Hann window it keeps the tile edges, which is where
    the overlap with a neighbour lives."""
    w = np.ones(n, dtype=np.float32)
    k = max(1, int(n * frac))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k, dtype=np.float32))
    w[:k] = ramp
    w[-k:] = ramp[::-1]
    return w


def _phase_corr_peaks(gr, gm, n_peaks=5):
    """Top phase-correlation peaks (py, px) between two gray frames, strongest
    first. Like ImageJ's `check_peaks`, several are kept because the strongest is
    not always the true shift."""
    H, W = gr.shape
    win = _edge_taper(H)[:, None] * _edge_taper(W)[None, :]
    A = np.fft.rfft2((gr - gr.mean()) * win)
    B = np.fft.rfft2((gm - gm.mean()) * win)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-8
    corr = np.fft.irfft2(R, s=(H, W))
    peaks = []
    for flat in np.argsort(corr, axis=None)[::-1][:2000]:
        py, px = divmod(int(flat), W)
        if all(min(abs(py - qy), H - abs(py - qy)) > 3 or
               min(abs(px - qx), W - abs(px - qx)) > 3 for qy, qx in peaks):
            peaks.append((py, px))
            if len(peaks) == n_peaks:
                break
    return peaks, H, W


def _shift_candidates(ref, mov):
    """All plausible shifts (dy, dx) placing mov relative to ref, scored by NCC
    of the overlap, best first. The FFT only knows each peak modulo the image
    size, so all four wrapped sign variants of every peak are tried."""
    gr, gm = to_gray(ref), to_gray(mov)
    try:
        peaks, H, W = _phase_corr_peaks(gr, gm)
    except Exception:
        return []
    min_area = 0.02 * H * W
    cands = []
    for py, px in peaks:
        for dy in (py, py - H):
            for dx in (px, px - W):
                if (H - abs(dy)) * (W - abs(dx)) < min_area:
                    continue
                cands.append((_ncc_at(gr, gm, (dy, dx)), np.array([float(dy), float(dx)])))
    cands.sort(key=lambda t: -t[0])
    return cands


def _full_shift(ref, mov):
    """Best shift (dy, dx) of mov relative to ref over the whole frame, and its NCC.
    Works at any overlap fraction, including ~80% overlaps."""
    cands = _shift_candidates(ref, mov)
    if not cands:
        return None, 0.0
    ncc, off = cands[0]
    return off, ncc


def register_pair(ref, mov, axis, overlap, refine=True):
    """Shift (dy,dx) placing `mov` relative to `ref`. axis='x' -> mov is to the
    right; axis='y' -> mov is below. Takes the best-correlating candidate that is
    consistent with that direction; falls back to the nominal grid position when
    none correlates well."""
    Hr, Wr = ref.shape[:2]
    nominal = (np.array([0.0, (1.0 - overlap) * Wr]) if axis == "x"
               else np.array([(1.0 - overlap) * Hr, 0.0]))
    if not refine:
        return nominal, 0.0
    for ncc, off in _shift_candidates(ref, mov):
        if ncc < 0.20:
            break
        if axis == "x":
            ov = 1.0 - off[1] / Wr
            ok = 0.02 < ov < 0.97 and abs(off[0]) <= 0.5 * Hr
        else:
            ov = 1.0 - off[0] / Hr
            ok = 0.02 < ov < 0.97 and abs(off[1]) <= 0.5 * Wr
        if ok:
            return off, ncc
    return nominal, 0.0


def auto_orient_pair(t0, t1):
    """For a 2-tile group, decide vertical (2x1) vs horizontal (1x2) from the
    measured full-frame shift: the axis carrying the larger displacement is the
    stitch direction. Acquisition may stack top/bottom OR left/right."""
    off, conf = _full_shift(t0, t1)
    if off is None or conf < 0.20:
        return (1, 2)
    return (2, 1) if abs(off[0]) >= abs(off[1]) else (1, 2)


# --------------------------------------------------------------------------- #
#  Placement + feather blend
# --------------------------------------------------------------------------- #
def feather_weight(h, w, ramp_frac=0.5):
    ry = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32) + 1.0
    rx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32) + 1.0
    wy = np.clip(ry / max(1.0, ramp_frac * h), 0, 1)
    wx = np.clip(rx / max(1.0, ramp_frac * w), 0, 1)
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def stitch_group(paths, rows, cols, overlap, refine=True, verbose=True,
                 order=core.DEFAULT_ORDER):
    positions = core.tile_positions(len(paths), rows, cols, order)
    return stitch_tiles(paths, positions, overlap, refine=refine, verbose=verbose)


def stitch_tiles(paths, positions, overlap, refine=True, verbose=True):
    """Stitch tiles whose grid cell (row, col) is given in `positions`."""
    return stitch_arrays([load_rgb(p) for p in paths], positions, overlap,
                         refine=refine, verbose=verbose)


def stitch_arrays(tiles, positions, overlap, refine=True, verbose=True):
    """Stitch in-memory float32 RGB tiles (HxWx3, 0-255) placed at grid cells
    `positions`. Used directly by the GUI's quick preview on shrunken tiles."""
    n = len(tiles)
    at = {rc: i for i, rc in enumerate(positions)}

    off = {0: np.array([0.0, 0.0])}
    order = sorted(range(n), key=lambda i: (positions[i][0], positions[i][1]))
    for i in order:
        if i in off:
            continue
        r, c = positions[i]
        placed, conf = None, 0.0
        if (r, c - 1) in at and at[(r, c - 1)] in off:
            j = at[(r, c - 1)]
            d, conf = register_pair(tiles[j], tiles[i], "x", overlap, refine)
            placed = off[j] + d
        elif (r - 1, c) in at and at[(r - 1, c)] in off:
            j = at[(r - 1, c)]
            d, conf = register_pair(tiles[j], tiles[i], "y", overlap, refine)
            placed = off[j] + d
        if placed is None:
            Hr, Wr = tiles[0].shape[:2]
            placed = np.array([r * (1 - overlap) * Hr, c * (1 - overlap) * Wr])
        off[i] = placed
        if verbose:
            print(f"    tile {i} @ grid({r},{c})  offset=({placed[0]:7.1f},{placed[1]:7.1f})  conf={conf:.2f}")

    ys = [off[i][0] for i in range(n)]
    xs = [off[i][1] for i in range(n)]
    hh = [tiles[i].shape[0] for i in range(n)]
    ww = [tiles[i].shape[1] for i in range(n)]
    y0, x0 = min(ys), min(xs)
    Hc = int(np.ceil(max(off[i][0] + hh[i] for i in range(n)) - y0)) + 1
    Wc = int(np.ceil(max(off[i][1] + ww[i] for i in range(n)) - x0)) + 1

    acc = np.zeros((Hc, Wc, 3), dtype=np.float32)
    wsum = np.zeros((Hc, Wc), dtype=np.float32)
    for i in range(n):
        t = tiles[i]
        h, w = t.shape[:2]
        yy = int(round(off[i][0] - y0))
        xx = int(round(off[i][1] - x0))
        wt = feather_weight(h, w)
        acc[yy:yy + h, xx:xx + w, :] += t * wt[..., None]
        wsum[yy:yy + h, xx:xx + w] += wt
    nz = wsum > 0
    ws = np.where(nz, wsum, 1.0)
    out = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    for ch in range(3):
        chan = acc[..., ch] / ws
        out[..., ch] = np.clip(np.where(nz, chan, 0.0), 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def run(folder, overlap=0.20, refine=True, ext=None, rows=None, cols=None,
        only_group=None, overwrite=False, order=core.DEFAULT_ORDER):
    folder = os.path.abspath(folder)
    samples, _ = core.scan_folder(folder)
    if ext:
        for s in samples:
            s.tiles = [t for t in s.tiles if t[1].lower().endswith(ext.lower())]
        samples = [s for s in samples if s.tiles]
    if not samples:
        print(f"No matching image tiles in {folder}")
        return
    out_dir = os.path.join(folder, core.OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    for s in samples:
        group, paths, n = s.name, s.paths, s.n
        if only_group and group != only_group:
            continue
        out_path = s.output_path(folder)
        if s.is_done(folder) and not overwrite:
            print(f"[{group}] skip (exists)")
            continue
        try:
            if n == 1:
                core.save_tiff(core.read_rgb(paths[0]), out_path)
                print(f"[{group}] 1 tile -> copied")
                continue
            if n == 2 and not (rows and cols):
                r, c = auto_orient_pair(load_rgb(paths[0]), load_rgb(paths[1]))
            else:
                r, c = choose_grid(n, rows, cols)
            print(f"[{group}] {n} tiles -> {r}x{c}")
            mosaic = stitch_group(paths, r, c, overlap, refine=refine, order=order)
            # Lossless: identical pixels to the blend, 8-bit RGB (source bit depth).
            core.save_tiff(mosaic, out_path)
            print(f"    saved {group}_stitched.tif  ({mosaic.shape[1]}x{mosaic.shape[0]})")
        except Exception as e:
            print(f"[{group}] ERROR: {e}")


def main():
    ap = argparse.ArgumentParser(description="Native Python mosaic stitcher (no ImageJ).")
    ap.add_argument("folder")
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--ext", default=None)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--order", default=core.DEFAULT_ORDER, choices=list(core.ORDERS))
    ap.add_argument("--group", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-stitch groups whose output already exists")
    args = ap.parse_args()
    run(args.folder, overlap=args.overlap, refine=not args.no_refine, ext=args.ext,
        rows=args.rows, cols=args.cols, only_group=args.group, overwrite=args.overwrite,
        order=args.order)


if __name__ == "__main__":
    main()
