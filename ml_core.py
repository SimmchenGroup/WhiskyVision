"""
ml_core.py — machine learning on whisky webs (backend for the ML tab).

Unlike Identify (HOG + nearest neighbour on the whole image), this finds the
droplet by its coffee ring, unwraps it radially (r/R × angle, so size and
rotation stop mattering) and describes it with interpretable features, then
trains a classifier on them. Ported from the research code in ml/ (13 Nov 2026
first-pass study: 18 whiskies, within-slip balanced accuracy 0.89 for
scalar + radial features with logistic regression).

Needs only numpy + OpenCV, like identify_core. matplotlib is optional and only
used by the figure_* functions (figures_available() says whether it is there).

Main entry points
-----------------
One droplet (Gallery-style views of selected images):
    d = analyse(path)                      # full analysis of one stitched image (~2-4 s)
    view_raw(d), view_ring_map(d), view_rim(d), view_background(d),
    view_subtracted(d), view_normalised(d), view_unwrap(d, normalised=...)
                                           # -> uint8 RGB arrays for display
    normalisation_numbers(d)               # background medians, droplet means, gains
    d.qc_ok, d.qc_reason, d.scalars, d.profiles

Many droplets / one whole sample:
    rec = features(path)                   # cached, small: QC + features + thumbnails
    s = summarise(records, label)          # average of every (QC-passed) droplet in a sample

Machine learning:
    ds = build_dataset(folder, log, progress, should_stop)   # every stitched image, by whisky
    res = cross_validate(ds, feature_set="scalar+radial", normalised=False)
    model = train_model(ds, ..., cv_result=res)   # pass res: calibrates the "unknown" check
    save_model(model, path); load_model(path)
    preds = predict(model, paths)          # one result per droplet
    sp = predict_sample(model, paths)      # several droplets of ONE sample judged together
                                           # (much more reliable; also flags "not a trained whisky")
    res.sample_accuracy(n)                 # accuracy when n droplets are averaged

Figures (matplotlib; return PIL images, save with .save(path)):
    figure_pipeline(droplets), figure_normalisation(droplets),
    figure_profiles(items, names), figure_sample_average(summary, name),
    figure_confusion(result, names), figure_cv_comparison(results)
"""
import hashlib
import io
import json
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

import stitcher_core as core

# radii outside the imaged area are all-NaN rows of the unwrap, which is expected;
# numpy's warnings about them are noise (and warnings.catch_warnings isn't thread-safe)
warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom", category=RuntimeWarning)

VERSION = "ml-v4"   # v4 (17 Nov 2026): body-silhouette rim, asymmetry + texture features
#               # change if the features change, so cached ones aren't reused
                                # v3 (16 Nov 2026): rim re-traced at the outer edge of bright droplets
                                # v2 (14 Nov 2026): normalised features from the unclipped image
DOWNSAMPLE = 4                  # stitched images are analysed at 1/4 size (fixed, keeps scale)
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".whisky_stitcher_ml")
READERS = 6                     # images analysed at once

N_R, N_T, R_MAX = 128, 360, 1.15     # unwrap: radii, angles, how far past the rim
WORK = 320                           # working size (longest side) for the ring search
N_BINS = 32                          # radial profiles are binned to this many radii for ML
PROFILE_KEYS = ["mean_R", "mean_G", "mean_B", "std_L", "p95_L", "hue_ratio"]

# QC (chosen by eye on the Khalifa set, Nov 2026)
QC_MIN_SCORE = 0.12             # rim contrast
QC_MAX_OUTSIDE = 0.45           # share of the image well outside the droplet (ghosts, several drops)
QC_MIN_VALID = 0.60             # share of the unwrap inside the imaged area
# QC rule sets. "phone": photos through a phone-microscope attachment frame the droplet
# loosely (lots of background), so the "outside fraction" ghost rule is meaningless and
# rims are fainter (Nov 2026 phone set: 78% wrongly rejected by the microscope rules;
# median ring score 0.095 for V). Ghost stitches can't occur in single photos anyway.
QC_MODES = {
    "microscope": dict(min_score=QC_MIN_SCORE, max_outside=QC_MAX_OUTSIDE, min_valid=QC_MIN_VALID),
    # 0.15 checked by eye (18 Nov 2026): below it the phone "rims" are mostly debris or
    # empty space (V's phone rims are near-invisible), above it they sit on the droplet
    "phone": dict(min_score=0.15, max_outside=None, min_valid=QC_MIN_VALID),
}

SCALAR_NAMES = [
    "radius_px", "ovality", "ring_score", "ring_strength",
    "in_mean_R", "in_std_R", "in_mean_G", "in_std_G", "in_mean_B", "in_std_B",
    "in_p50", "in_p90", "in_p99", "particle_density", "particle_mean_int",
    "ring_peak_r", "ring_peak_int", "ring_contrast", "ring_width", "centre_int",
    "centre_to_mid", "azim_hetero", "ring_continuity", "hue_centre", "hue_rim", "grad_energy",
    # v4 (17 Nov 2026): what the CNN saw and the features didn't (lopsided deposits,
    # interior texture, concentric rings). Rotation-invariant: harmonic amplitudes.
    "asym1_centre", "asym2_centre", "hue_asym_centre",
    "asym1_interior", "asym2_interior", "hue_asym_interior",
    "asym1_rim", "asym2_rim", "hue_asym_rim",
    "tex_fine", "tex_mid", "tex_coarse", "tex_concentric"]
ASYM_ZONES = (("centre", 0.0, 0.35), ("interior", 0.35, 0.85), ("rim", 0.85, 1.10))
RAW_BRIGHTNESS = ("ring_score", "ring_strength")   # measured on the raw image in both modes
NORM_CONSTANT = ("in_mean_R", "in_mean_G", "in_mean_B")   # = the target by construction once normalised
FEATURE_SETS = ("scalar", "radial", "scalar+radial", "cnn", "cnn+scalar+radial", "nuisance")
CNN_SETS = ("cnn", "cnn+scalar+radial")     # need PyTorch (see cnn_available / add_embeddings)
NUISANCE = ["radius_px", "ring_score", "coverage", "outside_frac", "h", "w"]

_feature_cache = {}             # (path, mtime, size, downsample) -> Record
_cache_lock = threading.Lock()


def _cv2():
    import cv2
    return cv2


def figures_available():
    """True if matplotlib is installed (only the figure_* functions need it)."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
#  Reading
# --------------------------------------------------------------------------- #
def read_small(path, downsample=DOWNSAMPLE):
    """RGB uint8 image at 1/downsample size (INTER_AREA, as in the research cache)."""
    cv2 = _cv2()
    rgb = core.read_rgb(path)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, -1)
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = (rgb.astype(np.float32) / max(float(rgb.max()), 1) * 255).astype(np.uint8)
    if downsample > 1:
        h, w = rgb.shape[:2]
        rgb = cv2.resize(rgb, (w // downsample, h // downsample), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


# --------------------------------------------------------------------------- #
#  Droplet finding
# --------------------------------------------------------------------------- #
def ring_map(rgb):
    """Brightness with the smooth haze removed and big particles capped: the
    coffee ring and sharp features light up. Returns (map 0..1, imaged-area mask)."""
    cv2 = _cv2()
    g = rgb.max(axis=2).astype(np.float32)
    canvas = g > 0
    hz = cv2.GaussianBlur(g, (0, 0), max(g.shape) / 40)
    t = np.clip(g - hz, 0, None)
    t = cv2.GaussianBlur(t, (0, 0), max(g.shape) / 400 + 0.5)
    hi = np.percentile(t[canvas], 99.0) + 1e-6 if canvas.any() else 1.0
    return np.clip(t / hi, 0, 1), canvas


def _conv(img, kernel):
    """'same'-size convolution with zero borders (kernels here are symmetric)."""
    cv2 = _cv2()
    return cv2.filter2D(img, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)


def _rim_polygon(cx, cy, rim, scale=1.0):
    th = np.linspace(0, 2 * np.pi, len(rim), endpoint=False)
    return np.stack([cx + scale * rim * np.cos(th), cy + scale * rim * np.sin(th)], 1).astype(np.int32)


BODY_MIN_CONTRAST = 25.0   # bright-body silhouette only for clearly bright droplets (0-255)
BODY_MIN_SOLIDITY = 0.93   # one convex blob; 2×2 ghost stitches and merged droplets score lower
BODY_GROW = 1.08           # body silhouette this much bigger than the ring fit -> use the body


def _ellipse_radii(cx, cy, ex, ey, ea, eb, eang):
    """Radius of an ellipse (centre ex, ey; full axes ea, eb; angle eang in degrees)
    seen from (cx, cy) at each of N_T angles."""
    th = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
    phi = np.deg2rad(eang)
    ts = np.linspace(0, 2 * np.pi, 720)
    xs = ex + ea / 2 * np.cos(ts) * np.cos(phi) - eb / 2 * np.sin(ts) * np.sin(phi)
    ys = ey + ea / 2 * np.cos(ts) * np.sin(phi) + eb / 2 * np.sin(ts) * np.cos(phi)
    ang = np.mod(np.arctan2(ys - cy, xs - cx), 2 * np.pi)
    rad = np.hypot(xs - cx, ys - cy)
    o = np.argsort(ang)
    return np.interp(th, ang[o], rad[o], period=2 * np.pi)


def body_silhouette(rgb, canvas):
    """The bright droplet body as one blob, for bright droplets whose contact line is
    faint next to a bright halo (Y, W, V, M...). Smoothed max-channel brightness is
    thresholded at background + 25% of (body − background), the largest blob kept and
    an ellipse fitted to its outline. Returns a dict (cx, cy, rim[N_T], solidity, fill,
    contrast, thr, smooth) or None for dim droplets / no blob."""
    cv2 = _cv2()
    g = rgb.max(2).astype(np.float32)
    sig = max(2.0, max(g.shape) / 150)
    gb = cv2.GaussianBlur(g, (0, 0), sig)
    v = gb[canvas]
    if v.size < 100:
        return None
    bg, hi = np.percentile(v, 5), np.percentile(v, 95)
    if hi - bg < BODY_MIN_CONTRAST:
        return None
    thr = bg + 0.25 * (hi - bg)
    m = ((gb > thr) & canvas).astype(np.uint8)
    k = int(2 * sig) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 20:
        return None
    area = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    (ex, ey), (ea, eb), eang = cv2.fitEllipse(c)
    fill = area / max(1.0, np.pi * ea * eb / 4)
    return dict(cx=float(ex), cy=float(ey), rim=_ellipse_radii(ex, ey, ex, ey, ea, eb, eang),
                solidity=float(area / max(1.0, hull)), fill=float(fill), contrast=float(hi - bg),
                thr=float(thr), smooth=gb, area=float(area))


EXPAND_MIN = 1.10      # outer bright edge this far past the fitted rim -> re-trace there
EXPAND_MAX = 1.45      # never grow the rim more than this (larger = a neighbour or ghost)
EXPAND_AGREE = 0.06    # new rim must match the bright region's edge to within 6% (median)


def _trace_rim(t_full, cx, cy, R, window=(0.8, 1.2)):
    """Per-angle rim trace (brightest ring-map radius in window × R) and a robust ellipse
    through it (3 rounds of outlier rejection). Returns (rim radius per angle, ring
    profile, angles with data, ellipse centre or None)."""
    cv2 = _cv2()
    th = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
    rr = np.linspace(window[0], window[1], 81)
    X = (cx + R * rr[:, None] * np.cos(th)[None]).astype(np.float32)
    Y = (cy + R * rr[:, None] * np.sin(th)[None]).astype(np.float32)
    prof = cv2.remap(t_full, X, Y, cv2.INTER_LINEAR, borderValue=np.nan)
    ok = ~np.all(np.isnan(prof), 0)
    rim = np.full(N_T, 1.0)
    if ok.any():
        rim[ok] = rr[np.nanargmax(np.nan_to_num(prof[:, ok], nan=-1), 0)]
    strength = np.nanmax(np.nan_to_num(prof, nan=-1), 0)
    use = ok & (strength > np.percentile(strength[ok], 30)) if ok.sum() > 20 else ok
    rim_px = R * np.ones(N_T)
    centre = None
    for _ in range(3):
        if use.sum() < 12:
            break
        P = np.stack([cx + R * rim[use] * np.cos(th[use]), cy + R * rim[use] * np.sin(th[use])], 1)
        (ex, ey), (ea, eb), eang = cv2.fitEllipse(P.astype(np.float32))
        phi = np.deg2rad(eang)
        ts = np.linspace(0, 2 * np.pi, 720)
        exs = ex + ea / 2 * np.cos(ts) * np.cos(phi) - eb / 2 * np.sin(ts) * np.sin(phi)
        eys = ey + ea / 2 * np.cos(ts) * np.sin(phi) + eb / 2 * np.sin(ts) * np.cos(phi)
        ang_e = np.mod(np.arctan2(eys - cy, exs - cx), 2 * np.pi)
        rad_e = np.hypot(exs - cx, eys - cy)
        o = np.argsort(ang_e)
        rim_px = np.interp(th, ang_e[o], rad_e[o], period=2 * np.pi)
        centre = (float(ex), float(ey))
        use = use & (np.abs(R * rim - rim_px) / R < 0.06)
    if not (0.6 * R < rim_px.mean() < 1.4 * R) or rim_px.min() < 0.5 * R:
        rim_px, centre = R * np.ones(N_T), None          # fall back to the circle
    return rim_px, prof, ok, centre


def _outer_edge(rgb, cx, cy, rim_px, canvas, min_contrast=12.0, frac=0.3):
    """How far past the fitted rim (as a multiple of it) the bright droplet body ends,
    or None if it can't be judged (dim droplet, little imaged area outside).

    Smoothed max-channel brightness is sampled along each angle from 0.9 to 1.8 × rim.
    background = median at 1.6-1.8 ×; inside = median at 0.90-0.95 ×. The edge on each
    angle is where brightness first falls below background + frac × (inside − background);
    the median over angles is returned. Dim webs (inside barely above background) return
    None, so their ring-based rim is kept."""
    cv2 = _cv2()
    g = cv2.GaussianBlur(rgb.max(2).astype(np.float32), (0, 0), max(1.0, float(np.mean(rim_px)) / 60))
    g[~canvas] = np.nan
    th = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
    rr = np.linspace(0.9, 1.8, 91)
    X = (cx + rr[:, None] * rim_px[None] * np.cos(th)[None]).astype(np.float32)
    Y = (cy + rr[:, None] * rim_px[None] * np.sin(th)[None]).astype(np.float32)
    P = cv2.remap(g, X, Y, cv2.INTER_LINEAR, borderValue=np.nan)
    with _quiet():
        bg = np.nanmedian(P[rr >= 1.6])
        inside = np.nanmedian(P[(rr >= 0.9) & (rr <= 0.95)])
    if not np.isfinite(bg) or not np.isfinite(inside) or inside - bg < min_contrast:
        return None
    thr = bg + frac * (inside - bg)
    edges = np.full(N_T, np.nan)
    for j in range(N_T):
        col = P[:, j]
        if np.isnan(col[rr >= 1.6]).all():
            continue
        below = np.where((col < thr) & (rr >= 1.0))[0]
        if len(below):
            edges[j] = rr[below[0]]
    if np.isfinite(edges).sum() < N_T // 4:
        return None
    return float(np.nanmedian(edges)), edges * rim_px      # (factor, edge radius per angle, px)


def segment(rgb, r_lo=0.18, r_hi=0.75, n_radii=36):
    """Find the main droplet by its rim.

    1. annulus matched filter over n_radii radii (ring brighter than the band just
       inside it), weighted by how much of the circle was imaged and penalised for
       leaving imaged area unexplained -> best circle (cx, cy, R);
    2. per-angle rim trace (brightest radius in 0.8-1.2 R) and a robust ellipse
       through it -> rim radius at each of N_T angles.
    Returns a dict (cx, cy, R, rim[N_T], mask, score, coverage, ...) or None."""
    cv2 = _cv2()
    t_full, canvas_full = ring_map(rgb)
    H, W = t_full.shape
    sc = WORK / max(H, W)
    t = cv2.resize(t_full, (max(1, int(W * sc)), max(1, int(H * sc))), interpolation=cv2.INTER_AREA)
    cv = cv2.resize(canvas_full.astype(np.float32), t.shape[::-1]) > 0.5
    h, w = t.shape
    pad = int(0.35 * max(h, w))
    tp = np.pad(t, pad).astype(np.float32)
    cvp = np.pad(cv.astype(np.float32), pad)
    best = (-np.inf, None)
    side = max(h, w)
    cv_area = max(1.0, float(cv.sum()))
    Yg, Xg = np.mgrid[0:h, 0:w]
    for R in np.geomspace(r_lo * side, r_hi * side, n_radii):
        k = int(np.ceil(R * 1.1))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        d = np.hypot(xx, yy)
        ring = (np.abs(d - R) < max(1.5, 0.03 * R)).astype(np.float32)
        inner = ((d > 0.75 * R) & (d < 0.9 * R)).astype(np.float32)
        s_ring, n_ring = _conv(tp, ring), _conv(cvp, ring)
        s_in, n_in = _conv(tp, inner), _conv(cvp, inner)
        cover = n_ring / ring.sum()
        score = (s_ring / np.maximum(n_ring, 1) - s_in / np.maximum(n_in, 1))
        score = score * np.sqrt(np.clip(cover, 0, 1)) * (cover > 0.35)
        flat = score.ravel()
        for i in np.argpartition(-flat, 5)[:5]:
            if flat[i] <= 0:
                continue
            yy0, xx0 = np.unravel_index(i, score.shape)
            x0, y0 = xx0 - pad, yy0 - pad
            outside = (cv & ((Xg - x0) ** 2 + (Yg - y0) ** 2 > (1.2 * R) ** 2)).sum() / cv_area
            s = flat[i] * (1 - 0.8 * outside)
            if s > best[0]:
                best = (s, (x0, y0, R, float(cover.flat[i]), float(flat[i])))
    if best[1] is None or best[0] <= 0:
        return None
    cx, cy, R, cover, raw = best[1]
    cx, cy, R = cx / sc, cy / sc, R / sc

    th = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
    rim_px, prof, ok, centre = _trace_rim(t_full, cx, cy, R)

    # Bright droplets (Y, W, V, M...): a bright centre inside an orange halo inside a
    # faint contact line. The ring filter can lock onto the centre/halo boundary instead
    # of the droplet edge (found 16 Nov 2026: ~25% of Y and W). If the bright region
    # clearly carries on past the fitted rim, re-trace the rim at its outer edge.
    oe = _outer_edge(rgb, cx, cy, rim_px, canvas_full)
    expanded = 1.0
    if oe is not None and EXPAND_MIN < oe[0] < EXPAND_MAX:
        edge, edge_px = oe
        R2 = float(np.mean(rim_px)) * edge
        rim2, prof2, ok2, centre2 = _trace_rim(t_full, cx, cy, R2, (0.85, 1.15))
        if centre2 is not None:                      # re-centre on the new ellipse, trace again
            cx2, cy2 = centre2
            rim2, prof2, ok2, _ = _trace_rim(t_full, cx2, cy2, float(np.mean(rim2)), (0.85, 1.15))
            # accept only if the new rim follows the bright region's own edge (measured
            # from the old centre) on most angles: rejects ghost copies / neighbours
            th_ = np.linspace(0, 2 * np.pi, N_T, endpoint=False)
            xs, ys = cx2 + rim2 * np.cos(th_), cy2 + rim2 * np.sin(th_)
            r_old = np.hypot(xs - cx, ys - cy)
            a_old = np.mod(np.arctan2(ys - cy, xs - cx), 2 * np.pi)
            o = np.argsort(a_old)
            rim2_from_old = np.interp(th_, a_old[o], r_old[o], period=2 * np.pi)
            ok_e = np.isfinite(edge_px)
            agree = np.median(np.abs(rim2_from_old[ok_e] / edge_px[ok_e] - 1)) if ok_e.any() else 1.0
            if 1.05 < np.mean(rim2) / np.mean(rim_px) < EXPAND_MAX and agree < EXPAND_AGREE:
                expanded = float(np.mean(rim2) / np.mean(rim_px))
                cx, cy, R, rim_px, prof, ok = cx2, cy2, float(np.mean(rim2)), rim2, prof2, ok2

    # Bright droplets whose ring fit sits on an inner ring (Y's concentric rings, found
    # 16 Nov 2026: ~24% of Y rejected as "ghosting" though every one is a single droplet):
    # if the bright body is one clean ellipse clearly bigger than the ring fit, use it.
    body = body_silhouette(rgb, canvas_full)
    body_ok = bool(body and body["solidity"] >= BODY_MIN_SOLIDITY and 0.9 <= body["fill"] <= 1.1)
    used_body = False
    ring_mean = float(np.mean(rim_px))
    if body_ok and np.mean(body["rim"]) > BODY_GROW * ring_mean:
        # snap to the contact line if the ring map has one close to the body edge
        bcx, bcy = body["cx"], body["cy"]
        rim_b, prof_b, ok_b, _ = _trace_rim(t_full, bcx, bcy, float(np.mean(body["rim"])), (0.92, 1.12))
        snap = np.median(np.abs(rim_b / body["rim"] - 1)) < 0.05
        cx, cy = bcx, bcy
        rim_px = rim_b if snap else body["rim"]
        R, prof, ok = float(np.mean(rim_px)), prof_b, ok_b
        expanded, used_body = float(np.mean(rim_px) / ring_mean), True

    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [_rim_polygon(cx, cy, rim_px)], 1)
    mask &= canvas_full.astype(np.uint8)
    with _quiet():
        ring_strength = float(np.nanmean(np.nanmax(prof[:, ok], 0))) if ok.any() else 0.0
    big = np.zeros((H, W), np.uint8)
    cv2.fillPoly(big, [_rim_polygon(cx, cy, rim_px, 1.2)], 1)
    far = canvas_full & ~big.astype(bool)
    outside = float(far.sum() / max(1, canvas_full.sum()))
    outside_ring = float(t_full[far].mean() / (ring_strength + 1e-6)) if far.any() else 0.0
    # bright material well outside the rim (another droplet, a ghost copy), as a share of
    # the droplet's own area. A single droplet that fills the frame has ~0 here even when
    # outside_frac is large; a 2×2 ghost stitch or a neighbouring droplet has a lot.
    if body is not None:
        outside_bright = float(((body["smooth"] > body["thr"]) & far).sum() / max(1.0, mask.sum()))
    else:
        outside_bright = float((t_full[far] > 0.5).sum() / max(1.0, mask.sum())) if far.any() else 0.0
    body_agrees = bool(body_ok and abs(np.mean(body["rim"]) / np.mean(rim_px) - 1) < 0.15)
    return dict(cx=float(cx), cy=float(cy), R=float(R), rim=rim_px, mask=mask, score=raw,
                coverage=cover, ring_strength=ring_strength, outside_frac=outside, expanded=expanded,
                outside_bright=outside_bright, body_ok=body_ok, body_agrees=body_agrees, used_body=used_body,
                outside_ring=outside_ring, ovality=float((rim_px / R).std() / (rim_px / R).mean()))


# --------------------------------------------------------------------------- #
#  Normalisation, unwrap, features
# --------------------------------------------------------------------------- #
def normalise(rgb, seg, target=0.15):
    """Strict per-image normalisation: subtract each channel's background (median of
    imaged pixels beyond 1.25 R), then divide by the channel's mean inside the
    droplet. Removes exposure, lamp brightness and lamp colour.
    Returns (uint8 RGB image, info dict: bg, mean_raw, mean_sub, gain, bg_mask)."""
    cv2 = _cv2()
    x = rgb.astype(np.float32) / 255
    canvas = rgb.max(2) > 0
    big = np.zeros(canvas.shape, np.uint8)
    cv2.fillPoly(big, [_rim_polygon(seg["cx"], seg["cy"], seg["rim"], 1.25)], 1)
    bgmask = canvas & ~big.astype(bool)
    m = seg["mask"].astype(bool)
    y = np.zeros_like(x)
    bg, mean_raw, mean_sub, gain = [], [], [], []
    for c in range(3):
        ch = x[..., c]
        b = float(np.median(ch[bgmask])) if bgmask.sum() > 500 else float(np.percentile(ch[canvas], 5))
        v = np.clip(ch - b, 0, None)
        gm = float(v[m].mean()) if m.any() else float(v[canvas].mean())
        y[..., c] = v / (gm + 1e-4) * target
        bg.append(b)
        mean_raw.append(float(ch[m].mean()) if m.any() else 0.0)
        mean_sub.append(gm)
        gain.append(target / (gm + 1e-4))
    y[~canvas] = 0
    # Features are measured on the UNCLIPPED image (info["float"], 0-255 scale, may
    # exceed 255). Clipping at 255 would let brightness back in: a dim droplet needs a
    # big gain, saturates more, and its clipped mean falls below the target, so the
    # amount of clipping encodes how dim the raw droplet was (seen 14 Nov 2026, when
    # in_mean_G/R topped the "normalised" model's weights).
    yf = y * 255
    yf[canvas & (yf.max(2) <= 0)] = 1e-3            # keep imaged pixels non-zero (canvas test)
    yd = np.clip(yf, 0, 255).astype(np.uint8)       # for display only
    yd[canvas & (yd.max(2) == 0)] = 1
    info = dict(bg=np.array(bg), mean_raw=np.array(mean_raw), mean_sub=np.array(mean_sub),
                gain=np.array(gain), bg_mask=bgmask, bg_fraction=float(bgmask.sum() / max(1, canvas.sum())),
                float=yf.astype(np.float32), clipped=float((yf[m] > 255).any(-1).mean()) if m.any() else 0.0)
    return yd, info


def tone_map(x, mean_out=0.375):
    """Soft, never-saturating brightness curve for normalised images shown to the CNN:
    x / (x + k) on the 0-1 scale, with k set so the droplet mean (0.15) lands at
    mean_out (what ×2.5 gives raw images). Bright rims compress instead of clipping,
    so the amount of saturation can't encode the original exposure."""
    k = 0.15 * (1 - mean_out) / mean_out
    x = np.asarray(x, np.float32) / 255
    return (np.clip(x / (x + k), 0, 1) * 255).astype(np.uint8)


def crop(rgb, seg, size=256):
    """Square droplet-centred crop, side 2.4 × mean rim radius, zero-padded."""
    cv2 = _cv2()
    half = max(2, int(1.2 * float(np.mean(seg["rim"]))))
    big = cv2.copyMakeBorder(rgb, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    x0, y0 = int(seg["cx"]), int(seg["cy"])
    c = big[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def unwrap(rgb, seg, n_r=N_R, n_t=N_T, r_max=R_MAX):
    """Radial unwrap -> (n_r, n_t, 3) float32 in 0..1, row 0 = centre, r/R = 1 at the
    traced rim at every angle. Outside the imaged area is NaN."""
    cv2 = _cv2()
    rim = np.interp(np.linspace(0, N_T, n_t, endpoint=False), np.arange(N_T + 1),
                    np.r_[seg["rim"], seg["rim"][0]])
    r = np.linspace(0, r_max, n_r)[:, None]
    t = np.linspace(0, 2 * np.pi, n_t, endpoint=False)[None, :]
    x = (seg["cx"] + r * rim[None] * np.cos(t)).astype(np.float32)
    y = (seg["cy"] + r * rim[None] * np.sin(t)).astype(np.float32)
    img = rgb.astype(np.float32) / 255
    out = np.stack([cv2.remap(img[..., c], x, y, cv2.INTER_LINEAR, borderValue=np.nan)
                    for c in range(3)], -1)
    valid = cv2.remap((rgb.max(2) > 0).astype(np.float32), x, y, cv2.INTER_NEAREST, borderValue=0)
    out[valid < 0.5] = np.nan
    return out


def radial_profiles(polar):
    """The radial feature axis: statistics over angle at each radius (length n_r each).
    mean_R/G/B, mean_L: ring structure; std_L: angular heterogeneity (spokes,
    patchiness); p95_L: bright-particle envelope; hue_ratio: (R-B)/(R+B)."""
    L = polar.mean(-1)
    with np.errstate(all="ignore"), _quiet():
        out = {f"mean_{c}": np.nanmean(polar[..., i], 1) for i, c in enumerate("RGB")}
        out["mean_L"] = np.nanmean(L, 1)
        out["std_L"] = np.nanstd(L, 1)
        out["p95_L"] = np.nanpercentile(L, 95, axis=1)
        out["hue_ratio"] = (out["mean_R"] - out["mean_B"]) / (out["mean_R"] + out["mean_B"] + 1e-6)
    return {k: np.nan_to_num(v).astype(np.float32) for k, v in out.items()}


class _quiet:
    """Silence numpy's 'All-NaN slice' warnings inside a block."""
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def scalar_features(rgb, seg, polar, prof):
    """26 interpretable features of one droplet (names in SCALAR_NAMES)."""
    cv2 = _cv2()
    g = rgb.max(2).astype(np.float32) / 255
    m = seg["mask"].astype(bool)
    f = {"radius_px": float(np.mean(seg["rim"])), "ovality": seg["ovality"],
         "ring_score": seg["score"], "ring_strength": seg["ring_strength"]}
    inside = rgb[m].astype(np.float32) / 255 if m.any() else np.zeros((1, 3), np.float32)
    for i, c in enumerate("RGB"):
        f[f"in_mean_{c}"] = inside[:, i].mean()
        f[f"in_std_{c}"] = inside[:, i].std()
    gi = g[m] if m.any() else np.zeros(1, np.float32)
    for q in (50, 90, 99):
        f[f"in_p{q}"] = np.percentile(gi, q)
    tophat = g - cv2.GaussianBlur(g, (0, 0), 6)
    peaks = (tophat > 0.08) & (g == cv2.dilate(g, np.ones((3, 3), np.uint8))) & m
    f["particle_density"] = peaks.sum() / max(1, m.sum())
    f["particle_mean_int"] = g[peaks].mean() if peaks.any() else 0.0
    rr = np.linspace(0, R_MAX, len(prof["mean_L"]))
    L = prof["mean_L"]
    rim_zone, centre = (rr > 0.8) & (rr < 1.1), rr < 0.5
    ipk = int(np.argmax(np.where(rim_zone, L, -1)))
    f["ring_peak_r"], f["ring_peak_int"] = rr[ipk], L[ipk]
    f["ring_contrast"] = L[ipk] / (L[centre].mean() + 1e-3)
    half = L[ipk] - 0.5 * (L[ipk] - L[(rr > 0.5) & (rr < 0.8)].mean())
    above = np.where(rim_zone & (L > half))[0]
    f["ring_width"] = (rr[above[-1]] - rr[above[0]]) if len(above) else 0.0
    f["centre_int"] = L[centre].mean()
    f["centre_to_mid"] = L[rr < 0.2].mean() / (L[(rr > 0.4) & (rr < 0.7)].mean() + 1e-3)
    f["azim_hetero"] = prof["std_L"][rr < 1].mean()
    with _quiet():
        rim_ang = np.nanmax(polar[(rr > 0.9) & (rr < 1.08)].mean(-1), 0)
        f["ring_continuity"] = np.nanmean(rim_ang > np.nanmedian(L[(rr > 0.3) & (rr < 0.8)]) * 1.5)
    f["hue_centre"] = prof["hue_ratio"][centre].mean()
    f["hue_rim"] = prof["hue_ratio"][rim_zone].mean()
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    f["grad_energy"] = np.sqrt(gx ** 2 + gy ** 2)[m].mean() if m.any() else 0.0
    f.update(_asymmetry_texture(rgb, seg, polar, m))
    return {k: float(np.nan_to_num(f[k])) for k in SCALAR_NAMES}


def _harmonics(a):
    """Amplitudes of the 1st and 2nd angular harmonics of a (N_T,) profile, NaN-filled by
    interpolation; None if under 70% of angles were imaged."""
    ok = np.isfinite(a)
    if ok.mean() < 0.7:
        return None
    x = np.arange(len(a))
    a = np.interp(x, x[ok], a[ok], period=len(a))
    F = np.fft.rfft(a - a.mean())
    return 2 * np.abs(F[1]) / len(a), 2 * np.abs(F[2]) / len(a), a.mean()


def _asymmetry_texture(img, seg, polar, mask):
    """Angular asymmetry per zone (1st/2nd harmonic of brightness ÷ mean, 1st harmonic of
    colour) and interior texture (band-pass energies ÷ mean brightness, and the share of
    gradient energy pointing radially = concentric rings)."""
    cv2 = _cv2()
    f = {}
    rr = np.linspace(0, R_MAX, polar.shape[0])
    L = polar.mean(-1)
    with _quiet():
        hue = (polar[..., 0] - polar[..., 2]) / (polar[..., 0] + polar[..., 2] + 1e-3)
        for z, lo, hi in ASYM_ZONES:
            rows = (rr >= lo) & (rr < hi)
            hL = _harmonics(np.nanmean(L[rows], 0))
            hH = _harmonics(np.nanmean(hue[rows], 0))
            f[f"asym1_{z}"] = hL[0] / (hL[2] + 1e-3) if hL else 0.0
            f[f"asym2_{z}"] = hL[1] / (hL[2] + 1e-3) if hL else 0.0
            f[f"hue_asym_{z}"] = hH[0] if hH else 0.0
    g = np.asarray(img, np.float32).mean(2)
    g = g / 255 if g.max() > 1.5 else g
    H, W = g.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - seg["cx"], yy - seg["cy"]
    ang = np.arctan2(dy, dx)
    rim = np.interp(np.mod(ang, 2 * np.pi), np.linspace(0, 2 * np.pi, N_T, endpoint=False),
                    seg["rim"], period=2 * np.pi)
    region = mask.astype(bool) & (np.hypot(dx, dy) < 0.85 * rim)
    if region.sum() < 50:
        f.update(tex_fine=0.0, tex_mid=0.0, tex_coarse=0.0, tex_concentric=0.0)
        return f
    sc = max(1.0, float(np.mean(seg["rim"])) / 100)          # scales follow droplet size
    b = [cv2.GaussianBlur(g, (0, 0), s * sc) for s in (1, 2, 5, 12)]
    mu = g[region].mean() + 1e-3
    for name, (u, v) in zip(("tex_fine", "tex_mid", "tex_coarse"), ((0, 1), (1, 2), (2, 3))):
        f[name] = np.abs(b[u] - b[v])[region].mean() / mu
    gx, gy = cv2.Sobel(b[1], cv2.CV_32F, 1, 0), cv2.Sobel(b[1], cv2.CV_32F, 0, 1)
    gr = gx * np.cos(ang) + gy * np.sin(ang)
    gt = -gx * np.sin(ang) + gy * np.cos(ang)
    er, et = (gr ** 2)[region].sum(), (gt ** 2)[region].sum()
    f["tex_concentric"] = er / (er + et + 1e-12)
    return f


def radial_vector(prof):
    """The radial feature axis as one ML vector: 6 profiles × N_BINS radii."""
    return np.concatenate([prof[k].reshape(N_BINS, -1).mean(1) for k in PROFILE_KEYS]).astype(np.float32)


def radial_vector_names():
    return [f"{k}@r{(i + 0.5) / N_BINS * R_MAX:.2f}" for k in PROFILE_KEYS for i in range(N_BINS)]


def _qc(seg, polar_valid, mode="microscope"):
    if seg is None:
        return False, "no droplet rim found"
    return _qc_rules(seg["score"], seg["outside_frac"], polar_valid, mode)


def _qc_rules(score, outside_frac, polar_valid, mode="microscope"):
    q = QC_MODES[mode]
    if score < q["min_score"]:
        return False, f"faint or no rim (ring score {score:.2f} < {q['min_score']})"
    if q["max_outside"] is not None and outside_frac > q["max_outside"]:
        return False, (f"several droplets or ghosting ({outside_frac:.0%} of the image "
                       f"outside the droplet)")
    if polar_valid < q["min_valid"]:
        return False, f"droplet mostly outside the image ({polar_valid:.0%} imaged)"
    return True, ""


def apply_qc(records, mode="microscope"):
    """Copies of the records with qc_ok / qc_reason re-judged under another QC rule set
    (QC_MODES: "microscope" or "phone"), from the measurements each record keeps. The
    cached originals are not changed."""
    import dataclasses
    out = []
    for r in records:
        if r is None or r.scalars is None or "ring_score" not in r.qc:
            out.append(r)
            continue
        ok, why = _qc_rules(r.qc["ring_score"], r.qc["outside_frac"], r.qc["polar_valid"], mode)
        out.append(dataclasses.replace(r, qc_ok=ok, qc_reason=why))
    return out


# --------------------------------------------------------------------------- #
#  One droplet, fully analysed (for the views)
# --------------------------------------------------------------------------- #
@dataclass
class Droplet:
    path: str
    label: str
    rgb: np.ndarray                          # 1/DOWNSAMPLE-size RGB
    seg: dict = None                         # None if no rim was found
    qc_ok: bool = False
    qc_reason: str = ""
    norm: np.ndarray = None                  # normalised RGB
    norm_info: dict = None
    polar: np.ndarray = None                 # N_R × N_T × 3, 0..1, NaN outside image
    polar_norm: np.ndarray = None
    profiles: dict = None                    # raw radial profiles (N_R each)
    profiles_norm: dict = None
    scalars: dict = None                     # raw scalar features
    scalars_norm: dict = None
    ring: np.ndarray = None                  # ring map 0..1

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.path))[0].replace("_stitched", "")


def label_of(path):
    """Whisky letter of a stitched image: the folder above "Stitched" (or the folder it is in)."""
    parent = os.path.dirname(path)
    if os.path.basename(parent) == core.OUT_DIR_NAME:
        parent = os.path.dirname(parent)
    return os.path.basename(parent)


def analyse(path, label=None, downsample=DOWNSAMPLE, rgb=None):
    """Full analysis of one stitched image (all images and numbers kept)."""
    rgb = read_small(path, downsample) if rgb is None else rgb
    d = Droplet(path=path, label=label or label_of(path), rgb=rgb)
    d.ring, _ = ring_map(rgb)
    seg = segment(rgb)
    if seg is None:
        d.qc_ok, d.qc_reason = _qc(None, 0)
        return d
    d.seg = seg
    d.polar = unwrap(rgb, seg)
    d.qc_ok, d.qc_reason = _qc(seg, float(np.isfinite(d.polar[..., 0]).mean()))
    d.profiles = radial_profiles(d.polar)
    d.scalars = scalar_features(rgb, seg, d.polar, d.profiles)
    d.norm, d.norm_info = normalise(rgb, seg)
    nf = d.norm_info["float"]                       # unclipped: see normalise()
    d.polar_norm = unwrap(nf, seg)
    d.profiles_norm = radial_profiles(d.polar_norm)
    d.scalars_norm = scalar_features(nf, seg, d.polar_norm, d.profiles_norm)
    return d


# --------------------------------------------------------------------------- #
#  Views (uint8 RGB arrays for the GUI)
# --------------------------------------------------------------------------- #
DISPLAY_KNEE = 0.8        # above this the display rolls off smoothly instead of clipping


def soft_clip(y, knee=DISPLAY_KNEE):
    """Map 0..inf to 0..1: linear up to `knee`, then an exponential shoulder that
    approaches 1 without ever flattening, so bright detail stays visible."""
    y = np.maximum(np.nan_to_num(np.asarray(y, np.float32)), 0)
    over = y > knee
    y = y.copy()
    y[over] = knee + (1 - knee) * (1 - np.exp(-(y[over] - knee) / (1 - knee)))
    return y


def display_gain(x, max_gain=2.5, target=0.7, pct=90):
    """Per-image display gain: brighten by up to `max_gain` (what dim webs need), but
    only so far that the `pct` percentile of the imaged pixels reaches `target`; the
    brightest ~10% (rims, flares, particles) then roll off softly (soft_clip). Bright
    droplets (M, O, V, W, Y) get ~×1.1-1.4 instead of being blown out, H ~×2.4, dim
    webs (A, K, S) the full ×2.5. Never below ×1, so an image is never shown darker
    than it really is. (Chosen by eye 15 Nov 2026 against p99.5/97/95; p99.5 left H
    and K too dark because a few rim pixels set the gain.)"""
    x = x.astype(np.float32) / 255 if x.dtype == np.uint8 else np.nan_to_num(np.asarray(x, np.float32))
    v = x.max(-1) if x.ndim == 3 else x
    v = v[v > 0]
    if not v.size:
        return float(max_gain)
    p = float(np.percentile(v, pct))
    return float(np.clip(target / max(p, 1e-6), 1.0, max_gain))


def _show(x, gain="auto", max_gain=2.5):
    """uint8 display image. gain: a number (fixed, e.g. to compare brightness between
    images) or "auto" (display_gain, up to max_gain). Either way the top end rolls off
    softly (soft_clip) instead of hard-clipping at white."""
    x = x.astype(np.float32) / 255 if x.dtype == np.uint8 else np.nan_to_num(np.asarray(x, np.float32))
    g = display_gain(x, max_gain) if gain == "auto" else float(gain)
    return (soft_clip(x * g) * 255).astype(np.uint8)


def display(img, gain="auto", max_gain=2.5):
    """Public helper for the GUI: any uint8 RGB array (e.g. SampleSummary.mean_droplet)
    -> display image with auto gain and soft roll-off. Use max_gain=2.5 for raw
    images, 1.6 for normalised ones."""
    return _show(img, gain, max_gain)


def _draw_rim(img, seg, scale=1.0, colour=(0, 255, 0), dashed=False):
    cv2 = _cv2()
    pts = _rim_polygon(seg["cx"], seg["cy"], seg["rim"], scale)
    t = max(1, img.shape[1] // 300)
    if dashed:
        for i in range(0, len(pts), 8):
            cv2.polylines(img, [pts[i:i + 4]], False, colour, t, cv2.LINE_AA)
    else:
        cv2.polylines(img, [pts], True, colour, t, cv2.LINE_AA)
    return img


def raw_gain(d, max_gain=2.5):
    """The auto display gain of droplet d's raw image (shared by the raw, background
    and subtracted views so their brightness stays comparable)."""
    return display_gain(d.rgb, max_gain)


def view_raw(d, gain="auto"):
    """Stitched image (1/4 size) for display: auto gain (×1-2.5, so dim webs are
    brightened but bright droplets aren't blown out) with a soft roll-off. Pass a
    number for a fixed gain."""
    return _show(d.rgb, gain)


def view_ring_map(d):
    """Ring map (magma colours) with the fitted rim (cyan) and centre."""
    cv2 = _cv2()
    img = cv2.applyColorMap((d.ring * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)[..., ::-1].copy()
    if d.seg:
        _draw_rim(img, d.seg, colour=(0, 220, 255))
        c = (int(d.seg["cx"]), int(d.seg["cy"]))
        cv2.drawMarker(img, c, (0, 220, 255), cv2.MARKER_CROSS, max(8, img.shape[1] // 25),
                       max(1, img.shape[1] // 300))
    return img


def view_rim(d, gain="auto"):
    """Raw image with the fitted rim: green if the droplet passes QC, red if not."""
    img = view_raw(d, gain).copy()
    if d.seg:
        _draw_rim(img, d.seg, colour=(0, 255, 0) if d.qc_ok else (255, 60, 60))
    return img


def view_background(d, gain="auto"):
    """Normalisation step 1: the background region (blue) = imaged pixels beyond 1.25 R."""
    g = raw_gain(d) if gain == "auto" else gain
    raw = view_raw(d, g)
    img = raw.astype(np.float32) * 0.55
    if d.norm_info is not None:
        bgm = d.norm_info["bg_mask"]
        img[bgm] = img[bgm] * 0.4 + np.array([25, 128, 255]) * 0.6
        m = d.seg["mask"].astype(bool)
        img[m] = raw[m]
        img = img.astype(np.uint8)
        _draw_rim(img, d.seg, colour=(255, 255, 255))
        _draw_rim(img, d.seg, 1.25, colour=(255, 255, 255), dashed=True)
    return img.astype(np.uint8)


def view_subtracted(d, gain="auto"):
    """Normalisation step 2: background subtracted, at the SAME gain as view_raw (auto
    gain is taken from the raw image), so the drop in brightness is visible."""
    g = raw_gain(d) if gain == "auto" else gain
    if d.norm_info is None:
        return view_raw(d, g)
    x = d.rgb.astype(np.float32) / 255
    sub = np.clip(x - d.norm_info["bg"], 0, None)
    sub[d.rgb.max(2) == 0] = 0
    return _show(sub, g)


def _norm_float(d):
    """Unclipped normalised image, 0..1 scale (values above 1 = brighter than white)."""
    if d.norm_info is not None and "float" in d.norm_info:
        return d.norm_info["float"] / 255
    return d.norm if d.norm is not None else d.rgb


def view_normalised(d, gain="auto"):
    """Normalisation step 3: every channel divided by its droplet mean. Shown from the
    unclipped values with auto gain (≤ ×1.6) and soft roll-off."""
    return _show(_norm_float(d), gain, max_gain=1.6)


def view_unwrap(d, normalised=True, gain="auto", height=256):
    """Radial unwrap for display: angle 0-360° left to right, centre at the BOTTOM,
    rim (r/R = 1) marked by a dashed white line."""
    cv2 = _cv2()
    p = d.polar_norm if normalised else d.polar
    if p is None:
        return np.zeros((height, 2 * height, 3), np.uint8)
    if gain is None:
        gain = "auto"
    img = _show(p, gain, max_gain=1.6 if normalised else 2.5)[::-1].copy()
    img = cv2.resize(img, (N_T * 2, height), interpolation=cv2.INTER_LINEAR)
    y = int(round((1 - 1 / R_MAX) * (height - 1)))
    for x in range(0, img.shape[1], 12):
        img[y, x:x + 6] = 255
    return img


def view_crop(d, normalised=False, size=256):
    """Droplet-centred square crop."""
    if d.seg is None:
        return _show(d.rgb)
    if normalised:
        return _show(crop(_norm_float(d).astype(np.float32), d.seg, size), max_gain=1.6)
    return _show(crop(d.rgb, d.seg, size))


def normalisation_numbers(d):
    """Per-channel (R, G, B) numbers of the normalisation, on a 0-255 scale."""
    if d.norm_info is None:
        return None
    i = d.norm_info
    return dict(background=(i["bg"] * 255).round(1).tolist(),
                droplet_mean_raw=(i["mean_raw"] * 255).round(1).tolist(),
                droplet_mean_after_subtraction=(i["mean_sub"] * 255).round(1).tolist(),
                gain=i["gain"].round(2).tolist(),
                background_share_of_image=round(i["bg_fraction"], 3))


# --------------------------------------------------------------------------- #
#  Cached per-image record (what the dataset and sample averages use)
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    path: str
    label: str
    qc_ok: bool
    qc_reason: str
    qc: dict                                 # ring_score, coverage, outside_frac, polar_valid, h, w
    scalars: np.ndarray = None               # len(SCALAR_NAMES), raw
    scalars_norm: np.ndarray = None
    radial: np.ndarray = None                # 6 × N_BINS, raw
    radial_norm: np.ndarray = None
    profiles: dict = None                    # full-resolution raw profiles (N_R)
    profiles_norm: dict = None
    polar_small: np.ndarray = None           # 64 × 180 × 3 uint8 (raw), 0 outside image
    polar_small_norm: np.ndarray = None
    polar_valid_small: np.ndarray = None     # 64 × 180 bool
    crop_small: np.ndarray = None            # 128 × 128 × 3 uint8 droplet crop (raw)
    crop_small_norm: np.ndarray = None
    cnn: np.ndarray = None                   # CNN embedding (attached by add_embeddings; cached
    cnn_norm: np.ndarray = None              # separately, never saved in the record file)

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.path))[0].replace("_stitched", "")


def _small_polar(p):
    cv2 = _cv2()
    valid = np.isfinite(p[..., 0])
    img = (np.clip(np.nan_to_num(p), 0, 1) * 255).astype(np.uint8)
    img = cv2.resize(img, (180, 64), interpolation=cv2.INTER_AREA)
    v = cv2.resize(valid.astype(np.uint8), (180, 64), interpolation=cv2.INTER_NEAREST).astype(bool)
    return img, v


def record_from_droplet(d):
    h, w = d.rgb.shape[:2]
    if d.seg is None:
        return Record(d.path, d.label, False, d.qc_reason, dict(h=h, w=w))
    pv = float(np.isfinite(d.polar[..., 0]).mean())
    qc = dict(ring_score=d.seg["score"], coverage=d.seg["coverage"],
              outside_frac=d.seg["outside_frac"], polar_valid=pv, h=h, w=w)
    ps, vs = _small_polar(d.polar)
    psn, _ = _small_polar(d.polar_norm)
    return Record(
        d.path, d.label, d.qc_ok, d.qc_reason, qc,
        np.array([d.scalars[k] for k in SCALAR_NAMES], np.float32),
        np.array([d.scalars_norm[k] for k in SCALAR_NAMES], np.float32),
        radial_vector(d.profiles), radial_vector(d.profiles_norm),
        d.profiles, d.profiles_norm, ps, psn, vs,
        crop(d.rgb, d.seg, 128), crop(d.norm, d.seg, 128))


def _record_file(path, downsample):
    st = os.stat(path)
    key = f"{VERSION}|{os.path.normcase(os.path.abspath(path))}|{st.st_mtime}|{st.st_size}|{downsample}"
    return (path, st.st_mtime, st.st_size, downsample), os.path.join(
        CACHE_DIR, hashlib.sha1(key.encode("utf-8")).hexdigest() + ".npz")


def _save_record(rec, fn):
    arrays = {k: v for k, v in rec.__dict__.items()
              if isinstance(v, np.ndarray) and k not in ("cnn", "cnn_norm")}
    for which in ("profiles", "profiles_norm"):
        for k, v in (getattr(rec, which) or {}).items():
            arrays[f"{which}__{k}"] = v
    meta = dict(path=rec.path, label=rec.label, qc_ok=rec.qc_ok, qc_reason=rec.qc_reason, qc=rec.qc)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = fn + f".{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, fn)


def _load_record(fn):
    z = np.load(fn, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    rec = Record(meta["path"], meta["label"], meta["qc_ok"], meta["qc_reason"], meta["qc"])
    prof, prof_n = {}, {}
    for k in z.files:
        if k.startswith("profiles__"):
            prof[k.split("__", 1)[1]] = z[k]
        elif k.startswith("profiles_norm__"):
            prof_n[k.split("__", 1)[1]] = z[k]
        elif k != "meta":
            setattr(rec, k, z[k])
    rec.profiles, rec.profiles_norm = prof or None, prof_n or None
    return rec


def features(path, label=None, downsample=DOWNSAMPLE):
    """Record for one image, from memory, the disk cache (~/.whisky_stitcher_ml) or
    a fresh analysis. Each image is only analysed once until the file changes."""
    key, fn = _record_file(path, downsample)
    with _cache_lock:
        if key in _feature_cache:
            return _feature_cache[key]
    try:
        rec = _load_record(fn)
    except (OSError, ValueError, KeyError):
        rec = record_from_droplet(analyse(path, label, downsample))
        try:
            _save_record(rec, fn)
        except OSError:
            pass
    if label:
        rec.label = label
    with _cache_lock:
        _feature_cache[key] = rec
    return rec


def features_many(paths, labels=None, downsample=DOWNSAMPLE, log=print, progress=None,
                  should_stop=lambda: False, workers=READERS):
    """Records for many images (threads). Images that can't be read are skipped and logged.
    Returns (records in input order with None for failures, [(path, error)])."""
    labels = labels or [None] * len(paths)
    out, skipped = [None] * len(paths), []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(features, p, l, downsample): i for i, (p, l) in enumerate(zip(paths, labels))}
        for done, fut in enumerate(as_completed(futs), 1):
            if should_stop():
                for f in futs:
                    f.cancel()
                raise InterruptedError("stopped")
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                skipped.append((paths[i], str(e)))
                log(f"  skipped {os.path.basename(paths[i])}: {e}")
            if progress:
                progress(done, len(paths))
    return out, skipped


# --------------------------------------------------------------------------- #
#  Optional: frozen CNN embeddings (PyTorch + torchvision)
# --------------------------------------------------------------------------- #
# A pretrained EfficientNet-B0 (ImageNet) is used only as a fixed feature extractor:
# it is never trained here. Each droplet-centred crop (side 2.4 R, 256 px) goes in and
# 1280 numbers come out; those feed the same standardise -> PCA -> logistic regression
# as the hand-made features. Same recipe as ml/embed_cnn.py (research run). The
# embedding is averaged over 4 rotations × 2 flips so droplet orientation doesn't matter.
CNN_VERSION = "effb0-imagenet1k-v1|crop256|x2.5|rot4flip2|norm-tonemap"
CNN_DIM = 1280
CNN_BATCH = 16
_cnn = {}
_cnn_lock = threading.Lock()


def cnn_available():
    """'' if the CNN feature sets can be used, else what to install."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return ""
    except ImportError:
        return ("The CNN feature sets need PyTorch (CPU is enough, about 1 GB):\n"
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")


def _cnn_net():
    """The network, loaded once. The ImageNet weights (~20 MB) are downloaded on first
    use into PyTorch's cache (~/.cache/torch)."""
    with _cnn_lock:
        if "net" not in _cnn:
            import torch
            import torchvision
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            net = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
            net.classifier = torch.nn.Identity()
            net.eval()
            _cnn["net"] = net
            _cnn["mean"] = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            _cnn["std"] = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return _cnn["net"], _cnn["mean"], _cnn["std"]


def embed_crops(crops, gain=2.5):
    """N × 256 × 256 × 3 uint8 crops -> N × 1280 float32 embeddings. gain brightens raw
    dark-field crops (dim next to ImageNet photos); tone-mapped normalised crops use 1."""
    import torch
    net, mean, std = _cnn_net()
    x = torch.from_numpy(np.ascontiguousarray(np.stack(crops))).permute(0, 3, 1, 2).float() / 255
    x = torch.clamp(x * gain, 0, 1)
    with torch.inference_mode():
        out = 0
        for k in range(4):
            v = torch.rot90(x, k, (2, 3))
            for w in (v, torch.flip(v, (3,))):
                out = out + net((w - mean) / std)
    return (out / 8).numpy().astype(np.float32)


def _embed_file(path, downsample):
    st = os.stat(path)
    # VERSION is part of the key: the crops depend on the droplet segmentation
    key = f"{CNN_VERSION}|{VERSION}|{os.path.normcase(os.path.abspath(path))}|{st.st_mtime}|{st.st_size}|{downsample}"
    return os.path.join(CACHE_DIR, "cnn", hashlib.sha1(key.encode("utf-8")).hexdigest() + ".npz")


def _cnn_crops(path, downsample):
    """(raw crop, normalised crop) at 256 px, or None if no droplet. Reads the image
    again: the feature cache only keeps 128 px crops, too coarse for the CNN."""
    rgb = read_small(path, downsample)
    seg = segment(rgb)
    if seg is None:
        return None
    _, info = normalise(rgb, seg)
    # raw crop: brightened ×2.5 with a hard clamp in embed_crops. A soft roll-off instead
    # (so bright M/O/V/W/Y centres aren't white blobs) was tested 15 Nov 2026 and did not
    # help: balanced accuracy 0.958 -> 0.953 (V recall 0.93 -> 0.84, the rest ±0.01), so
    # the clamp stays. The display uses the soft roll-off (view_*), the CNN does not.
    return crop(rgb, seg, 256), crop(tone_map(info["float"]), seg, 256)


def add_embeddings(records, downsample=DOWNSAMPLE, log=print, progress=None, should_stop=lambda: False,
                   workers=READERS):
    """Attach .cnn and .cnn_norm to every record with a droplet, from the disk cache
    (~/.whisky_stitcher_ml/cnn) or by computing them. The first time, each image is
    read again (about as slow as the first feature run) and the CNN costs ~0.3 s per
    droplet on a CPU; afterwards it is instant. Returns the number computed."""
    problem = cnn_available()
    if problem:
        raise RuntimeError(problem)
    todo = []
    for r in records:
        if r is None or r.scalars is None or r.cnn is not None:
            continue
        try:
            fn = _embed_file(r.path, downsample)
            z = np.load(fn)
            r.cnn, r.cnn_norm = z["raw"], z["norm"]
        except (OSError, KeyError, ValueError):
            todo.append(r)
    if not todo:
        return 0
    log(f"CNN embeddings: computing {len(todo)} (cached afterwards)...")
    _cnn_net()
    os.makedirs(os.path.join(CACHE_DIR, "cnn"), exist_ok=True)
    done = 0
    batches = [todo[i:i + CNN_BATCH] for i in range(0, len(todo), CNN_BATCH)]
    with ThreadPoolExecutor(workers) as pool:
        # Images for the next batches are read while the CNN works on this one
        # (reading is network-bound, the CNN is CPU-bound). At most 3 batches ahead.
        pending = []

        def submit(b):
            pending.append([pool.submit(_safe, _cnn_crops, r.path, downsample) for r in b])
        for b in batches[:3]:
            submit(b)
        for bi, chunk in enumerate(batches):
            futs = pending.pop(0)
            if bi + 3 < len(batches):
                submit(batches[bi + 3])
            crops = [f.result() for f in futs]
            if should_stop():
                for fl in pending:
                    for f in fl:
                        f.cancel()
                raise InterruptedError("stopped")
            ok = [(r, c) for r, c in zip(chunk, crops) if isinstance(c, tuple)]
            for r, c in zip(chunk, crops):
                if not isinstance(c, tuple):
                    log(f"  no CNN embedding for {r.name}: {c or 'no droplet found'}")
            if ok:
                e_raw = embed_crops([c[0] for _, c in ok], gain=2.5)
                e_norm = embed_crops([c[1] for _, c in ok], gain=1.0)
                for i, (r, _) in enumerate(ok):
                    r.cnn, r.cnn_norm = e_raw[i], e_norm[i]
                    try:
                        fn = _embed_file(r.path, downsample)
                        tmp = fn + f".{os.getpid()}.tmp"
                        with open(tmp, "wb") as fh:
                            np.savez(fh, raw=r.cnn, norm=r.cnn_norm)
                        os.replace(tmp, fn)
                    except OSError:
                        pass
            done += len(chunk)
            if progress:
                progress(done, len(todo))
    return len(todo)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        return str(e)


# --------------------------------------------------------------------------- #
#  Average of a whole sample
# --------------------------------------------------------------------------- #
@dataclass
class SampleSummary:
    label: str
    n_images: int
    n_used: int                              # passed QC (or all, if include_failed)
    rejected: list                           # [(name, reason)]
    r: np.ndarray                            # r/R of each profile point
    profile_mean: dict                       # key -> mean over droplets (raw)
    profile_sd: dict
    profile_mean_norm: dict
    profile_sd_norm: dict
    mean_unwrap: np.ndarray                  # 64 × 180 × 3 uint8, average over droplets
    mean_unwrap_norm: np.ndarray
    mean_droplet: np.ndarray                 # 128 × 128 × 3 uint8: the "average droplet"
    mean_droplet_norm: np.ndarray
    scalar_mean: dict
    scalar_sd: dict


def summarise(records, label=None, include_failed=False):
    """Average of every droplet of one sample: mean ± SD radial profiles, the mean
    unwrap, the 'average droplet' (mean of droplet-centred crops; rotations are
    random, so it shows the radial structure) and mean ± SD of the scalar features."""
    recs = [r for r in records if r is not None]
    use = [r for r in recs if r.scalars is not None and (r.qc_ok or include_failed)]
    rejected = [(r.name, r.qc_reason) for r in recs if not r.qc_ok]
    if not use:
        raise ValueError("no droplet in this sample passed QC" if recs else "no images")

    def stack_prof(which):
        keys = list(getattr(use[0], which).keys())
        arr = {k: np.stack([getattr(r, which)[k] for r in use]) for k in keys}
        return {k: v.mean(0) for k, v in arr.items()}, {k: v.std(0) for k, v in arr.items()}

    pm, ps = stack_prof("profiles")
    pmn, psn = stack_prof("profiles_norm")

    def mean_polar(attr):
        P = np.stack([getattr(r, attr).astype(np.float32) for r in use])
        V = np.stack([r.polar_valid_small for r in use])[..., None].astype(np.float32)
        return (np.sum(P * V, 0) / np.maximum(V.sum(0), 1)).astype(np.uint8)

    S = np.stack([r.scalars for r in use])
    return SampleSummary(
        label=label or use[0].label, n_images=len(recs), n_used=len(use), rejected=rejected,
        r=np.linspace(0, R_MAX, N_R), profile_mean=pm, profile_sd=ps,
        profile_mean_norm=pmn, profile_sd_norm=psn,
        mean_unwrap=mean_polar("polar_small"), mean_unwrap_norm=mean_polar("polar_small_norm"),
        mean_droplet=np.mean([r.crop_small for r in use], 0).astype(np.uint8),
        mean_droplet_norm=np.mean([r.crop_small_norm for r in use], 0).astype(np.uint8),
        scalar_mean=dict(zip(SCALAR_NAMES, S.mean(0).tolist())),
        scalar_sd=dict(zip(SCALAR_NAMES, S.std(0).tolist())))


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    folder: str
    records: list                            # Record per image (failed reads dropped)
    skipped: list = field(default_factory=list)

    @property
    def labels(self):
        return np.array([r.label for r in self.records])

    @property
    def qc_ok(self):
        return np.array([r.qc_ok for r in self.records])

    def qc_table(self):
        """[(label, images, passed QC)] in key order."""
        from identify_core import code_order
        out = {}
        for r in self.records:
            n, k = out.get(r.label, (0, 0))
            out[r.label] = (n + 1, k + int(r.qc_ok))
        return [(l, *out[l]) for l in sorted(out, key=code_order)]


def build_dataset(folder, log=print, progress=None, should_stop=lambda: False, downsample=DOWNSAMPLE):
    """Features of every stitched image in the whisky sub-folders of `folder`
    (same images as Identify and Gallery). The first run analyses every image
    (~1 s per image with 6 at once; cached afterwards)."""
    import identify_core as idc
    classes, unstitched = idc.scan_classes(folder)
    if unstitched:
        log(f"  not stitched yet, left out: {', '.join(unstitched)}")
    items = [(label, p) for label, paths in classes for p in paths]
    if not items:
        raise ValueError("no images found in sub-folders of this folder.")
    log(f"Analysing {len(items)} images from {len(classes)} whiskies...")
    recs, skipped = features_many([p for _, p in items], [l for l, _ in items], downsample,
                                  log, progress, should_stop)
    ds = Dataset(folder, [r for r in recs if r is not None], skipped)
    log(f"  {int(ds.qc_ok.sum())} of {len(ds.records)} droplets passed QC")
    return ds


def feature_matrix(records, feature_set="scalar+radial", normalised=False):
    """(X, column names) for records (all must have features)."""
    sc_names = [n for n in SCALAR_NAMES if not (normalised and n in RAW_BRIGHTNESS + NORM_CONSTANT)]
    sc_idx = [SCALAR_NAMES.index(n) for n in sc_names]
    if feature_set == "nuisance":
        cols = []
        for r in records:
            q = r.qc
            cols.append([r.scalars[SCALAR_NAMES.index("radius_px")], q["ring_score"], q["coverage"],
                         q["outside_frac"], q["h"], q["w"]])
        return np.array(cols, np.float32), list(NUISANCE)
    parts, names = [], []
    if "scalar" in feature_set:
        parts.append(np.stack([(r.scalars_norm if normalised else r.scalars)[sc_idx] for r in records]))
        names += sc_names
    if "radial" in feature_set:
        parts.append(np.stack([r.radial_norm if normalised else r.radial for r in records]))
        names += radial_vector_names()
    if "cnn" in feature_set:
        emb = [r.cnn_norm if normalised else r.cnn for r in records]
        if any(e is None for e in emb):
            raise ValueError("CNN embeddings are missing: call add_embeddings() first")
        parts.insert(0, np.stack(emb))
        names = [f"cnn_{i}" for i in range(len(emb[0]))] + names
    if not parts:
        raise ValueError(f"unknown feature set {feature_set!r}; use one of {FEATURE_SETS}")
    return np.hstack(parts).astype(np.float32), names


# --------------------------------------------------------------------------- #
#  Classifier: standardise -> PCA -> multinomial logistic regression (numpy)
# --------------------------------------------------------------------------- #
@dataclass
class Model:
    classes: list
    feature_set: str
    normalised: bool
    feature_names: list
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray                   # PCA rows (k × d)
    W: np.ndarray                            # k × n_classes
    b: np.ndarray
    info: dict = field(default_factory=dict) # training summary (n per class, CV result, date,
                                             # unknown_threshold: {n droplets: threshold})

    def proba(self, X):
        Z = ((np.asarray(X, np.float32) - self.mean) / self.scale) @ self.components.T
        return _softmax(Z @ self.W + self.b)

    def unknown_threshold(self, n):
        """Threshold on 1 − max(mean probability) for a sample of n droplets, above which
        it looks unlike every trained whisky (calibrated at ~5% false alarms in CV), or None."""
        t = self.info.get("unknown_threshold")
        if not t:
            return None
        return t[str(min(max(int(n), 1), max(int(k) for k in t)))]


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit(X, y, classes, feature_set, normalised, feature_names, C=0.5, var_keep=0.99,
        balanced=True, max_iter=3000, tol=1e-6, history=None, X_val=None, y_val=None,
        record_every=10, should_stop=None, lr=0.05, lr_halflife=300, plateau=1e-6):
    """Fit the classifier. y: labels (strings) in `classes`.

    var_keep: share of variance the PCA keeps. 0.99 (tested 14 Nov 2026: 0.95 cost the
    scalar/radial models ~3 points of balanced accuracy and didn't help the CNN sets).
    Adam with the learning rate halving every `lr_halflife` steps (a fixed rate made
    the loss spike long after convergence), stopping once the loss improves by less
    than `plateau` (relative) over 50 steps.
    history: a list to append {"step", "loss", "train_acc", "val_acc"} to every
    `record_every` steps (each step is one full pass over the data, i.e. one epoch).
    X_val / y_val: optional held-out data scored at the same steps."""
    X = np.asarray(X, np.float64)
    yi = np.array([classes.index(v) for v in y])
    mean = X.mean(0)
    scale = X.std(0)
    scale[scale < 1e-9] = 1.0
    Xs = (X - mean) / scale
    _, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    var = s ** 2 / max(1e-12, (s ** 2).sum())
    k = int(np.searchsorted(np.cumsum(var), var_keep) + 1)
    comps = Vt[:k]
    Z = Xs @ comps.T
    n, d = Z.shape
    K = len(classes)
    Y = np.eye(K)[yi]
    counts = np.bincount(yi, minlength=K).astype(np.float64)
    sw = (n / (K * np.maximum(counts, 1)))[yi] if balanced else np.ones(n)
    lam = 1.0 / (C * n)
    W, b = np.zeros((d, K)), np.zeros(K)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    lr0, b1, b2 = lr, 0.9, 0.999
    prev_loss = np.inf
    if history is not None and X_val is not None and len(X_val):
        Zv = ((np.asarray(X_val, np.float64) - mean) / scale) @ comps.T
        yv = np.array([classes.index(v) if v in classes else -1 for v in y_val])
    else:
        Zv = None

    def record(it, P):
        loss = float(-(sw * np.log(P[np.arange(n), yi] + 1e-12)).sum() / n + 0.5 * lam * (W ** 2).sum())
        h = dict(step=it, loss=loss, train_acc=float((P.argmax(1) == yi).mean()), val_acc=None)
        if Zv is not None:
            h["val_acc"] = float(((Zv @ W + b).argmax(1) == yv).mean())
        history.append(h)

    for it in range(1, max_iter + 1):         # Adam on the L2-regularised cross-entropy
        P = _softmax(Z @ W + b)
        if history is not None and (it <= 50 or it % record_every == 0):
            record(it, P)
        if should_stop is not None and it % 50 == 0 and should_stop():
            raise InterruptedError("stopped")
        G = (P - Y) * sw[:, None] / n
        gW, gb = Z.T @ G + lam * W, G.sum(0)
        mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW ** 2
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
        lr_t = lr0 * 0.5 ** (it / lr_halflife)
        W -= lr_t * (mW / (1 - b1 ** it)) / (np.sqrt(vW / (1 - b2 ** it)) + 1e-8)
        b -= lr_t * (mb / (1 - b1 ** it)) / (np.sqrt(vb / (1 - b2 ** it)) + 1e-8)
        if it % 50 == 0:
            if max(np.abs(gW).max(), np.abs(gb).max()) < tol:
                break
            loss = float(-(sw * np.log(P[np.arange(n), yi] + 1e-12)).sum() / n + 0.5 * lam * (W ** 2).sum())
            if it >= 200 and prev_loss - loss < plateau * max(1.0, abs(loss)):
                break
            prev_loss = loss
    if history is not None and (not history or history[-1]["step"] != it):
        record(it, _softmax(Z @ W + b))
    return Model(list(classes), feature_set, normalised, list(feature_names),
                 mean.astype(np.float32), scale.astype(np.float32), comps.astype(np.float32),
                 W.astype(np.float32), b.astype(np.float32))


def _folds(y, k, groups=None, seed=0):
    """Fold number of each sample: stratified by class, or by whole groups if given."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), int)
    if groups is not None:
        g = np.asarray(groups)
        ug = rng.permutation(np.unique(g))
        for i, gg in enumerate(ug):
            fold[g == gg] = i % k
        return fold
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = (np.arange(len(idx)) + rng.integers(k)) % k
    return fold


@dataclass
class CVResult:
    classes: list
    feature_set: str
    normalised: bool
    folds: int
    grouped: bool
    y_true: np.ndarray
    y_pred: np.ndarray
    proba: np.ndarray
    paths: list
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    chance: float
    confusion: np.ndarray                    # rows = true, cols = predicted (classes order)
    recall: dict                             # per class
    excluded: dict                           # reason -> count

    def summary(self):
        return (f"{self.feature_set}{' (normalised)' if self.normalised else ''}: balanced accuracy "
                f"{self.balanced_accuracy:.3f} (accuracy {self.accuracy:.3f}, macro-F1 {self.macro_f1:.3f}; "
                f"chance {self.chance:.3f}; {len(self.y_true)} droplets, {len(self.classes)} whiskies, "
                f"{self.folds}-fold{' grouped' if self.grouped else ''})")

    def sample_accuracy(self, n, repeats=50, seed=0):
        """Balanced accuracy when n held-out droplets of the same whisky are averaged
        (simulated from the CV probabilities): how much several droplets help."""
        rng = np.random.default_rng(seed)
        rec = []
        for j, c in enumerate(self.classes):
            idx = np.where(self.y_true == c)[0]
            if len(idx) < n:
                continue
            hits = [self.proba[rng.choice(idx, n, replace=False)].mean(0).argmax() == j for _ in range(repeats)]
            rec.append(np.mean(hits))
        return float(np.mean(rec)) if rec else float("nan")

    def unknown_thresholds(self, max_n=10, false_alarm=0.05, repeats=200, seed=0):
        """{n: threshold} on 1 − max(mean probability) of n droplets of a KNOWN whisky,
        exceeded by only `false_alarm` of them (simulated from the CV probabilities).
        A sample above it looks unlike every trained whisky. One droplet alone is weak
        (Nov 2026 test: ~50% of an unseen whisky caught); 3-5 droplets are far better."""
        rng = np.random.default_rng(seed)
        out = {}
        for n in range(1, max_n + 1):
            s = []
            for c in self.classes:
                idx = np.where(self.y_true == c)[0]
                if len(idx) >= n:
                    s += [1 - self.proba[rng.choice(idx, n, replace=False)].mean(0).max()
                          for _ in range(max(1, repeats // len(self.classes)))]
            if s:
                out[str(n)] = float(np.percentile(s, 100 * (1 - false_alarm)))
        return out


def _metrics(y, p, classes):
    K = len(classes)
    yi = np.array([classes.index(v) for v in y])
    pi = np.array([classes.index(v) for v in p])
    cm = np.zeros((K, K), int)
    np.add.at(cm, (yi, pi), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-12), 0)
    present = cm.sum(1) > 0
    return dict(accuracy=float(np.trace(cm) / max(1, cm.sum())),
                balanced_accuracy=float(rec[present].mean()),
                macro_f1=float(f1[present].mean()), confusion=cm,
                recall=dict(zip(classes, rec.round(4).tolist())))


def _usable(ds, min_per_class, feature_set=""):
    recs = [r for r in ds.records if r.qc_ok and r.scalars is not None]
    excluded = {}
    for r in ds.records:
        if not r.qc_ok:
            excluded[r.qc_reason.split(" (")[0]] = excluded.get(r.qc_reason.split(" (")[0], 0) + 1
    if "cnn" in feature_set:
        if all(r.cnn is None for r in recs):
            raise ValueError("CNN embeddings are missing: call add_embeddings(ds.records) first")
        n = sum(r.cnn is None for r in recs)
        if n:
            excluded["no CNN embedding"] = n
        recs = [r for r in recs if r.cnn is not None]
    counts = {}
    for r in recs:
        counts[r.label] = counts.get(r.label, 0) + 1
    small = {l for l, n in counts.items() if n < min_per_class}
    if small:
        excluded[f"whisky with fewer than {min_per_class} good droplets"] = sum(counts[l] for l in small)
    recs = [r for r in recs if r.label not in small]
    from identify_core import code_order
    classes = sorted({r.label for r in recs}, key=code_order)
    return recs, classes, excluded, sorted(small, key=code_order)


def cross_validate(ds, feature_set="scalar+radial", normalised=False, folds=5, groups=None,
                   min_per_class=5, C=0.5, progress=None, should_stop=lambda: False, log=print):
    """k-fold cross-validation on the QC-passed droplets.

    groups: None (stratified random folds; droplets of one slip can be split
    across folds, so this measures 'recognise another droplet from the same
    slip'), or a function path -> group id (e.g. slip or day) so whole groups are
    held out together."""
    recs, classes, excluded, small = _usable(ds, min_per_class, feature_set)
    if small:
        log(f"  left out (fewer than {min_per_class} good droplets): {', '.join(small)}")
    if len(classes) < 2:
        raise ValueError("need at least two whiskies with enough good droplets")
    X, names = feature_matrix(recs, feature_set, normalised)
    y = np.array([r.label for r in recs])
    g = [groups(r.path) for r in recs] if callable(groups) else groups
    fold = _folds(y, folds, g)
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(classes)), np.float32)
    for k in range(folds):
        if should_stop():
            raise InterruptedError("stopped")
        te, tr = fold == k, fold != k
        if not te.any():
            continue
        tr_classes = [c for c in classes if (y[tr] == c).any()]
        m = fit(X[tr], y[tr], tr_classes, feature_set, normalised, names, C=C)
        p = m.proba(X[te])
        for j, c in enumerate(tr_classes):
            proba[te, classes.index(c)] = p[:, j]
        pred[te] = np.array(tr_classes)[p.argmax(1)]
        if progress:
            progress(k + 1, folds)
    met = _metrics(y, pred, classes)
    return CVResult(classes, feature_set, normalised, folds, g is not None, y, pred.astype(str), proba,
                    [r.path for r in recs], met["accuracy"], met["balanced_accuracy"], met["macro_f1"],
                    1 / len(classes), met["confusion"], met["recall"], excluded)


def train_model(ds, feature_set="scalar+radial", normalised=False, min_per_class=5, C=0.5,
                cv_result=None):
    """Train on every QC-passed droplet. Pass the CVResult to store its scores in the model."""
    import datetime
    recs, classes, excluded, small = _usable(ds, min_per_class, feature_set)
    X, names = feature_matrix(recs, feature_set, normalised)
    y = [r.label for r in recs]
    m = fit(X, y, classes, feature_set, normalised, names, C=C)
    m.info = dict(trained=datetime.datetime.now().isoformat(timespec="seconds"),
                  folder=ds.folder, version=VERSION, downsample=DOWNSAMPLE,
                  n_per_class={c: y.count(c) for c in classes}, left_out=small,
                  qc=dict(min_score=QC_MIN_SCORE, max_outside=QC_MAX_OUTSIDE, min_valid=QC_MIN_VALID))
    if "cnn" in feature_set:
        m.info["cnn"] = CNN_VERSION
    if cv_result is not None:
        m.info["cv"] = dict(balanced_accuracy=cv_result.balanced_accuracy, accuracy=cv_result.accuracy,
                            macro_f1=cv_result.macro_f1, folds=cv_result.folds, grouped=cv_result.grouped,
                            sample_accuracy={n: cv_result.sample_accuracy(n) for n in (1, 3, 5)})
        m.info["unknown_threshold"] = cv_result.unknown_thresholds()
    return m


def training_report(ds, feature_set="scalar+radial", normalised=False, min_per_class=5, C=0.5,
                    holdout=5, should_stop=lambda: False, log=print):
    """Look inside one training run, for figure_training: trains on (holdout-1)/holdout
    of the droplets and scores the held-out rest after every step.

    Returns a dict with the PCA explained variance and kept components, every
    droplet's position on PC1/PC2, the per-step history (loss, train and held-out
    accuracy) and each original feature's weight in the final model."""
    recs, classes, _, small = _usable(ds, min_per_class, feature_set)
    if small:
        log(f"  left out (fewer than {min_per_class} good droplets): {', '.join(small)}")
    if len(classes) < 2:
        raise ValueError("need at least two whiskies with enough good droplets")
    X, names = feature_matrix(recs, feature_set, normalised)
    y = np.array([r.label for r in recs])
    fold = _folds(y, holdout)
    tr, te = fold != 0, fold == 0
    tr_classes = [c for c in classes if (y[tr] == c).any()]
    hist = []
    m = fit(X[tr], y[tr], tr_classes, feature_set, normalised, names, C=C, history=hist,
            X_val=X[te], y_val=y[te], should_stop=should_stop)
    Xs = (np.asarray(X, np.float64) - m.mean) / m.scale
    s = np.linalg.svd(Xs[tr], compute_uv=False)
    var = s ** 2 / max(1e-12, (s ** 2).sum())
    # Weight of each original (standardised) feature: PCA rows back-projected through W,
    # size taken over the whiskies. Large = the model leans on that feature.
    weight = np.linalg.norm(m.components.T.astype(np.float64) @ m.W, axis=1)
    return dict(classes=tr_classes, feature_set=feature_set, normalised=normalised,
                explained=var, kept=len(m.components), n_features=X.shape[1],
                scores=Xs @ m.components[:2].T.astype(np.float64), labels=y, held_out=te,
                history=hist, feature_names=names, feature_weight=weight,
                n_train=int(tr.sum()), n_test=int(te.sum()))


def save_model(model, path):
    meta = dict(classes=model.classes, feature_set=model.feature_set, normalised=model.normalised,
                feature_names=model.feature_names, info=model.info)
    with open(path, "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta)), mean=model.mean, scale=model.scale,
                            components=model.components, W=model.W, b=model.b)


def load_model(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta.get("info", {}).get("version") not in (None, VERSION):
        raise ValueError(f"model was made with {meta['info']['version']}, this app uses {VERSION}; "
                         "train it again")
    if "cnn" in meta["feature_set"] and meta.get("info", {}).get("cnn") != CNN_VERSION:
        raise ValueError("model was made with a different CNN feature extractor; train it again")
    return Model(meta["classes"], meta["feature_set"], meta["normalised"], meta["feature_names"],
                 z["mean"], z["scale"], z["components"], z["W"], z["b"], meta.get("info", {}))


@dataclass
class Prediction:
    path: str
    label: str                               # best whisky ("" if QC failed and not forced)
    probability: float
    top: list                                # [(label, probability)] best first
    margin: float                            # best minus runner-up probability
    qc_ok: bool
    qc_reason: str
    unknown: bool = False                    # looks unlike every trained whisky (weak for one
                                             # droplet; use predict_sample for a real check)

    @property
    def confidence(self):
        if not self.qc_ok:
            return "unreliable (failed QC)"
        if self.unknown:
            return "possibly not a trained whisky"
        if self.margin < 0.1:
            return "low (close call)"
        return "high" if self.probability >= 0.8 else "medium"


@dataclass
class SamplePrediction:
    """Several droplets of one sample, judged together (mean of their probabilities)."""
    label: str
    probability: float                       # mean probability of `label`
    top: list                                # [(label, mean probability)]
    margin: float
    n_used: int                              # droplets that passed QC and were used
    n_images: int
    agreement: float                         # share of single droplets voting for `label`
    unknown: bool                            # unlike every trained whisky (5% false-alarm threshold)
    unknown_score: float                     # 1 − max mean probability
    unknown_threshold: float                 # None if the model has no calibration
    droplets: list                           # Prediction per image

    @property
    def verdict(self):
        if self.n_used == 0:
            return "no usable droplets (all failed QC)"
        if self.unknown:
            return f"not like any trained whisky (closest: {self.label})"
        if self.margin < 0.1:
            return f"{self.label}, but close to {self.top[1][0]}"
        return self.label


def predict(model, paths, top=3, log=print, progress=None, should_stop=lambda: False,
            predict_failed=True, qc_mode="microscope", downsample=None):
    """Predict the whisky of each image. Images failing QC still get a prediction
    (flagged) when predict_failed is True, unless no rim was found at all.
    qc_mode: "microscope" (stitched dark-field images) or "phone" (single photos, e.g.
    phone-microscope attachment; prepare them first, see ml/phone_prep.py). downsample:
    analysis size (default: as in training; phone photos at 4032 px also suit 4)."""
    ds_ = downsample or model.info.get("downsample", DOWNSAMPLE)
    recs, _ = features_many(list(paths), None, ds_, log, progress, should_stop)
    recs = apply_qc(recs, qc_mode)
    if "cnn" in model.feature_set:
        add_embeddings(recs, ds_, log, progress, should_stop)
    out = []
    for p, r in zip(paths, recs):
        if r is None or r.scalars is None or (not r.qc_ok and not predict_failed):
            out.append(Prediction(p, "", 0.0, [], 0.0, False, r.qc_reason if r else "could not read"))
            continue
        if "cnn" in model.feature_set and r.cnn is None:
            out.append(Prediction(p, "", 0.0, [], 0.0, False, "no CNN embedding"))
            continue
        X, _ = feature_matrix([r], model.feature_set, model.normalised)
        pr = model.proba(X)[0]
        o = np.argsort(-pr)
        tops = [(model.classes[i], float(pr[i])) for i in o[:top]]
        thr = model.unknown_threshold(1)
        out.append(Prediction(p, tops[0][0], tops[0][1], tops,
                              float(pr[o[0]] - (pr[o[1]] if len(o) > 1 else 0)), r.qc_ok, r.qc_reason,
                              bool(thr is not None and 1 - pr[o[0]] > thr)))
    return out


def predict_sample(model, paths, top=3, log=print, progress=None, should_stop=lambda: False,
                   qc_mode="microscope", downsample=None):
    """Judge several droplets of ONE sample together: average the class probabilities of
    the QC-passed droplets. Far more reliable than one droplet, both for the whisky and
    for spotting a sample unlike any trained whisky (Nov 2026 test, 6 trained whiskies:
    an unseen whisky was caught 53% of the time from 1 droplet, 92-94% from 3-5)."""
    singles = predict(model, paths, top, log, progress, should_stop, predict_failed=False,
                      qc_mode=qc_mode, downsample=downsample)
    good = [p for p in singles if p.qc_ok and p.top]
    if not good:
        return SamplePrediction("", 0.0, [], 0.0, 0, len(singles), 0.0, False, 1.0, None, singles)
    ds_ = downsample or model.info.get("downsample", DOWNSAMPLE)
    recs = [features(p.path, None, ds_) for p in good]
    if "cnn" in model.feature_set:
        add_embeddings(recs, ds_, log)
    X, _ = feature_matrix(recs, model.feature_set, model.normalised)
    pm = model.proba(X).mean(0)
    o = np.argsort(-pm)
    tops = [(model.classes[i], float(pm[i])) for i in o[:top]]
    thr = model.unknown_threshold(len(good))
    score = float(1 - pm[o[0]])
    return SamplePrediction(
        tops[0][0], tops[0][1], tops, float(pm[o[0]] - (pm[o[1]] if len(o) > 1 else 0)),
        len(good), len(singles), float(np.mean([p.label == tops[0][0] for p in good])),
        bool(thr is not None and score > thr), score, thr, singles)


# --------------------------------------------------------------------------- #
#  Figures (matplotlib, optional) -> PIL images
# --------------------------------------------------------------------------- #
def _plt():
    if not figures_available():
        raise RuntimeError("figures need matplotlib: pip install matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _to_pil(fig, dpi=150):
    from PIL import Image
    plt = _plt()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _title(name_of, label, extra=""):
    nm = name_of(label) if callable(name_of) else (name_of or {}).get(label, "")
    return f"{label}{' · ' + nm if nm else ''}{extra}"


_PROFILE_LABELS = {"mean_L": "mean intensity", "std_L": "angular std (heterogeneity)",
                   "hue_ratio": "colour (R−B)/(R+B)", "p95_L": "95th percentile (particles)",
                   "mean_R": "mean red", "mean_G": "mean green", "mean_B": "mean blue"}


def figure_pipeline(droplets, names=None, dpi=150):
    """Rows of: stitched | ring map + rim | normalised | unwrap (normalised) | radial profile."""
    plt = _plt()
    ds = [d for d in droplets if d.seg is not None]
    if not ds:
        raise ValueError("none of the selected images has a detectable droplet")
    fig, ax = plt.subplots(len(ds), 5, figsize=(19, 3.7 * len(ds)), squeeze=False,
                           gridspec_kw=dict(width_ratios=[1, 1, 1, 1.6, 1.2]))
    r = np.linspace(0, R_MAX, N_R)
    for i, d in enumerate(ds):
        ax[i, 0].imshow(view_raw(d))
        ax[i, 0].set_ylabel(_title(names, d.label, f"\n{d.name}") + ("" if d.qc_ok else "\n(failed QC)"),
                            fontsize=9)
        ax[i, 1].imshow(view_ring_map(d))
        ax[i, 2].imshow(view_normalised(d))
        ax[i, 3].imshow(_show(d.polar_norm, max_gain=1.6), aspect="auto", origin="lower",
                        extent=[0, 360, 0, R_MAX])
        ax[i, 3].axhline(1, color="w", lw=0.6, ls="--")
        ax[i, 3].set_xlabel("angle (°)"); ax[i, 3].set_ylabel("r / R")
        for k, c in zip(("mean_R", "mean_G", "mean_B"), ("tab:red", "tab:green", "tab:blue")):
            ax[i, 4].plot(r, d.profiles[k], color=c, lw=1.3, label=k[-1])
        ax[i, 4].plot(r, d.profiles["std_L"], "k--", lw=1, label="angular std")
        ax[i, 4].axvline(1, color="grey", lw=0.6, ls=":")
        ax[i, 4].set_xlabel("r / R")
        for j in range(3):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[0, 4].legend(fontsize=8)
    for j, t in enumerate(["1. Stitched (¼ size, auto gain)", "2. Ring map + fitted rim", "3. Normalised",
                           "4. Radial unwrap (normalised)", "5. Radial profile (raw)"]):
        ax[0, j].set_title(t, fontsize=11)
    fig.tight_layout()
    return _to_pil(fig, dpi)


def figure_normalisation(droplets, names=None, dpi=150):
    """Rows of: raw | background region | background histograms | subtracted |
    droplet means before/after | normalised."""
    plt = _plt()
    ds = [d for d in droplets if d.seg is not None]
    if not ds:
        raise ValueError("none of the selected images has a detectable droplet")
    cols = ("tab:red", "tab:green", "tab:blue")
    # columns 1 and 4 share ONE gain across rows so brightness can be compared between
    # droplets; it is the smallest per-image auto gain, so no row is blown out
    g = min(raw_gain(d) for d in ds)
    fig, ax = plt.subplots(len(ds), 6, figsize=(24, 4.2 * len(ds)), squeeze=False,
                           gridspec_kw=dict(width_ratios=[1, 1, 1.1, 1, 0.9, 1]))
    for i, d in enumerate(ds):
        info = d.norm_info
        x = d.rgb.astype(np.float32) / 255
        ax[i, 0].imshow(view_raw(d, g))
        ax[i, 0].set_ylabel(_title(names, d.label, f"\n{d.name}"), fontsize=10)
        ax[i, 1].imshow(view_background(d, g))
        for c in range(3):
            ax[i, 2].hist(x[..., c][info["bg_mask"]] * 255, bins=np.arange(0, 80, 1), histtype="step",
                          color=cols[c], lw=1.3)
            ax[i, 2].axvline(info["bg"][c] * 255, color=cols[c], ls="--", lw=1)
        ax[i, 2].set_xlabel("pixel value (0–255)"); ax[i, 2].set_ylabel("pixels")
        ax[i, 2].text(0.97, 0.95, "median R/G/B\n" + " / ".join(f"{b * 255:.0f}" for b in info["bg"]),
                      transform=ax[i, 2].transAxes, ha="right", va="top", fontsize=9)
        ax[i, 3].imshow(view_subtracted(d, g))
        nf = _norm_float(d) * 255                   # unclipped, so the means hit the target
        mean_after = np.array([nf[..., c][d.seg["mask"].astype(bool)].mean() for c in range(3)])
        xs = np.arange(3)
        ax[i, 4].bar(xs - 0.27, info["mean_raw"] * 255, 0.27, color=cols, alpha=0.35, label="raw")
        ax[i, 4].bar(xs, info["mean_sub"] * 255, 0.27, color=cols, alpha=0.65, label="− background")
        ax[i, 4].bar(xs + 0.27, mean_after, 0.27, color=cols, edgecolor="k", label="÷ droplet mean")
        ax[i, 4].set_xticks(xs, ["R", "G", "B"]); ax[i, 4].set_ylabel("mean inside droplet (0–255)")
        ax[i, 4].text(0.02, 0.97, "gain ×" + " / ".join(f"{g:.1f}" for g in info["gain"]),
                      transform=ax[i, 4].transAxes, va="top", fontsize=9)
        ax[i, 5].imshow(view_normalised(d))
        for j in (0, 1, 3, 5):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[0, 4].legend(fontsize=7, loc="upper right")
    for j, t in enumerate([f"1. Raw (display ×{g:.1f}, same for all rows)", "2. Background (blue): beyond 1.25 R",
                           "3. Background histogram (dashed = median)", f"4. Background subtracted (×{g:.1f})",
                           "5. Droplet mean: raw → −bg → ÷ own mean", "6. Normalised (auto gain)"]):
        ax[0, j].set_title(t, fontsize=10)
    fig.tight_layout()
    return _to_pil(fig, dpi)


def figure_profiles(items, names=None, keys=("mean_L", "std_L", "hue_ratio"), normalised=False, dpi=150):
    """Radial profiles of several droplets (Droplet/Record) and/or sample averages
    (SampleSummary, drawn as mean ± SD band), one panel per profile key."""
    plt = _plt()
    fig, ax = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.3), squeeze=False)
    ax = ax[0]
    cmap = plt.get_cmap("tab20")
    r = np.linspace(0, R_MAX, N_R)
    for i, it in enumerate(items):
        col = cmap(i % 20)
        if isinstance(it, SampleSummary):
            m = it.profile_mean_norm if normalised else it.profile_mean
            s = it.profile_sd_norm if normalised else it.profile_sd
            lab = _title(names, it.label, f" (n={it.n_used})")
        else:
            m = it.profiles_norm if normalised else it.profiles
            s, lab = None, _title(names, it.label, f" {it.name}")
        if m is None:
            continue
        for j, k in enumerate(keys):
            ax[j].plot(r, m[k], color=col, lw=1.4, label=lab if j == len(keys) - 1 else None)
            if s is not None:
                ax[j].fill_between(r, m[k] - s[k], m[k] + s[k], color=col, alpha=0.15, lw=0)
    for j, k in enumerate(keys):
        ax[j].axvline(1, color="k", lw=0.6, ls="--")
        ax[j].set_title(_PROFILE_LABELS.get(k, k) + (" (normalised)" if normalised else ""))
        ax[j].set_xlabel("r / R (1 = rim)")
    ax[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return _to_pil(fig, dpi)


def figure_sample_average(summary, names=None, dpi=150):
    """One sample: average droplet (raw, normalised), mean unwrap, mean ± SD profiles."""
    plt = _plt()
    s = summary
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.7, 1.2, 1.2]))
    ax[0].imshow(display(s.mean_droplet)); ax[0].set_title("average droplet (raw)")
    ax[1].imshow(display(s.mean_droplet_norm, max_gain=1.6)); ax[1].set_title("average droplet (normalised)")
    ax[2].imshow(display(s.mean_unwrap_norm, max_gain=1.6), aspect="auto", origin="lower", extent=[0, 360, 0, R_MAX])
    ax[2].axhline(1, color="w", lw=0.6, ls="--")
    ax[2].set_title("mean unwrap (normalised)"); ax[2].set_xlabel("angle (°)"); ax[2].set_ylabel("r / R")
    for k, c in zip(("mean_R", "mean_G", "mean_B"), ("tab:red", "tab:green", "tab:blue")):
        ax[3].plot(s.r, s.profile_mean[k], color=c, lw=1.4, label=k[-1])
        ax[3].fill_between(s.r, s.profile_mean[k] - s.profile_sd[k], s.profile_mean[k] + s.profile_sd[k],
                           color=c, alpha=0.15, lw=0)
    ax[3].set_title("mean ± SD profile (raw)"); ax[3].legend(fontsize=8)
    for k, c, lab in (("std_L", "k", "angular std"), ("hue_ratio", "tab:purple", "(R−B)/(R+B)")):
        ax[4].plot(s.r, s.profile_mean[k], color=c, lw=1.4, label=lab)
        ax[4].fill_between(s.r, s.profile_mean[k] - s.profile_sd[k], s.profile_mean[k] + s.profile_sd[k],
                           color=c, alpha=0.15, lw=0)
    ax[4].set_title("heterogeneity and colour"); ax[4].legend(fontsize=8)
    for a in ax[3:]:
        a.axvline(1, color="grey", lw=0.6, ls=":"); a.set_xlabel("r / R")
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(_title(names, s.label, f": {s.n_used} of {s.n_images} droplets passed QC"), fontsize=12)
    fig.tight_layout()
    return _to_pil(fig, dpi)


def figure_confusion(result, names=None, dpi=150):
    """Row-normalised confusion matrix of a CVResult."""
    plt = _plt()
    cm = result.confusion
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(result.classes)
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * n + 3), max(4.5, 0.45 * n + 2.5)))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), result.classes)
    ax.set_yticks(range(n), [_title(names, c) for c in result.classes], fontsize=8)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im, fraction=0.04, label="fraction of true class")
    ax.set_title(f"{result.feature_set}{' (normalised)' if result.normalised else ''}: "
                 f"balanced accuracy {result.balanced_accuracy:.2f}")
    fig.tight_layout()
    return _to_pil(fig, dpi)


def figure_cv_comparison(results, dpi=150):
    """Bar chart of balanced accuracy for several CVResults (e.g. feature sets)."""
    plt = _plt()
    labs = [f"{r.feature_set}{' (norm.)' if r.normalised else ''}" for r in results]
    vals = [r.balanced_accuracy for r in results]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(results) + 1.8))
    ax.barh(labs, vals, color=["#bbbbbb" if r.feature_set == "nuisance" else "#3b6ea8" for r in results])
    if results:
        ax.axvline(results[0].chance, color="r", ls="--", lw=1, label=f"chance ({results[0].chance:.3f})")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("balanced accuracy (cross-validation)"); ax.legend(loc="lower right")
    fig.tight_layout()
    return _to_pil(fig, dpi)


def figure_training(report, names=None, top=15, dpi=150):
    """Inside one training run (from training_report): PCA explained variance, the
    droplets on PC1/PC2, loss and accuracy per step, and the features the model uses."""
    plt = _plt()
    r = report
    fig = plt.figure(figsize=(19, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 1.1])
    cmap = plt.get_cmap("tab20")

    ax = fig.add_subplot(gs[0, 0])                       # 1. PCA explained variance
    ev = r["explained"]
    show = min(len(ev), max(r["kept"] + 10, 30))
    xs = np.arange(1, show + 1)
    ax.bar(xs, ev[:show] * 100, color="#9db7d5", label="each component")
    ax2 = ax.twinx()
    ax2.plot(xs, np.cumsum(ev)[:show] * 100, "k-", lw=1.3, label="cumulative")
    ax2.axvline(r["kept"] + 0.5, color="tab:red", ls="--", lw=1)
    ax2.text(r["kept"] - 0.01 * show, 8, f"kept {r['kept']}\n({100 * np.sum(ev[:r['kept']]):.0f}% of variance)", color="tab:red",
             fontsize=9, ha="right")
    ax2.set_ylim(0, 101); ax2.set_ylabel("cumulative variance (%)")
    ax.set_xlabel("principal component"); ax.set_ylabel("variance explained (%)")
    ax.set_title(f"1. PCA: {r['n_features']} features → {r['kept']} components")

    ax = fig.add_subplot(gs[0, 1])                       # 2. droplets on PC1 / PC2
    sc, lab, te = r["scores"], r["labels"], r["held_out"]
    for i, c in enumerate(r["classes"]):
        sel = lab == c
        ax.scatter(sc[sel & ~te, 0], sc[sel & ~te, 1], s=10, color=cmap(i % 20), alpha=0.55,
                   label=_title(names, c))
        ax.scatter(sc[sel & te, 0], sc[sel & te, 1], s=22, color=cmap(i % 20), marker="x", lw=0.9)
    pc = ev[:2] * 100
    ax.set_xlabel(f"PC1 ({pc[0]:.0f}% of variance)"); ax.set_ylabel(f"PC2 ({pc[1]:.0f}%)")
    ax.set_title("2. Droplets on the first two components (× = held out)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1 + len(r["classes"]) // 18)

    ax = fig.add_subplot(gs[0, 2])                       # 3. features the model leans on
    w = r["feature_weight"]
    o = np.argsort(w)[::-1][:top][::-1]
    ax.barh([r["feature_names"][i] for i in o], w[o], color="#3b6ea8")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("weight in the model (standardised features)")
    ax.set_title(f"3. Top {top} features used")

    h = r["history"]
    steps = [e["step"] for e in h]
    ax = fig.add_subplot(gs[1, 0])                       # 4. loss per step
    ax.plot(steps, [e["loss"] for e in h], color="k", lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("training step (one full pass over the data = one epoch)")
    ax.set_ylabel("training loss (cross-entropy + L2)")
    ax.set_title(f"4. Loss per step ({steps[-1]} steps"
                 f"{', stopped early: converged' if steps[-1] < 3000 else ''})")

    ax = fig.add_subplot(gs[1, 1:])                      # 5. accuracy per step
    ax.plot(steps, [e["train_acc"] for e in h], color="#3b6ea8", lw=1.5,
            label=f"training droplets ({r['n_train']})")
    if h and h[0]["val_acc"] is not None:
        va = [e["val_acc"] for e in h]
        ax.plot(steps, va, color="tab:orange", lw=1.5, label=f"held-out droplets ({r['n_test']})")
        best = int(np.argmax(va))
        ax.plot(steps[best], va[best], "o", color="tab:orange")
        ax.annotate(f"best held-out {va[best]:.2f} at step {steps[best]}\nfinal {va[-1]:.2f}",
                    (steps[best], va[best]), xytext=(12, -32), textcoords="offset points", fontsize=9)
    ax.axhline(1 / len(r["classes"]), color="tab:red", ls="--", lw=1, label="chance")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("training step (log scale)"); ax.set_ylabel("accuracy")
    ax.set_title("5. Accuracy per step: a gap between the lines = overfitting")
    ax.legend(loc="lower right")

    fig.suptitle(f"Inside one training run · {r['feature_set']}{' (normalised)' if r['normalised'] else ''} · "
                 f"{len(r['classes'])} whiskies", fontsize=13)
    fig.tight_layout()
    return _to_pil(fig, dpi)


# --------------------------------------------------------------------------- #
#  Explanations: which parts of the droplet drive the prediction
# --------------------------------------------------------------------------- #
# The classifier is linear in the standardised features (PCA + logistic regression),
# so a droplet's score for a whisky splits EXACTLY into one term per feature:
#     logit_c - mean_k logit_k = sum_j  x_j * w_jc      (x = standardised feature)
# Each feature is a statistic of a known image region (a ring at r/R, the rim band,
# the particles, the centre...), so each term is spread over the pixels that feature
# was computed from, weighted by how much each pixel enters the statistic. Summed,
# that is a Grad-CAM-like map, but exact for these features (not an approximation).
# CNN feature sets are not covered (that needs gradients through the network).

EXPLAIN_ZONES = (("centre", 0.0, 0.35), ("interior", 0.35, 0.85), ("rim", 0.85, 1.10),
                 ("outside", 1.10, R_MAX + 1))
_PROFILE_WORDS = {"mean_R": "red", "mean_G": "green", "mean_B": "blue", "std_L": "patchiness",
                  "p95_L": "bright particles", "hue_ratio": "colour (R-B)/(R+B)"}
_SCALAR_GROUP = {
    "radius_px": "droplet size", "ovality": "droplet shape",
    "ring_score": "coffee ring", "ring_strength": "coffee ring", "ring_peak_r": "coffee ring",
    "ring_peak_int": "coffee ring", "ring_contrast": "coffee ring", "ring_width": "coffee ring",
    "ring_continuity": "coffee ring", "hue_rim": "rim colour",
    "in_mean_R": "interior colour", "in_mean_G": "interior colour", "in_mean_B": "interior colour",
    "in_std_R": "interior contrast", "in_std_G": "interior contrast", "in_std_B": "interior contrast",
    "in_p50": "interior brightness", "in_p90": "interior brightness", "in_p99": "bright particles",
    "particle_density": "bright particles", "particle_mean_int": "bright particles",
    "centre_int": "centre deposit", "centre_to_mid": "centre deposit", "hue_centre": "centre colour",
    "azim_hetero": "interior patchiness", "grad_energy": "interior texture",
    "tex_fine": "interior texture", "tex_mid": "interior texture", "tex_coarse": "interior texture",
    "tex_concentric": "concentric rings",
    **{f"{k}_{z}": f"{z} lopsidedness" for z in ("centre", "interior", "rim") for k in ("asym1", "asym2")},
    **{f"hue_asym_{z}": f"{z} colour lopsidedness" for z in ("centre", "interior", "rim")}}


@dataclass
class Explanation:
    target: str                              # whisky explained
    versus: str                              # "average whisky" or the comparison label
    probability: float
    top: list                                # [(label, prob)] best first
    features: list                           # [(name, contribution)] sorted by |contribution|
    groups: list                             # [(group name, contribution)] sorted by |contribution|
    group_maps: dict                         # group -> HxW map (its contribution spread over pixels)
    heat: np.ndarray                         # HxW signed map: sum of all contributions (+ = for target)
    image: np.ndarray                        # HxWx3 uint8 display image the maps line up with
    total: float                             # sum of contributions = logit gap explained
    cnn_total: float = 0.0                   # combined models: the CNN embedding's share of `total`
                                             # (not mapped here; see cnn_map). Features part = total - cnn_total


def _pixel_geometry(d):
    """r/R and the radial-bin index of every pixel (r/R = 1 on the traced rim)."""
    s = d.seg
    H, W = d.rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - s["cx"], yy - s["cy"]
    ang = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    rim = np.interp(ang, np.linspace(0, 2 * np.pi, N_T, endpoint=False), s["rim"], period=2 * np.pi)
    rr = np.hypot(dx, dy) / np.maximum(rim, 1e-6)
    k = np.clip(np.round(rr / R_MAX * (N_R - 1)).astype(int), 0, N_R)       # N_R = beyond the unwrap
    return rr, np.where(k < N_R, k // (N_R // N_BINS), -1)


def _feature_pixel_weights(d, normalised):
    """A function name -> (unnormalised, >= 0) pixel weights of that feature: how much
    each pixel enters the statistic the feature is."""
    cv2 = _cv2()
    img = np.asarray(_norm_float(d) if normalised else d.rgb.astype(np.float32) / 255, np.float32)
    canvas = d.rgb.max(2) > 0
    L = img.mean(2)
    g = img.max(2)
    rr, rbin = _pixel_geometry(d)
    inside = d.seg["mask"].astype(bool)
    rim_band = canvas & (rr > 0.85) & (rr < 1.10)
    ringw = d.ring * rim_band
    tophat = g - cv2.GaussianBlur(g, (0, 0), 6)
    peaks = (tophat > 0.08) & (g == cv2.dilate(g, np.ones((3, 3), np.uint8))) & inside
    peaks = cv2.dilate(peaks.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2) * inside
    hue = np.abs(img[..., 0] - img[..., 2])
    cidx = {"R": 0, "G": 1, "B": 2}
    zero = np.zeros_like(L)

    def dev(x, region):
        return np.abs(x - x[region].mean()) * region if region.any() else zero

    def above(x, region, q):
        return ((x >= np.percentile(x[region], q)) & region).astype(np.float32) if region.any() else zero

    def radial(key, j):
        a = canvas & (rbin == j)
        if key in ("mean_R", "mean_G", "mean_B"):
            return img[..., cidx[key[-1]]] * a
        if key == "std_L":
            return dev(L, a)
        if key == "p95_L":
            return above(L, a, 95)
        return (hue + 1e-3) * a                                   # hue_ratio

    def scalar(name):
        if name in ("in_mean_R", "in_mean_G", "in_mean_B"):
            return img[..., cidx[name[-1]]] * inside
        if name in ("in_std_R", "in_std_G", "in_std_B"):
            return dev(img[..., cidx[name[-1]]], inside)
        if name == "in_p50":
            return inside.astype(np.float32)
        if name == "in_p90":
            return above(g, inside, 90)
        if name == "in_p99":
            return above(g, inside, 99)
        if name in ("particle_density", "particle_mean_int"):
            return peaks
        if name == "centre_int":
            return (canvas & (rr < 0.5)).astype(np.float32)
        if name == "hue_centre":
            return (hue + 1e-3) * (canvas & (rr < 0.5))
        if name == "centre_to_mid":
            return (canvas & ((rr < 0.2) | ((rr > 0.4) & (rr < 0.7)))).astype(np.float32)
        if name == "ring_contrast":
            c = (canvas & (rr < 0.5)).astype(np.float32)
            return ringw / max(float(ringw.sum()), 1e-6) + c / max(float(c.sum()), 1e-6)
        if name == "hue_rim":
            return (hue + 1e-3) * rim_band
        if name == "azim_hetero":
            out = np.zeros_like(L)
            for j in range(N_BINS):
                a = canvas & (rbin == j) & (rr < 1)
                if a.any():
                    out += dev(L, a)
            return out
        if name == "grad_energy" or name.startswith("tex_"):
            return grad
        if name.startswith(("asym", "hue_asym")):
            z = name.rsplit("_", 1)[1]
            lo, hi = next((a, b) for zz, a, b in ASYM_ZONES if zz == z)
            zone = canvas & (rr >= lo) & (rr < hi)
            x = hue if name.startswith("hue") else L
            return dev(x, zone)                                   # where the zone departs from its mean
        return ringw                                              # rim-defined: size, shape, ring_*

    def weights(name):
        if "@r" in name:
            key, r = name.split("@r")
            j = min(N_BINS - 1, int(float(r) / R_MAX * N_BINS))
            return radial(key, j)
        return scalar(name)
    return weights


def _group_of(name):
    if "@r" in name:
        key, r = name.split("@r")
        r = float(r)
        zone = next(z for z, lo, hi in EXPLAIN_ZONES if lo <= r < hi)
        return f"{zone} {_PROFILE_WORDS.get(key, key)}"
    return _SCALAR_GROUP.get(name, name)


def explain(model, d, target=None, versus=None, smooth=1.5):
    """Why `model` scores droplet `d` (from analyse()) as `target` (default: the predicted
    whisky). versus: None = against the average of the other whiskies, or a label (e.g. the
    runner-up) for "why target rather than versus". Returns an Explanation; contributions
    are in logit units and sum exactly to the explained logit gap."""
    if model.feature_set in ("cnn", "nuisance"):
        raise ValueError("explanations cover the scalar / radial feature sets and cnn+scalar+radial "
                         "(use cnn_map for a CNN-only model)")
    if d.seg is None:
        raise ValueError("no droplet found in this image")
    cv2 = _cv2()
    rec = record_from_droplet(d)
    if "cnn" in model.feature_set:                     # combined: attach the (cached) embedding
        add_embeddings([rec], (model.info or {}).get("downsample", DOWNSAMPLE), log=lambda *_: None)
        if rec.cnn is None:
            raise ValueError("no CNN embedding for this droplet")
    X, names = feature_matrix([rec], model.feature_set, model.normalised)
    p = model.proba(X)[0]
    order = np.argsort(-p)
    ti = model.classes.index(target) if target else int(order[0])
    Weff = model.components.T.astype(np.float64) @ model.W.astype(np.float64)        # d x K
    if versus is None:
        w = Weff[:, ti] - np.delete(Weff, ti, axis=1).mean(1)
        vs = "average whisky"
    else:
        w, vs = Weff[:, ti] - Weff[:, model.classes.index(versus)], versus
    contrib = (X[0].astype(np.float64) - model.mean) / model.scale * w
    is_cnn = np.array([n.startswith("cnn_") for n in names])
    cnn_total = float(contrib[is_cnn].sum())
    weights = _feature_pixel_weights(d, model.normalised)
    H, Wd = d.rgb.shape[:2]
    heat = np.zeros((H, Wd), np.float64)
    gmaps, gsum = {}, {}
    for name, c, cn in zip(names, contrib, is_cnn):
        if cn or abs(c) < 1e-9:                        # CNN terms have no pixel footprint here
            continue
        m = weights(name).astype(np.float64)
        s = m.sum()
        if s <= 0:
            continue
        spread = c * m / s
        heat += spread
        gname = _group_of(name)
        gmaps[gname] = gmaps.get(gname, 0) + spread
        gsum[gname] = gsum.get(gname, 0.0) + float(c)
    if smooth:
        sig = max(1.0, smooth * max(H, Wd) / 300)
        heat = cv2.GaussianBlur(heat, (0, 0), sig)
        gmaps = {g: cv2.GaussianBlur(m, (0, 0), sig) for g, m in gmaps.items()}
    feats = sorted(((n, c) for n, c, cn in zip(names, contrib.tolist(), is_cnn) if not cn),
                   key=lambda t: -abs(t[1]))
    groups = sorted(gsum.items(), key=lambda t: -abs(t[1]))
    img = view_normalised(d) if model.normalised else view_raw(d)
    return Explanation(model.classes[ti], vs, float(p[ti]),
                       [(model.classes[i], float(p[i])) for i in order[:3]],
                       feats, groups, {g: m.astype(np.float32) for g, m in gmaps.items()},
                       heat.astype(np.float32), img, float(contrib.sum()), cnn_total)


EXPLAIN_COLOURS = [(255, 80, 80), (80, 200, 255), (255, 210, 60), (120, 255, 120),
                   (230, 120, 255), (255, 150, 40), (60, 255, 220), (255, 120, 190)]


def view_explain_heat(e, alpha=0.7):
    """Signed heat map over the droplet: red = pixels pushing towards e.target,
    blue = pushing away (towards the comparison). Grey image underneath."""
    base = e.image.astype(np.float32).mean(2, keepdims=True).repeat(3, 2) * 0.6
    # square-root scale: evidence spread over a whole ring is thin per pixel next to
    # compact features (particles), so a linear scale hides it
    h = e.heat / (np.percentile(np.abs(e.heat), 99.5) + 1e-12)
    h = (np.sign(h) * np.sqrt(np.clip(np.abs(h), 0, 1)))[..., None]
    red, blue = np.array([255, 40, 30], np.float32), np.array([40, 120, 255], np.float32)
    a = alpha * np.abs(h)
    col = np.where(h > 0, red, blue)
    return np.clip(base * (1 - a) + col * a, 0, 255).astype(np.uint8)


def view_explain_groups(e, k=6, sign=+1, alpha=0.75):
    """The k feature groups contributing most towards (sign=+1) or against (sign=-1)
    e.target, each in its own colour (EXPLAIN_COLOURS, in rank order) over a dimmed image.
    Returns (image, [(group, contribution, colour)]) for a legend."""
    base = e.image.astype(np.float32) * 0.45
    picked = [(g, c) for g, c in e.groups if np.sign(c) == sign][:k]
    out, legend = base.copy(), []
    for i, (g, c) in enumerate(picked):
        m = np.abs(e.group_maps[g])
        if (m > 0).any():
            m = np.clip(m / (np.percentile(m[m > 0], 99) + 1e-12), 0, 1)
        col = np.array(EXPLAIN_COLOURS[i % len(EXPLAIN_COLOURS)], np.float32)
        a = (alpha * m)[..., None]
        out = out * (1 - a) + col * a
        legend.append((g, c, EXPLAIN_COLOURS[i % len(EXPLAIN_COLOURS)]))
    return np.clip(out, 0, 255).astype(np.uint8), legend


def figure_explain(explanations, names=None, k=6, dpi=150):
    """One row per Explanation: image | signed heat map | top-k groups FOR the whisky |
    top-k AGAINST | bar chart of those groups (colours match the overlays)."""
    plt = _plt()
    n = len(explanations)
    fig, ax = plt.subplots(n, 5, figsize=(25, 4.7 * n), squeeze=False,
                           gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.4]))
    for i, e in enumerate(explanations):
        ax[i, 0].imshow(e.image)
        ax[i, 0].set_ylabel(_title(names, e.target, f"\np = {e.probability:.2f}"), fontsize=10)
        ax[i, 0].set_xlabel("predicted: " + ", ".join(f"{l} {p:.2f}" for l, p in e.top), fontsize=8)
        ax[i, 1].imshow(view_explain_heat(e))
        pos, lp = view_explain_groups(e, k, +1)
        neg, ln = view_explain_groups(e, k, -1)
        ax[i, 2].imshow(pos)
        ax[i, 3].imshow(neg)
        for j in range(4):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        bars = lp[::-1] + ln
        ys = np.arange(len(bars))
        ax[i, 4].barh(ys, [c for _, c, _ in bars], color=[np.array(col) / 255 for _, _, col in bars],
                      edgecolor="k", lw=0.4)
        ax[i, 4].set_yticks(ys, [g for g, _, _ in bars], fontsize=8)
        ax[i, 4].axvline(0, color="k", lw=0.6)
        ax[i, 4].set_xlabel(f"contribution to {e.target} vs {e.versus} (logit units)"
                            + (f"\nfeatures {e.total - e.cnn_total:+.2f} · CNN part {e.cnn_total:+.2f} (not mapped)"
                               if e.cnn_total else ""), fontsize=8)
    for j, t in enumerate(["image", "evidence map (red = for, blue = against)",
                           f"top {k} feature groups FOR", f"top {k} feature groups AGAINST",
                           "ranked feature groups"]):
        ax[0, j].set_title(t, fontsize=10)
    fig.tight_layout()
    return _to_pil(fig, dpi)


# --------------------------------------------------------------------------- #
#  CNN class-activation maps (exact for this model)
# --------------------------------------------------------------------------- #
# The CNN embedding is the global average of EfficientNet's last feature map (1280 ×
# 8 × 8 for a 256 px crop), averaged over 4 rotations × 2 flips, and the classifier is
# linear in it. So the CNN part of a droplet's score is EXACTLY the sum, over the 8 × 8
# cells of each view, of Σ_k w_k A_k(cell) / (64 · 8): a CAM, not a Grad-CAM heuristic.
# It is coarse (each cell ≈ 1/8 of the crop width) and says WHERE, not WHAT.

@dataclass
class CNNMap:
    target: str
    versus: str
    probability: float
    top: list
    cells: np.ndarray                        # 8 × 8 signed contribution per cell (crop frame)
    crop: np.ndarray                         # 256 × 256 × 3 uint8 display crop the map lines up with
    total: float                             # Σ cells (= CNN-part logit gap minus a constant offset)


def cnn_map(model, path, target=None, versus=None, downsample=DOWNSAMPLE):
    """Where in the droplet the CNN evidence for `target` (default: predicted) sits.
    Works for models with "cnn" in feature_set (uses the CNN part of the features)."""
    import torch
    if "cnn" not in model.feature_set:
        raise ValueError("cnn_map needs a model trained on a CNN feature set")
    rec = features(path, None, downsample)
    add_embeddings([rec], downsample, log=lambda *_: None)
    X, names = feature_matrix([rec], model.feature_set, model.normalised)
    p = model.proba(X)[0]
    order = np.argsort(-p)
    ti = model.classes.index(target) if target else int(order[0])
    Weff = model.components.T.astype(np.float64) @ model.W.astype(np.float64)
    if versus is None:
        w = Weff[:, ti] - np.delete(Weff, ti, axis=1).mean(1)
        vs = "average whisky"
    else:
        w, vs = Weff[:, ti] - Weff[:, model.classes.index(versus)], versus
    idx = [i for i, n in enumerate(names) if n.startswith("cnn_")]
    alpha = torch.tensor((w[idx] / model.scale[idx]).astype(np.float32))
    crops = _cnn_crops(path, downsample)
    if crops is None:
        raise ValueError("no droplet found in this image")
    cr = crops[1] if model.normalised else crops[0]
    gain = 1.0 if model.normalised else 2.5
    net, mean, std = _cnn_net()
    x = torch.clamp(torch.from_numpy(cr.copy()).permute(2, 0, 1)[None].float() / 255 * gain, 0, 1)
    cells = 0
    with torch.inference_mode():
        for k in range(4):
            v = torch.rot90(x, k, (2, 3))
            for flip in (False, True):
                vv = torch.flip(v, (3,)) if flip else v
                A = net.features((vv - mean) / std)[0]                 # 1280 × h × w
                m = torch.einsum("k,khw->hw", alpha, A) / (A.shape[1] * A.shape[2] * 8)
                if flip:
                    m = torch.flip(m, (1,))
                cells = cells + torch.rot90(m, -k, (0, 1))
    cells = cells.numpy().astype(np.float32)
    disp = _show(cr, 1.0) if model.normalised else _show(cr)
    return CNNMap(model.classes[ti], vs, float(p[ti]), [(model.classes[i], float(p[i])) for i in order[:3]],
                  cells, disp, float(cells.sum()))


def view_cnn_map(cm, alpha=0.65):
    """CAM over the droplet crop: red = CNN evidence for cm.target, blue = against.
    Upsampled smoothly from the 8 × 8 cells (the true resolution: don't over-read it)."""
    cv2 = _cv2()
    S = cm.crop.shape[0]
    h = cv2.resize(cm.cells, (S, S), interpolation=cv2.INTER_CUBIC)
    h = np.clip(h / (np.abs(cm.cells).max() + 1e-12), -1, 1)[..., None]
    base = cm.crop.astype(np.float32).mean(2, keepdims=True).repeat(3, 2) * 0.6
    col = np.where(h > 0, np.array([255, 40, 30], np.float32), np.array([40, 120, 255], np.float32))
    a = alpha * np.abs(h)
    return np.clip(base * (1 - a) + col * a, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Average explanations over many droplets (a whole sample / whisky)
# --------------------------------------------------------------------------- #
# Each droplet's maps are cut out with the same droplet-centred crop as the CNN (side
# 2.4 × mean rim radius), so every droplet lands in one common frame: centre in the
# middle, rim on the same circle, whatever its size. Rotations are random, so the
# average is close to radially symmetric; the radial evidence profile (evidence vs r/R)
# is the most readable summary and is computed exactly from the per-pixel maps.

AVG_SIZE = 256


@dataclass
class AverageExplanation:
    target: str
    versus: str
    kind: str                                # "features" or "cnn"
    n: int                                   # droplets used
    skipped: list                            # [(name, reason)]
    mean_image: np.ndarray                   # AVG_SIZE² × 3 uint8: the average droplet (display)
    heat: np.ndarray                         # AVG_SIZE² signed mean evidence (+ = for target)
    heat_sd: np.ndarray                      # AVG_SIZE² SD over droplets (how consistent)
    radial_r: np.ndarray                     # r/R bins
    radial: np.ndarray                       # mean evidence per unit area at r/R (+ = for target)
    radial_sd: np.ndarray
    groups: list                             # [(group, mean contribution, SD)] ranked by |mean|  (features only)
    group_maps: dict                         # group -> AVG_SIZE² mean map                       (features only)
    probability: float                       # mean probability of target over the droplets
    accuracy: float                          # share of droplets predicted as target
    features_part: float = 0.0               # mean (over droplets) of the mapped features' total
    features_part_sd: float = 0.0
    cnn_part: float = 0.0                    # combined models: mean CNN share of the score (not mapped)
    cnn_part_sd: float = 0.0


def _crop_map(m, seg, size=AVG_SIZE):
    """crop() for a float map (same geometry, zero outside the image)."""
    cv2 = _cv2()
    half = max(2, int(1.2 * float(np.mean(seg["rim"]))))
    big = cv2.copyMakeBorder(np.asarray(m, np.float32), half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    x0, y0 = int(seg["cx"]), int(seg["cy"])
    c = big[y0:y0 + 2 * half, x0:x0 + 2 * half]
    # area-preserving: keep the map's total (evidence) when shrinking
    out = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
    return out * (c.shape[0] * c.shape[1]) / float(size * size)


def _radial_profile(m, size=AVG_SIZE, nbins=46):
    """Evidence in each ring of r/R (rim at r/R = 1 sits at 1/1.2 of the half-width), as a
    SUM over the ring, so the bands add up to the total score (a per-pixel mean would
    exaggerate the small centre)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.hypot(xx - size / 2, yy - size / 2) / (size / 2 / 1.2)
    edges = np.linspace(0, 1.2 * np.sqrt(2), nbins + 1)
    idx = np.clip(np.digitize(rr, edges) - 1, 0, nbins - 1)
    s = np.bincount(idx.ravel(), m.ravel(), nbins)
    n = np.bincount(idx.ravel(), None, nbins)
    r = (edges[:-1] + edges[1:]) / 2
    keep = r <= R_MAX + 0.05
    return r[keep], s[keep]


def _pick(paths, max_n, seed=0):
    if max_n and len(paths) > max_n:
        rng = np.random.default_rng(seed)
        return [paths[i] for i in sorted(rng.choice(len(paths), max_n, replace=False))]
    return list(paths)


def explain_average(model, paths, target, versus=None, max_n=40, qc_mode="microscope",
                    downsample=None, log=print, progress=None, should_stop=lambda: False):
    """Average feature-evidence maps for `target` over the droplets in `paths` (e.g. every
    image of one whisky; a random max_n of them, default 40, ~1-3 s each). Droplets failing
    QC are skipped. Scalar / radial models only (use cnn_map_average for CNN models)."""
    if model.feature_set in ("cnn", "nuisance"):
        raise ValueError("explain_average covers scalar / radial and cnn+scalar+radial; "
                         "use cnn_map_average for a CNN-only model")
    ds_ = downsample or model.info.get("downsample", DOWNSAMPLE)
    use = _pick(list(paths), max_n)
    heats, imgs, gm, gsum, probs, hits, skipped = [], [], {}, {}, [], [], []
    parts = []                                         # (features part, CNN part) per droplet
    for i, p in enumerate(use, 1):
        if should_stop():
            raise InterruptedError("stopped")
        try:
            d = analyse(p, downsample=ds_)
            ok, why = (False, d.qc_reason) if d.seg is None else _qc(
                d.seg, float(np.isfinite(d.polar[..., 0]).mean()), qc_mode)
            if not ok:
                skipped.append((d.name, why))
                continue
            e = explain(model, d, target=target, versus=versus, smooth=0)
        except Exception as ex:  # noqa: BLE001
            skipped.append((os.path.basename(p), str(ex)))
            continue
        heats.append(_crop_map(e.heat, d.seg))
        parts.append((e.total - e.cnn_total, e.cnn_total))
        img = view_normalised(d) if model.normalised else view_raw(d)
        imgs.append(crop(img, d.seg, AVG_SIZE).astype(np.float32))
        for g, m in e.group_maps.items():
            gm[g] = gm.get(g, 0) + _crop_map(m, d.seg)
        for g, c in e.groups:
            gsum.setdefault(g, []).append(c)
        probs.append(dict(e.top).get(target, 0.0) if target else e.probability)
        hits.append(e.top[0][0] == e.target)
        if progress:
            progress(i, len(use))
    if not heats:
        raise ValueError("no usable droplet (all failed QC or had no droplet)")
    H = np.stack(heats)
    n = len(heats)
    r, prof = _radial_profile(H.mean(0))
    per = np.stack([_radial_profile(h)[1] for h in H])
    groups = sorted(((g, float(np.mean(v + [0] * (n - len(v)))), float(np.std(v + [0] * (n - len(v)))))
                     for g, v in gsum.items()), key=lambda t: -abs(t[1]))
    cv2 = _cv2()
    sig = AVG_SIZE / 150
    return AverageExplanation(
        target or "", versus or "average whisky", "features", n, skipped,
        np.clip(np.mean(imgs, 0), 0, 255).astype(np.uint8),
        cv2.GaussianBlur(H.mean(0), (0, 0), sig), H.std(0), r, prof, per.std(0), groups,
        {g: cv2.GaussianBlur(m / n, (0, 0), sig) for g, m in gm.items()},
        float(np.mean(probs)), float(np.mean(hits)),
        *(float(v) for pair in zip(np.mean(parts, 0), np.std(parts, 0)) for v in pair))


def cnn_map_average(model, paths, target, versus=None, max_n=40, downsample=None,
                    log=print, progress=None, should_stop=lambda: False):
    """Average CNN class-activation map (8 × 8 cells, crop frame) for `target` over
    the droplets in `paths`. CNN models only. Exact per droplet (see cnn_map)."""
    cv2 = _cv2()
    ds_ = downsample or model.info.get("downsample", DOWNSAMPLE)
    use = _pick(list(paths), max_n)
    cells, imgs, probs, hits, skipped = [], [], [], [], []
    for i, p in enumerate(use, 1):
        if should_stop():
            raise InterruptedError("stopped")
        try:
            cm = cnn_map(model, p, target=target, versus=versus, downsample=ds_)
        except Exception as ex:  # noqa: BLE001
            skipped.append((os.path.basename(p), str(ex)))
            continue
        cells.append(cm.cells)
        imgs.append(cv2.resize(cm.crop, (AVG_SIZE, AVG_SIZE)).astype(np.float32))
        probs.append(dict(cm.top).get(target, cm.probability) if target else cm.probability)
        hits.append(cm.top[0][0] == cm.target)
        if progress:
            progress(i, len(use))
    if not cells:
        raise ValueError("no usable droplet")
    C = np.stack(cells)
    # upsample, then rescale so each map still sums to its droplet's CNN score
    up = np.stack([cv2.resize(c, (AVG_SIZE, AVG_SIZE), interpolation=cv2.INTER_CUBIC) for c in C])
    up *= C[0].size / float(AVG_SIZE * AVG_SIZE)
    r, prof = _radial_profile(up.mean(0))
    per = np.stack([_radial_profile(u)[1] for u in up])
    return AverageExplanation(
        target or "", versus or "average whisky", "cnn", len(cells), skipped,
        np.clip(np.mean(imgs, 0), 0, 255).astype(np.uint8), up.mean(0), up.std(0), r, prof, per.std(0),
        [], {}, float(np.mean(probs)), float(np.mean(hits)))


def view_average_heat(a, alpha=0.7):
    """Mean evidence over the average droplet: red = for a.target, blue = against."""
    base = a.mean_image.astype(np.float32).mean(2, keepdims=True).repeat(3, 2) * 0.6
    h = a.heat / (np.percentile(np.abs(a.heat), 99.5) + 1e-12)
    h = (np.sign(h) * np.sqrt(np.clip(np.abs(h), 0, 1)))[..., None]
    col = np.where(h > 0, np.array([255, 40, 30], np.float32), np.array([40, 120, 255], np.float32))
    al = alpha * np.abs(h)
    return np.clip(base * (1 - al) + col * al, 0, 255).astype(np.uint8)


def view_average_groups(a, k=6, sign=+1, alpha=0.75):
    """Top-k feature groups for (sign=+1) / against (-1), coloured, over the average droplet.
    Returns (image, [(group, mean contribution, colour)])."""
    base = a.mean_image.astype(np.float32) * 0.45
    picked = [(g, c) for g, c, _ in a.groups if np.sign(c) == sign][:k]
    out, legend = base.copy(), []
    for i, (g, c) in enumerate(picked):
        m = np.abs(a.group_maps.get(g, np.zeros(base.shape[:2])))
        if (m > 0).any():
            m = np.clip(m / (np.percentile(m[m > 0], 99) + 1e-12), 0, 1)
        col = np.array(EXPLAIN_COLOURS[i % len(EXPLAIN_COLOURS)], np.float32)
        al = (alpha * m)[..., None]
        out = out * (1 - al) + col * al
        legend.append((g, c, EXPLAIN_COLOURS[i % len(EXPLAIN_COLOURS)]))
    return np.clip(out, 0, 255).astype(np.uint8), legend


def figure_explain_average(avgs, names=None, k=6, dpi=150):
    """One row per AverageExplanation (feature or CNN): average droplet | mean evidence
    map | (features: top groups FOR / AGAINST; CNN: consistency map) | radial evidence
    profile ± SD | ranked groups (features) or summary (CNN)."""
    plt = _plt()
    n = len(avgs)
    fig, ax = plt.subplots(n, 5, figsize=(25, 4.7 * n), squeeze=False,
                           gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15, 1.4]))
    for i, a in enumerate(avgs):
        ax[i, 0].imshow(a.mean_image)
        ax[i, 0].set_ylabel(_title(names, a.target, f"\n{a.kind}, n = {a.n}"), fontsize=10)
        ax[i, 0].set_xlabel(f"mean p({a.target}) {a.probability:.2f} · predicted as {a.target}: {a.accuracy:.0%}",
                            fontsize=8)
        ax[i, 1].imshow(view_average_heat(a))
        if a.kind == "features":
            pos, lp = view_average_groups(a, k, +1)
            ax[i, 2].imshow(pos)
            ax[i, 2].set_title(f"top {k} groups FOR (mean)", fontsize=9)
        else:
            sd = a.heat_sd / (np.abs(a.heat).max() + 1e-12)
            ax[i, 2].imshow(np.clip(sd, 0, 2), cmap="magma")
            ax[i, 2].set_title("spread over droplets (SD / max |mean|)", fontsize=9)
        ax[i, 3].plot(a.radial_r, a.radial, "k-", lw=1.4)
        ax[i, 3].fill_between(a.radial_r, a.radial - a.radial_sd, a.radial + a.radial_sd, color="grey", alpha=0.3, lw=0)
        ax[i, 3].fill_between(a.radial_r, 0, a.radial, where=a.radial > 0, color="tab:red", alpha=0.35, lw=0)
        ax[i, 3].fill_between(a.radial_r, 0, a.radial, where=a.radial < 0, color="tab:blue", alpha=0.35, lw=0)
        ax[i, 3].axhline(0, color="k", lw=0.5); ax[i, 3].axvline(1, color="grey", lw=0.6, ls=":")
        ax[i, 3].set_xlabel("r / R (1 = rim)"); ax[i, 3].set_title("evidence in each r/R band (sums to the score; mean ± SD)", fontsize=9)
        if a.kind == "features":
            g = a.groups[:12][::-1]
            ys = np.arange(len(g))
            ax[i, 4].barh(ys, [c for _, c, _ in g], xerr=[s for _, _, s in g],
                          color=["tab:red" if c > 0 else "tab:blue" for _, c, _ in g], ecolor="grey", capsize=2)
            ax[i, 4].set_yticks(ys, [x for x, _, _ in g], fontsize=8)
            ax[i, 4].axvline(0, color="k", lw=0.6)
            ax[i, 4].set_xlabel(f"mean contribution to {a.target} vs {a.versus} (logit) ± SD", fontsize=8)
        else:
            ax[i, 4].axis("off")
            ax[i, 4].text(0, 0.5, f"CNN evidence for {a.target} vs {a.versus}\naveraged over {a.n} droplets\n"
                          f"(8 × 8 cells per droplet, exact).\n\nSkipped: {len(a.skipped)}", fontsize=10, va="center")
        for j in range(3):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[0, 0].set_title("average droplet", fontsize=10)
    ax[0, 1].set_title("mean evidence (red = for, blue = against)", fontsize=10)
    fig.tight_layout()
    return _to_pil(fig, dpi)
