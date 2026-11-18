"""Prepare phone-microscope droplet photos to look more like the dark-field microscope
images (17 Nov 2026).  Originals are never changed.

    python phone_prep.py "<in root>" "<out root>" [--sat 3] [--no-bg] [--preview out.png]

Per image: (1) find the droplet (ml_core.segment on a 1/4-size copy); (2) fit a smooth
background (quadratic surface, per channel) to the pixels well outside the droplet and
subtract it -> dark, neutral background, no vignetting or colour cast; (3) multiply the
HSV saturation by --sat. Written as PNG, same folder structure, full resolution.
"""
import argparse, os, sys, json
import numpy as np, cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_core as mc


def background(rgb_small, seg, full_shape):
    """Quadratic surface per channel fitted to pixels beyond 1.3 R, evaluated at full size."""
    h, w = rgb_small.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.ones((h, w), np.uint8)
    if seg is not None:
        cv2.fillPoly(out, [mc._rim_polygon(seg["cx"], seg["cy"], seg["rim"], 1.3)], 0)
    m = out.astype(bool)
    if m.sum() < 200:                                     # droplet fills the frame: use the dimmest 20%
        m = rgb_small.mean(2) <= np.percentile(rgb_small.mean(2), 20)
    quad = m.mean() > 0.35            # little background visible: a plane (a curved fit extrapolates badly)

    def basis(x, y):
        t = [np.ones_like(x), x, y] + ([x * x, y * y, x * y] if quad else [])
        return np.stack(t, -1)
    A = basis(xx[m] / w, yy[m] / h)
    H, W = full_shape
    YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
    B = basis(XX / W, YY / H)
    bg = np.empty((H, W, 3), np.float32)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, rgb_small[..., c][m].astype(np.float32), rcond=None)
        bg[..., c] = B @ coef
    return bg


def saturate(rgb, k):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * k, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def prepare(path, sat=3.0, remove_bg=True):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    info = {}
    if remove_bg:
        small = cv2.resize(rgb, (rgb.shape[1] // 4, rgb.shape[0] // 4), interpolation=cv2.INTER_AREA)
        seg = mc.segment(small)
        bg = background(small, seg, rgb.shape[:2])
        x = np.clip(rgb.astype(np.float32) - bg, 0, 255)
        # stretch so the droplet's bright end uses the range like the microscope (p99.5 -> ~200)
        p = np.percentile(x, 99.5) + 1e-3
        x = np.clip(x * min(200.0 / p, 4.0), 0, 255).astype(np.uint8)
        x[x.max(2) == 0] = 1                              # keep the image "imaged" (no zero pixels)
        info = dict(droplet_found=seg is not None, bg_mean=bg.reshape(-1, 3).mean(0).round(1).tolist(),
                    stretch=round(float(min(200.0 / p, 4.0)), 2))
        rgb = x
    return saturate(rgb, sat), info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--sat", type=float, default=3.0)
    ap.add_argument("--no-bg", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = []
    for L in sorted(os.listdir(a.inp)):
        d = os.path.join(a.inp, L)
        if os.path.isdir(d):
            os.makedirs(os.path.join(a.out, L), exist_ok=True)
            jobs += [(os.path.join(d, f), os.path.join(a.out, L, os.path.splitext(f)[0] + ".png"))
                     for f in sorted(os.listdir(d))
                     if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")) and not f.startswith(".")]
    def run(j):
        src, dst = j
        if os.path.exists(dst):
            return dst, "exists"
        try:
            img, info = prepare(src, a.sat, not a.no_bg)
        except Exception as e:  # noqa: BLE001  unreadable / corrupt file: log it, carry on
            return dst, f"SKIPPED: {e}"
        Image.fromarray(img).save(dst)
        return dst, info
    log = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (dst, info) in enumerate(ex.map(run, jobs), 1):
            log[dst] = info
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)
    json.dump(dict(settings=vars(a), images=log), open(os.path.join(a.out, "prep_log.json"), "w"), indent=1)
