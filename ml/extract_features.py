"""Segment every cached droplet, radially unwrap it and extract features.

    python extract_features.py "<cache dir>" "<out dir>" [--workers 16]

Writes (incrementally; re-running skips images already done):
  <out>/per_image/<L>_<id>.npz   polar unwrap (uint8, N_R x N_T x 3) + radial profiles
  <out>/features.csv             one row per image: QC + scalar features
  <out>/qc/<L>_<id>.jpg          small overlay of the detected rim
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

import wwml


def work(job):
    L, sid, path, out, norm = job
    npz = os.path.join(out, "per_image", f"{L}_{sid}.npz")
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    row = dict(letter=L, sample=sid, h=rgb.shape[0], w=rgb.shape[1])
    seg = wwml.segment(rgb)                  # always segment on the raw image
    if seg is None:
        row["seg_ok"] = False
        return row
    if norm:
        rgb = wwml.normalise(rgb, seg)
    pol = wwml.unwrap(rgb, seg)
    prof = wwml.radial_profiles(pol)
    row.update(seg_ok=True, cx=seg["cx"], cy=seg["cy"], ring_score=seg["score"],
               coverage=seg["coverage"], inside_canvas=seg["inside_canvas"],
               outside_frac=seg["outside_frac"], outside_ring=seg["outside_ring"],
               polar_valid=float(np.isfinite(pol[..., 0]).mean()))
    row.update(wwml.scalar_features(rgb, seg, pol, prof))
    np.savez_compressed(npz, polar=np.nan_to_num(pol * 255, nan=0).astype(np.uint8),
                        valid=np.isfinite(pol[..., 0]), crop=wwml.crop(rgb, seg), **prof)
    # QC overlay
    vis = np.clip(rgb.astype(np.float32) * 2.5, 0, 255).astype(np.uint8)
    th = np.linspace(0, 2 * np.pi, len(seg["rim"]), endpoint=False)
    pts = np.stack([seg["cx"] + seg["rim"] * np.cos(th), seg["cy"] + seg["rim"] * np.sin(th)], 1)
    cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 255, 0), max(1, rgb.shape[1] // 250))
    s = 300 / max(vis.shape[:2])
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out, "qc", f"{L}_{sid}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--norm", action="store_true",
                   help="per-image background/gain/white-balance normalisation before features")
    a = p.parse_args()
    for d in ("per_image", "qc"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    csv = os.path.join(a.out, "features.csv")
    old = pd.read_csv(csv, dtype={"sample": str}) if os.path.exists(csv) else pd.DataFrame()
    done = set(zip(old.get("letter", []), old.get("sample", [])))
    jobs = []
    for L in sorted(os.listdir(a.cache)):
        d = os.path.join(a.cache, L)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".png") and (L, f[:-4]) not in done:
                jobs.append((L, f[:-4], os.path.join(d, f), a.out, a.norm))
    print(f"{len(jobs)} new images", flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=2), 1):
            rows.append(r)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
                pd.concat([old, pd.DataFrame(rows)], ignore_index=True).sort_values(
                    ["letter", "sample"]).to_csv(csv, index=False)        # checkpoint
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df.sort_values(["letter", "sample"]).to_csv(csv, index=False)
    print("done", len(df), flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
