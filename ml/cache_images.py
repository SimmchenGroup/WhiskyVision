"""Cache downsampled copies of stitched whisky-web images locally.

Reads <root>/<letter>/Stitched/*_stitched.tif over the network once and writes
<cache>/<letter>/<sample>.png at a FIXED downsample factor (default 4), so that
physical scale (droplet size) is preserved across images.

    python cache_images.py "<Khalifa Samples>" "<cache dir>" --letters A,H,I --factor 4
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tifffile


def process(args):
    src, dst, factor = args
    if os.path.exists(dst):
        return None
    try:
        a = tifffile.imread(src)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        if a.shape[0] <= 4 and a.ndim == 3:
            a = np.moveaxis(a, 0, -1)
        a = a[..., :3]
        if a.dtype != np.uint8:
            a = (a.astype(np.float32) / max(a.max(), 1) * 255).astype(np.uint8)
        h, w = a.shape[:2]
        small = cv2.resize(a, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        return {"src": src, "dst": dst, "h": h, "w": w}
    except Exception as e:  # noqa: BLE001
        return {"src": src, "error": str(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("cache")
    p.add_argument("--letters", required=True)
    p.add_argument("--factor", type=int, default=4)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()

    jobs = []
    for L in a.letters.split(","):
        sdir = os.path.join(a.root, L, "Stitched")
        os.makedirs(os.path.join(a.cache, L), exist_ok=True)
        for f in sorted(os.listdir(sdir)):
            if f.lower().endswith((".tif", ".tiff")) and "_stitched" in f:
                jobs.append((os.path.join(sdir, f),
                             os.path.join(a.cache, L, f.split("_stitched")[0] + ".png"),
                             a.factor))
    print(f"{len(jobs)} images", flush=True)
    log = os.path.join(a.cache, "cache_log.jsonl")
    done = 0
    with ThreadPoolExecutor(a.workers) as ex, open(log, "a") as fh:
        for r in ex.map(process, jobs):
            done += 1
            if r:
                fh.write(json.dumps(r) + "\n")
            if done % 50 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
