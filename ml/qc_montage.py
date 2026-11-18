"""Montage of QC overlays sorted by ring_score, to choose a QC threshold by eye.

    python qc_montage.py "<features dir>" out.png [--n 40] [--lo 0 --hi 1]
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("feat")
p.add_argument("out")
p.add_argument("--n", type=int, default=40)
p.add_argument("--lo", type=float, default=0)
p.add_argument("--hi", type=float, default=9)
p.add_argument("--letters", default="")
p.add_argument("--sort", default="ring_score")
a = p.parse_args()
df = pd.read_csv(os.path.join(a.feat, "features.csv"), dtype={"sample": str})
df = df[df.seg_ok & (df.ring_score >= a.lo) & (df.ring_score < a.hi)]
if a.letters:
    df = df[df.letter.isin(a.letters.split(","))]
p2 = a.sort if hasattr(a, "sort") else "ring_score"
df = df.sort_values(p2)
if len(df) > a.n:
    df = df.iloc[np.linspace(0, len(df) - 1, a.n).astype(int)]
tiles = []
for _, r in df.iterrows():
    im = cv2.imread(os.path.join(a.feat, "qc", f"{r.letter}_{r['sample']}.jpg"))
    c = np.zeros((300, 300, 3), np.uint8)
    c[:im.shape[0], :im.shape[1]] = im
    cv2.putText(c, f"{r.letter}{r['sample']} s{r.ring_score:.2f} o{r.outside_frac:.2f} g{r.outside_ring:.2f}",
                (4, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tiles.append(c)
cols = 8
while len(tiles) % cols:
    tiles.append(np.zeros((300, 300, 3), np.uint8))
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(a.out, np.vstack(rows))
