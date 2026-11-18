"""Illustrative figures of the processing pipeline for the lab report.

    python report_figures.py "<cache dir>" "<features dir>" "<out dir>"

fig_pipeline.png  - raw -> ring map + rim fit -> normalised -> polar unwrap -> radial profile
fig_qc.png        - accepted vs rejected droplets (and why)
fig_gallery.png   - one typical droplet per whisky
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wwml

cache, feat, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
key = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "Whisky_Key.csv")).set_index(
    "Letter Designation")
df = pd.read_csv(os.path.join(feat, "features.csv"), dtype={"sample": str})
ok = df.seg_ok.astype(bool) & (df.ring_score >= 0.12) & (df.outside_frac <= 0.45) & (df.polar_valid >= 0.6)


def load(L, s):
    return cv2.cvtColor(cv2.imread(os.path.join(cache, L, s + ".png")), cv2.COLOR_BGR2RGB)


def show(rgb, gain=2.5):
    return np.clip(rgb.astype(np.float32) * gain / 255, 0, 1)


def rim_xy(seg):
    th = np.linspace(0, 2 * np.pi, len(seg["rim"]) + 1)
    rim = np.r_[seg["rim"], seg["rim"][0]]
    return seg["cx"] + rim * np.cos(th), seg["cy"] + rim * np.sin(th)


def typical(L):
    d = df[ok & (df.letter == L)]
    return d.iloc[(d.ring_score - d.ring_score.median()).abs().argsort().iloc[0]]["sample"]


# ------------------------------------------------------------------ pipeline
ex = [("H", typical("H")), ("Q", typical("Q")), ("K", typical("K"))]
fig, ax = plt.subplots(len(ex), 5, figsize=(19, 3.7 * len(ex)),
                       gridspec_kw=dict(width_ratios=[1, 1, 1, 1.6, 1.2]))
for i, (L, s) in enumerate(ex):
    rgb = load(L, s)
    seg = wwml.segment(rgb)
    ring, _ = wwml._ring_map(rgb)
    nrm = wwml.normalise(rgb, seg)
    pol = wwml.unwrap(nrm, seg)
    prof = wwml.radial_profiles(wwml.unwrap(rgb, seg))
    ax[i, 0].imshow(show(rgb))
    ax[i, 0].set_ylabel(f"{L}-{s}\n{key.loc[L, 'Name']}", fontsize=10)
    ax[i, 1].imshow(ring, cmap="magma")
    x, y = rim_xy(seg)
    ax[i, 1].plot(x, y, "c-", lw=1.2)
    ax[i, 1].plot(seg["cx"], seg["cy"], "c+", ms=10)
    ax[i, 2].imshow(show(nrm, 1.6))
    ax[i, 3].imshow(np.clip(np.nan_to_num(pol) * 1.6, 0, 1), aspect="auto", origin="lower",
                    extent=[0, 360, 0, wwml.R_MAX])
    ax[i, 3].axhline(1, color="w", lw=0.6, ls="--")
    ax[i, 3].set_xlabel("angle (°)")
    ax[i, 3].set_ylabel("r / R")
    r = np.linspace(0, wwml.R_MAX, wwml.N_R)
    for k, c in zip(("mean_R", "mean_G", "mean_B"), ("tab:red", "tab:green", "tab:blue")):
        ax[i, 4].plot(r, prof[k], color=c, lw=1.3, label=k[-1])
    ax[i, 4].plot(r, prof["std_L"], "k--", lw=1, label="azim. std")
    ax[i, 4].axvline(1, color="grey", lw=0.6, ls=":")
    ax[i, 4].set_xlabel("r / R")
    for j in range(3):
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    if i == 0:
        ax[i, 4].legend(fontsize=8)
for j, t in enumerate(["1. Stitched image (4× down, brightened)", "2. Ring map + fitted rim",
                       "3. Per-image normalised", "4. Radial unwrap (normalised)",
                       "5. Radial feature axis (raw)"]):
    ax[0, j].set_title(t, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(out, "fig_pipeline.png"), dpi=150)
plt.close()

# ------------------------------------------------------------------ QC examples
bad = df[df.seg_ok.astype(bool)]
picks = [("accepted", df[ok].sort_values("ring_score").iloc[[len(df[ok]) // 2, len(df[ok]) // 5]]),
         ("rejected: several droplets / ghosting (outside_frac > 0.45)",
          bad[(bad.outside_frac > 0.6) & (bad.ring_score > 0.2)].head(2)),
         ("rejected: no clear rim (ring_score < 0.12)", bad[bad.ring_score < 0.1].head(2))]
fig, ax = plt.subplots(1, 6, figsize=(20, 3.8))
k = 0
for lab, rows in picks:
    for _, r in rows.iterrows():
        rgb = load(r.letter, r["sample"])
        seg = wwml.segment(rgb)
        ax[k].imshow(show(rgb))
        if seg:
            x, y = rim_xy(seg)
            ax[k].plot(x, y, "lime" if lab == "accepted" else "r", lw=1.2)
        ax[k].set_title(f"{lab}\n{r.letter}-{r['sample']}  score {r.ring_score:.2f}, outside {r.outside_frac:.2f}",
                        fontsize=8)
        ax[k].axis("off")
        k += 1
plt.tight_layout()
plt.savefig(os.path.join(out, "fig_qc.png"), dpi=150)
plt.close()

# ------------------------------------------------------------------ gallery
letters = sorted(df.letter.unique())
fig, ax = plt.subplots(3, 6, figsize=(18, 9.6))
for a, L in zip(ax.ravel(), letters):
    s = typical(L)
    rgb = load(L, s)
    seg = wwml.segment(rgb)
    a.imshow(show(wwml.crop(rgb, seg, 384)))
    a.set_title(f"{L} · {key.loc[L, 'Name']}\n({L}-{s})", fontsize=9)
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(out, "fig_gallery.png"), dpi=150)
plt.close()
print("ok")
