"""Step-by-step figure of the per-image normalisation (wwml.normalise).

    python fig_normalisation.py "<cache dir>" "<features dir>" out.png

Columns: raw | background region | background histograms | after background
subtraction | droplet mean per channel (before / after) | normalised.
Columns 1 and 4 share ONE display gain across all rows, so brightness differences
between droplets are shown honestly; column 6 uses the normalised display gain.
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
df = pd.read_csv(os.path.join(feat, "features.csv"), dtype={"sample": str})
key = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "Whisky_Key.csv")).set_index(
    "Letter Designation")
EX = [("H", "121"), ("Q", "048"), ("K", "107")]
GAIN = 2.5            # shared display gain for raw and background-subtracted
COLS = ("tab:red", "tab:green", "tab:blue")

fig, ax = plt.subplots(len(EX), 6, figsize=(24, 4.2 * len(EX)),
                       gridspec_kw=dict(width_ratios=[1, 1, 1.1, 1, 0.9, 1]))
for i, (L, s) in enumerate(EX):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(cache, L, s + ".png")), cv2.COLOR_BGR2RGB)
    seg = wwml.segment(rgb)
    x = rgb.astype(np.float32) / 255
    canvas = rgb.max(2) > 0
    H, W = canvas.shape
    th = np.linspace(0, 2 * np.pi, len(seg["rim"]), endpoint=False)
    big = np.zeros((H, W), np.uint8)
    cv2.fillPoly(big, [np.stack([seg["cx"] + 1.25 * seg["rim"] * np.cos(th),
                                 seg["cy"] + 1.25 * seg["rim"] * np.sin(th)], 1).astype(np.int32)], 1)
    bgmask = canvas & ~big.astype(bool)
    m = seg["mask"].astype(bool)
    bg = np.array([np.median(x[..., c][bgmask]) for c in range(3)])
    sub = np.clip(x - bg, 0, None)
    sub[~canvas] = 0
    mean_before = np.array([x[..., c][m].mean() for c in range(3)])
    mean_sub = np.array([sub[..., c][m].mean() for c in range(3)])
    nrm = wwml.normalise(rgb, seg).astype(np.float32) / 255
    mean_after = np.array([nrm[..., c][m].mean() for c in range(3)])

    # 1 raw
    ax[i, 0].imshow(np.clip(x * GAIN, 0, 1))
    ax[i, 0].set_ylabel(f"{L}-{s}\n{key.loc[L, 'Name']}", fontsize=11)
    # 2 background region
    vis = np.clip(x * GAIN, 0, 1) * 0.55
    vis[bgmask] = vis[bgmask] * 0.4 + np.array([0.1, 0.5, 1.0]) * 0.6
    vis[m] = np.clip(x[m] * GAIN, 0, 1)
    ax[i, 1].imshow(vis)
    for k, ls in ((1.0, "-"), (1.25, "--")):
        ax[i, 1].plot(np.r_[seg["cx"] + k * seg["rim"] * np.cos(th), seg["cx"] + k * seg["rim"][0]],
                      np.r_[seg["cy"] + k * seg["rim"] * np.sin(th), seg["cy"]], color="w", lw=1, ls=ls)
    ax[i, 1].text(4, H - 8, f"background: {100 * bgmask.sum() / canvas.sum():.0f}% of imaged area",
                  color="w", fontsize=8)
    # 3 background histograms with medians
    for c in range(3):
        v = x[..., c][bgmask] * 255
        ax[i, 2].hist(v, bins=np.arange(0, 80, 1), histtype="step", color=COLS[c], lw=1.3)
        ax[i, 2].axvline(bg[c] * 255, color=COLS[c], ls="--", lw=1)
    ax[i, 2].set_xlabel("pixel value (0–255)")
    ax[i, 2].set_ylabel("pixels")
    ax[i, 2].text(0.97, 0.95, "median R/G/B\n" + " / ".join(f"{b * 255:.0f}" for b in bg),
                  transform=ax[i, 2].transAxes, ha="right", va="top", fontsize=9)
    # 4 after subtraction
    ax[i, 3].imshow(np.clip(sub * GAIN, 0, 1))
    # 5 droplet means before/after
    xs = np.arange(3)
    ax[i, 4].bar(xs - 0.27, mean_before * 255, 0.27, color=COLS, alpha=0.35, label="raw")
    ax[i, 4].bar(xs, mean_sub * 255, 0.27, color=COLS, alpha=0.65, label="− background")
    ax[i, 4].bar(xs + 0.27, mean_after * 255, 0.27, color=COLS, edgecolor="k", label="÷ droplet mean")
    ax[i, 4].set_xticks(xs, ["R", "G", "B"])
    ax[i, 4].set_ylabel("mean inside droplet (0–255)")
    ax[i, 4].text(0.02, 0.97, "gain ×" + " / ".join(f"{0.15 / ms:.1f}" for ms in mean_sub),
                  transform=ax[i, 4].transAxes, va="top", fontsize=9)
    if i == 0:
        ax[i, 4].legend(fontsize=7, loc="upper right")
    ax[i, 4].set_ylim(0, max(60, mean_before.max() * 255 * 1.25))
    # 6 normalised
    ax[i, 5].imshow(np.clip(nrm * 1.6, 0, 1))
    for j in (0, 1, 3, 5):
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])

titles = ["1. Raw (display gain ×2.5, same for all rows)",
          "2. Background region (blue):\nimaged pixels beyond 1.25 R (dashed)",
          "3. Background histogram per channel\n(dashed = median = value subtracted)",
          "4. After background subtraction\n(same display gain as 1)",
          "5. Droplet mean per channel:\nraw → −background → ÷ own mean",
          "6. Normalised: every droplet's mean\n= 0.15 in R, G and B (display ×1.6)"]
for j, t in enumerate(titles):
    ax[0, j].set_title(t, fontsize=10)
plt.tight_layout()
plt.savefig(out, dpi=150)
print("ok")
