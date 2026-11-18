"""Visual check of segmentation + unwrap on a few cached images."""
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wwml

paths = sys.argv[2:]
out = sys.argv[1]
fig, ax = plt.subplots(2, len(paths), figsize=(3.2 * len(paths), 6))
for i, p in enumerate(paths):
    rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    s = wwml.segment(rgb)
    vis = np.clip(rgb.astype(np.float32) * 2.5, 0, 255).astype(np.uint8)
    if s:
        th = np.linspace(0, 2 * np.pi, len(s["rim"]), endpoint=False)
        pts = np.stack([s["cx"] + s["rim"] * np.cos(th), s["cy"] + s["rim"] * np.sin(th)], 1)
        cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 255, 0), max(1, rgb.shape[1] // 300))
        pol = wwml.unwrap(rgb, s)
        ax[1, i].imshow(np.clip(np.nan_to_num(pol) * 2.5, 0, 1), aspect="auto", origin="lower",
                        extent=[0, 360, 0, wwml.R_MAX])
        ttl = f"score {s['score']:.3f} cov {s['coverage']:.2f}\noval {s['ovality']:.3f} in {s['inside_canvas']:.2f}"
    else:
        ttl = "FAILED"
    ax[0, i].imshow(vis)
    ax[0, i].set_title(p.replace("\\", "/").split("/")[-2] + "/" + p.split("/")[-1] + "\n" + ttl, fontsize=7)
    ax[0, i].axis("off")
    ax[1, i].set_xlabel("angle (deg)", fontsize=7)
    ax[1, i].set_ylabel("r / R", fontsize=7)
plt.tight_layout()
plt.savefig(out, dpi=110)
