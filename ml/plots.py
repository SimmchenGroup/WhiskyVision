"""Figures for the whisky-web ML run.

    python plots.py "<results dir>" "<Whisky_Key.csv>"
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

out, keyf = sys.argv[1], sys.argv[2]
key = pd.read_csv(keyf).set_index("Letter Designation")
z = np.load(os.path.join(out, "for_plots.npz"), allow_pickle=True)
P, y, cm, labels = z["P"], z["y"], z["cm"], list(z["labels"])
name = {l: f"{l} {key.loc[l, 'Name']}" for l in labels}
r = (np.arange(32) + 0.5) / 32 * 1.15
cmap = plt.get_cmap("tab20")

# 1. radial feature axis per whisky
fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
L = P[:, 0:96].reshape(len(P), 3, 32).mean(1)            # mean of R,G,B profiles
for j, (blk, ttl) in enumerate([(L, "mean intensity"), (P[:, 96:128], "azimuthal std (heterogeneity)"),
                                (P[:, 160:192], "colour (R-B)/(R+B)")]):
    for i, l in enumerate(labels):
        m = y == l
        mu = blk[m].mean(0)
        ax[j].plot(r, mu, color=cmap(i % 20), lw=1.4, label=name[l])
    ax[j].axvline(1, color="k", lw=0.6, ls="--")
    ax[j].set_title(ttl)
    ax[j].set_xlabel("r / R (1 = rim)")
ax[0].set_ylabel("value (class mean)")
ax[2].legend(fontsize=7, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(out, "radial_feature_axis.png"), dpi=150)
plt.close()

# 2. confusion matrix (row-normalised) of the best model
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)), labels)
ax.set_yticks(range(len(labels)), [name[l] for l in labels], fontsize=8)
for i in range(len(labels)):
    for j in range(len(labels)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cmn[i, j] > 0.5 else "black")
ax.set_xlabel("predicted")
ax.set_ylabel("true")
plt.colorbar(im, fraction=0.04, label="fraction of true class")
res = pd.read_csv(os.path.join(out, "cv_results.csv")).iloc[0]
ax.set_title(f"Best: {res.features} / {res.model}, balanced acc {res.balanced_acc:.2f}")
plt.tight_layout()
plt.savefig(os.path.join(out, "confusion_best.png"), dpi=150)
plt.close()

# 3. results bar chart (all feature sets, best model per set)
res = pd.read_csv(os.path.join(out, "cv_results.csv"))
for f in os.listdir(out):
    if f.startswith("cv_polar"):
        res = pd.concat([res, pd.read_csv(os.path.join(out, f))])
best = res.sort_values("balanced_acc").groupby("features").tail(1).sort_values("balanced_acc")
fig, ax = plt.subplots(figsize=(8, 4.5))
cols = ["#bbbbbb" if f == "nuisance" else "#3b6ea8" for f in best.features]
ax.barh([f"{f} ({m})" for f, m in zip(best.features, best.model)], best.balanced_acc, color=cols)
ax.axvline(1 / len(labels), color="r", ls="--", lw=1, label=f"chance ({1 / len(labels):.3f})")
for i, v in enumerate(best.balanced_acc):
    ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlabel("balanced accuracy (5-fold CV)")
ax.set_xlim(0, 1)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(out, "results_bar.png"), dpi=150)
plt.close()
print("ok")
