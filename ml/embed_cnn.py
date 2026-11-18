"""Frozen EfficientNet-B0 (ImageNet) embeddings of droplet-centred crops.

    python embed_cnn.py "<cache dir>" "<features dir>"

Crop = square of side 2.4 x mean rim radius centred on the droplet (zero-padded),
resized to 256 px. The embedding is averaged over 4 rotations x 2 flips so it
is orientation-invariant. Writes <features>/cnn_embed.npz (keys, X).
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

cache, feat = sys.argv[1], sys.argv[2]
df = pd.read_csv(os.path.join(feat, "features.csv"), dtype={"sample": str})
df = df[df.seg_ok.astype(bool)]
torch.set_num_threads(20)
m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
m.classifier = torch.nn.Identity()
m.eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
S = 256


def crop(rgb, cx, cy, r):
    half = int(1.2 * r)
    big = cv2.copyMakeBorder(rgb, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    x0, y0 = int(cx), int(cy)
    c = big[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA)


keys, X = [], []
batch, bkeys = [], []


def flush():
    if not batch:
        return
    x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255
    # brighten: dark-field webs are very dim relative to ImageNet photos
    x = torch.clamp(x * 2.5, 0, 1)
    views = []
    for k in range(4):
        v = torch.rot90(x, k, (2, 3))
        views += [v, torch.flip(v, (3,))]
    with torch.no_grad():
        e = torch.stack([m((v - mean) / std) for v in views]).mean(0)
    X.append(e.numpy())
    keys.extend(bkeys)
    batch.clear()
    bkeys.clear()


for n, (_, r) in enumerate(df.iterrows(), 1):
    z = np.load(os.path.join(feat, "per_image", f"{r.letter}_{r['sample']}.npz"))
    if "crop" in z:                       # saved by extract_features (raw or normalised)
        batch.append(z["crop"])
    else:
        rgb = cv2.cvtColor(cv2.imread(os.path.join(cache, r.letter, r["sample"] + ".png")),
                           cv2.COLOR_BGR2RGB)
        batch.append(crop(rgb, r.cx, r.cy, r.radius_px))
    bkeys.append(f"{r.letter}_{r['sample']}")
    if len(batch) == 32:
        flush()
        print(n, flush=True)
flush()
np.savez_compressed(os.path.join(feat, "cnn_embed.npz"), keys=np.array(keys), X=np.concatenate(X))
print("done", len(keys))
