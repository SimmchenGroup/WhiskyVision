"""Small CNN trained directly on the radial unwrap (r/R x angle).

    python polar_cnn.py "<features dir>" "<out dir>" [--epochs 30] [--folds 5]

Design: in the polar image a rotation of the droplet is a circular shift along
the angle axis, so convolutions use circular padding in angle, training uses
random circular shifts/flips, and the head averages over angle but KEEPS the
radial position -> rotation-invariant, but knows *where* along r/R a pattern
sits (centre deposit vs coffee ring vs halo). Optional --with-profile also
feeds the 1-D radial profile vector into the head (fusion).
Same QC filter and same CV folds as train.py.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from train import load


class CircConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, 3, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):                       # x: B,C,R,T
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, 1, 1))
        return F.relu(self.bn(self.c(x)))


class PolarNet(nn.Module):
    def __init__(self, n_cls, n_r=64, n_prof=0):
        super().__init__()
        ch = [3, 32, 64, 96, 128]
        pools = [(1, 2), (2, 2), (2, 2), (2, 2)]
        self.blocks = nn.ModuleList()
        for i in range(4):
            self.blocks.append(nn.Sequential(CircConv(ch[i], ch[i + 1]), CircConv(ch[i + 1], ch[i + 1]),
                                             nn.MaxPool2d(pools[i])))
        r_out = n_r // 8
        self.prof = nn.Sequential(nn.Linear(n_prof, 128), nn.ReLU(), nn.Dropout(0.3)) if n_prof else None
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(128 * r_out + (128 if n_prof else 0), n_cls))

    def forward(self, x, prof=None):
        for b in self.blocks:
            x = b(x)
        x = x.mean(3).flatten(1)                 # average over angle, keep radius
        if self.prof is not None:
            x = torch.cat([x, self.prof(prof)], 1)
        return self.head(x)


def load_polar(feat, df):
    X = []
    for _, r in df.iterrows():
        z = np.load(os.path.join(feat, "per_image", f"{r.letter}_{r['sample']}.npz"))
        p = z["polar"].astype(np.float32) / 255           # 128 x 360 x 3
        p = p.reshape(64, 2, 180, 2, 3).mean((1, 3))      # -> 64 x 180
        X.append(np.sqrt(p))                              # lift the dim interior
    X = np.stack(X).transpose(0, 3, 1, 2)                 # N,3,64,180
    return X.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feat")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--with-profile", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.12)
    ap.add_argument("--max-outside", type=float, default=0.45)
    ap.add_argument("--min-valid", type=float, default=0.6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(20)
    torch.manual_seed(0)
    np.random.seed(0)

    df, P, _ = load(a.feat, a.min_score, a.max_outside, a.min_valid)
    X = load_polar(a.feat, df)
    labels = sorted(df.letter.unique())
    y = np.array([labels.index(l) for l in df.letter])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, df.letter.values))[:a.folds]

    pred = np.full(len(y), -1)
    tag = "polar_cnn+radial" if a.with_profile else "polar_cnn"
    for k, (tr, te) in enumerate(folds):
        mu = X[tr].mean((0, 2, 3), keepdims=True)
        sd = X[tr].std((0, 2, 3), keepdims=True) + 1e-6
        Xn = torch.from_numpy((X - mu) / sd)
        pm, ps = P[tr].mean(0), P[tr].std(0) + 1e-6
        Pn = torch.from_numpy(((P - pm) / ps).astype(np.float32))
        yt = torch.from_numpy(y)
        net = PolarNet(len(labels), n_prof=P.shape[1] if a.with_profile else 0)
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
        steps = a.epochs * int(np.ceil(len(tr) / 32))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
        w = torch.tensor(len(tr) / (len(labels) * np.bincount(y[tr], minlength=len(labels)) + 1e-6),
                         dtype=torch.float32)
        t0 = time.time()
        for ep in range(a.epochs):
            net.train()
            perm = np.random.permutation(tr)
            tot = 0
            for i in range(0, len(perm), 32):
                b = perm[i:i + 32]
                xb = Xn[b]
                # rotation = circular shift in angle; mirror = flip in angle
                xb = torch.roll(xb, int(np.random.randint(180)), 3)
                if np.random.rand() < 0.5:
                    xb = torch.flip(xb, (3,))
                xb = xb * (1 + 0.05 * torch.randn(len(b), 1, 1, 1))
                loss = F.cross_entropy(net(xb, Pn[b]), yt[b], weight=w, label_smoothing=0.05)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
                tot += loss.item() * len(b)
            if ep % 5 == 4 or ep == a.epochs - 1:
                print(f"fold {k} ep {ep + 1} loss {tot / len(tr):.3f} ({time.time() - t0:.0f}s)", flush=True)
        net.eval()
        with torch.no_grad():
            xt = Xn[te]
            logit = sum(net(torch.roll(xt, s, 3), Pn[te]).softmax(1) for s in (0, 45, 90, 135))
        pred[te] = logit.argmax(1).numpy()
        print(f"fold {k} acc {np.mean(pred[te] == y[te]):.3f}", flush=True)

    m = pred >= 0
    yt, pt = np.array(labels)[y[m]], np.array(labels)[pred[m]]
    res = dict(features=tag, model="cnn", n_features=int(np.prod(X.shape[1:])),
               accuracy=accuracy_score(yt, pt), balanced_acc=balanced_accuracy_score(yt, pt),
               macro_f1=f1_score(yt, pt, average="macro"))
    print(res)
    pd.DataFrame([res]).to_csv(os.path.join(a.out, f"cv_{tag}.csv"), index=False)
    pd.DataFrame({"letter": df.letter[m], "sample": df["sample"][m], "pred": pt}).to_csv(
        os.path.join(a.out, f"predictions_{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
