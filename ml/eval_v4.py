"""17 Nov 2026: evaluation of ml_core v4 (body-silhouette rim + asymmetry/texture features).
Ablation on the SAME v4 records: scalar+radial with and without the 13 new features,
so the segmentation change and the new features are judged separately (v3 run = before)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_core as mc
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
mc.CACHE_DIR = os.path.join(OUT, "cache")
log = open(os.path.join(OUT, "log.txt"), "a", buffering=1)
say = lambda *a: print(*a, file=log, flush=True)
C = r"C:\Users\joshk\jarvis\jarvis\home\outputs\whisky\ml_cache"
paths, labels = [], []
for L in sorted(os.listdir(C)):
    d = os.path.join(C, L)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".png"): paths.append(os.path.join(d, f)); labels.append(L)
say("=== start", time.ctime(), mc.VERSION)
t = time.time(); last = [0]
def prog(n, tot, what="features"):
    if n - last[0] >= 400 or n == tot:
        last[0] = n; say(f"  {what} {n}/{tot} ({time.time() - t:.0f}s)")
recs, _ = mc.features_many(paths, labels, downsample=1, log=say, progress=prog, workers=12)
recs = [r for r in recs if r is not None]
t = time.time(); last[0] = 0
mc.add_embeddings(recs, downsample=1, log=say, progress=lambda n, tot: prog(n, tot, "cnn"), workers=8)
ds = mc.Dataset(C, recs)
say("QC: " + json.dumps(ds.qc_table()))
rows = []
NEW = set(mc.SCALAR_NAMES[26:])
orig_fm = mc.feature_matrix
def fm_without_new(records, feature_set="scalar+radial", normalised=False):
    X, names = orig_fm(records, feature_set, normalised)
    keep = [i for i, n in enumerate(names) if n not in NEW]
    return X[:, keep], [names[i] for i in keep]
for tag, fm in (("v4 without new features", fm_without_new), ("v4", orig_fm)):
    mc.feature_matrix = fm
    for fs in ("scalar", "scalar+radial", "cnn+scalar+radial"):
        for norm in (False, True):
            res = mc.cross_validate(ds, fs, norm, log=lambda *_: None)
            rows.append(dict(version=tag, features=fs, normalised=norm, balanced_acc=round(res.balanced_accuracy, 4),
                             n=len(res.y_true), recall=res.recall,
                             wrong=[[p, a, b] for p, a, b in zip(res.paths, res.y_true.tolist(), res.y_pred.tolist()) if a != b]))
            say(json.dumps({k: v for k, v in rows[-1].items() if k not in ("recall", "wrong")}))
mc.feature_matrix = orig_fm
res = mc.cross_validate(ds, "cnn", False, log=lambda *_: None)
rows.append(dict(version="v4", features="cnn", normalised=False, balanced_acc=round(res.balanced_accuracy, 4),
                 n=len(res.y_true), recall=res.recall,
                 wrong=[[p, a, b] for p, a, b in zip(res.paths, res.y_true.tolist(), res.y_pred.tolist()) if a != b]))
say(json.dumps({k: v for k, v in rows[-1].items() if k not in ("recall", "wrong")}))
json.dump(rows, open(os.path.join(OUT, "eval_v4.json"), "w"), indent=1)
say("=== ALL DONE", time.ctime())
