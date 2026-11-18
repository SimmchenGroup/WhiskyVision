"""15 Nov 2026: CNN raw input with soft roll-off (CNN_VERSION ...x2.5soft) vs the hard
clamp. Reuses the v2 feature cache from compare_fix.py; only CNN embeddings are new.
    python compare_cnn_softclip.py <old cache dir (ml_fix/cache)> <out dir>"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import ml_core as mc
OLD, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
mc.CACHE_DIR = OLD                         # feature records (ml-v2) already there; new CNN files are added
log = open(os.path.join(OUT, "log.txt"), "a", buffering=1)
say = lambda *a: print(*a, file=log, flush=True)
C = r"C:\Users\joshk\jarvis\jarvis\home\outputs\whisky\ml_cache"
paths, labels = [], []
for L in sorted(os.listdir(C)):
    d = os.path.join(C, L)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".png"):
                paths.append(os.path.join(d, f)); labels.append(L)
say("=== start", time.ctime(), mc.CNN_VERSION)
recs, _ = mc.features_many(paths, labels, downsample=1, log=say, workers=12)
recs = [r for r in recs if r is not None]
say(f"features: {len(recs)}")
NEW = mc.CNN_VERSION
mc.CNN_VERSION = "effb0-imagenet1k-v1|crop256|x2.5|rot4flip2|norm-tonemap"   # old: hard clamp (cached)
n_old = mc.add_embeddings(recs, downsample=1, log=say, workers=8)
say(f"old embeddings loaded (computed {n_old})")
ds0 = mc.Dataset(C, recs)
old_rows = []
for fs in ("cnn", "cnn+scalar+radial"):
    for norm in (False, True):
        res = mc.cross_validate(ds0, fs, norm, log=lambda *_: None)
        old_rows.append(dict(version="hard clamp", features=fs, normalised=norm,
                             balanced_acc=round(res.balanced_accuracy, 4), recall=res.recall))
        say(json.dumps({k: v for k, v in old_rows[-1].items() if k != "recall"}))
for r in recs:
    r.cnn = r.cnn_norm = None
mc.CNN_VERSION = NEW
t = time.time(); last = [0]
def prog(done, total):
    if done - last[0] >= 320 or done == total:
        last[0] = done; say(f"  cnn {done}/{total} ({time.time() - t:.0f}s)")
mc.add_embeddings(recs, downsample=1, log=say, progress=prog, workers=8)
ds = mc.Dataset(C, recs)
rows = []
for fs in ("cnn", "cnn+scalar+radial"):
    for norm in (False, True):
        res = mc.cross_validate(ds, fs, norm, log=lambda *_: None)
        rows.append(dict(version="soft roll-off", features=fs, normalised=norm, balanced_acc=round(res.balanced_accuracy, 4),
                         accuracy=round(res.accuracy, 4), recall=res.recall))
        say(json.dumps({k: v for k, v in rows[-1].items() if k != "recall"}))
json.dump(old_rows + rows, open(os.path.join(OUT, "cnn_softclip.json"), "w"), indent=1)
say("=== ALL DONE", time.ctime())
