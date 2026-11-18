"""16 Nov 2026: full evaluation after the rim re-trace (ml_core v3). Compares with v2
(ml_fix/comparison.json, 'after', var_keep 0.99) and gives per-whisky recall."""
import json, os, sys, time
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
for fs in ("scalar", "radial", "scalar+radial", "cnn", "cnn+scalar+radial", "nuisance"):
    for norm in ((False, True) if fs != "nuisance" else (False,)):
        res = mc.cross_validate(ds, fs, norm, log=lambda *_: None)
        rows.append(dict(features=fs, normalised=norm, balanced_acc=round(res.balanced_accuracy, 4),
                         n=len(res.y_true), recall=res.recall))
        say(json.dumps({k: v for k, v in rows[-1].items() if k != "recall"}))
json.dump(rows, open(os.path.join(OUT, "eval_v3.json"), "w"), indent=1)
say("=== ALL DONE", time.ctime())
