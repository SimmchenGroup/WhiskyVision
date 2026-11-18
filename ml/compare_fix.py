"""Before/after comparison for the 14 Nov 2026 ml_core fixes.

  before: research-run features (identical to ml_core v1: clipped normalised image,
          in_mean_* kept, CNN normalised crop = clipped ×2.5)
  after:  ml_core v2 (unclipped normalised features, in_mean_* dropped, CNN
          normalised crop tone-mapped), recomputed from the same 1/4-size cache.
Each is scored with PCA keeping 99% and 95% of variance, same folds, same classifier.

    python compare_fix.py <out dir>
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_core as mc  # noqa: E402

OUT = sys.argv[1]
import ml_paths  # noqa: E402  (set WHISKY_ML_WORK, see ml_paths.py)
CACHE = ml_paths.need(ml_paths.IMAGE_CACHE, "cache_images.py")
F_RAW = ml_paths.need(ml_paths.FEATURES, "extract_features.py + embed_cnn.py")
F_NORM = ml_paths.need(ml_paths.FEATURES_NORM, "extract_features.py --norm + embed_cnn.py")
mc.CACHE_DIR = os.path.join(OUT, "cache")          # keep ~/.whisky_stitcher_ml free of these paths
log = open(os.path.join(OUT, "log.txt"), "a", buffering=1)


def say(*a):
    print(*a, file=log, flush=True)


def before_records():
    raw = {(r["letter"], r["sample"]): r for r in csv.DictReader(open(os.path.join(F_RAW, "features.csv")))}
    nrm = {(r["letter"], r["sample"]): r for r in csv.DictReader(open(os.path.join(F_NORM, "features.csv")))}
    er, en = np.load(os.path.join(F_RAW, "cnn_embed.npz")), np.load(os.path.join(F_NORM, "cnn_embed.npz"))
    cr = dict(zip(er["keys"], er["X"])); cn = dict(zip(en["keys"], en["X"]))
    recs = []
    for key, r in raw.items():
        if r["seg_ok"] != "True" or key not in nrm:
            continue
        L, s = key
        ok = (float(r["ring_score"]) >= mc.QC_MIN_SCORE and float(r["outside_frac"]) <= mc.QC_MAX_OUTSIDE
              and float(r["polar_valid"]) >= mc.QC_MIN_VALID)
        pr = np.load(os.path.join(F_RAW, "per_image", f"{L}_{s}.npz"))
        pn = np.load(os.path.join(F_NORM, "per_image", f"{L}_{s}.npz"))
        rec = mc.Record(os.path.join(CACHE, L, s + ".png"), L, ok, "" if ok else "failed QC",
                        dict(ring_score=float(r["ring_score"]), coverage=float(r["coverage"]),
                             outside_frac=float(r["outside_frac"]), polar_valid=float(r["polar_valid"]),
                             h=float(r["h"]), w=float(r["w"])),
                        np.array([float(r[k]) for k in mc.SCALAR_NAMES], np.float32),
                        np.array([float(nrm[key][k]) for k in mc.SCALAR_NAMES], np.float32),
                        mc.radial_vector({k: pr[k] for k in mc.PROFILE_KEYS}),
                        mc.radial_vector({k: pn[k] for k in mc.PROFILE_KEYS}))
        rec.cnn, rec.cnn_norm = cr.get(f"{L}_{s}"), cn.get(f"{L}_{s}")
        recs.append(rec)
    return recs


def after_records():
    paths, labels = [], []
    for L in sorted(os.listdir(CACHE)):
        d = os.path.join(CACHE, L)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".png"):
                    paths.append(os.path.join(d, f)); labels.append(L)
    t = time.time()
    last = [0]

    def prog(done, total):
        if done - last[0] >= 200 or done == total:
            last[0] = done
            say(f"  features {done}/{total} ({time.time() - t:.0f}s)")
    recs, skipped = mc.features_many(paths, labels, downsample=1, log=say, progress=prog, workers=12)
    recs = [r for r in recs if r is not None]
    say(f"features done: {len(recs)} records, {len(skipped)} skipped")
    t = time.time()
    last[0] = 0

    def prog2(done, total):
        if done - last[0] >= 320 or done == total:
            last[0] = done
            say(f"  cnn {done}/{total} ({time.time() - t:.0f}s)")
    mc.add_embeddings(recs, downsample=1, log=say, progress=prog2, workers=8)
    return recs


def evaluate(tag, recs):
    ds = mc.Dataset(CACHE, recs)
    rows = []
    orig_fit = mc.fit
    for vk in (0.99, 0.95):
        mc.fit = lambda *a, _vk=vk, **k: orig_fit(*a, **{**k, "var_keep": _vk})
        for fs in ("scalar", "radial", "scalar+radial", "cnn", "cnn+scalar+radial"):
            for norm in (False, True):
                t = time.time()
                res = mc.cross_validate(ds, fs, norm, log=lambda *_: None)
                rows.append(dict(version=tag, var_keep=vk, features=fs, normalised=norm,
                                 balanced_acc=round(res.balanced_accuracy, 4), accuracy=round(res.accuracy, 4),
                                 n=len(res.y_true), classes=len(res.classes), seconds=round(time.time() - t, 1)))
                say(json.dumps(rows[-1]))
    mc.fit = orig_fit
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    say("=== start", time.ctime())
    rows = evaluate("before", before_records())
    after = after_records()
    rows += evaluate("after", after)
    with open(os.path.join(OUT, "comparison.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    # inside one training run, after the fix (normalised cnn+scalar+radial, PCA 95%)
    ds = mc.Dataset(CACHE, after)
    rep = mc.training_report(ds, "cnn+scalar+radial", True, log=say)
    o = np.argsort(-rep["feature_weight"])[:20]
    say("top weights after fix: " + ", ".join(f"{rep['feature_names'][i]}={rep['feature_weight'][i]:.3f}" for i in o))
    say(f"kept {rep['kept']} of {rep['n_features']} components; steps {rep['history'][-1]['step']}; "
        f"final held-out {rep['history'][-1]['val_acc']:.3f}")
    if mc.figures_available():
        names = {}
        try:
            import identify_core as idc
            names = idc.load_key(idc.BUILTIN_KEY)
        except Exception:  # noqa: BLE001
            pass
        mc.figure_training(rep, names).save(os.path.join(OUT, "training_after_fix.png"))
    say("=== ALL DONE", time.ctime())
