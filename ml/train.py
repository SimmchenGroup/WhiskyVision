"""Whisky-web classification experiments (classic ML on feature sets).

    python train.py "<features dir>" "<out dir>" [--min-score 0.12 --max-outside 0.45]

Feature sets compared (5-fold stratified CV, same folds for all):
  nuisance  - radius, ring score, coverage, canvas size  (control: imaging artefacts)
  scalar    - interpretable droplet features (ring width/contrast, colour, particles...)
  radial    - the radial 'feature axis' from the polar unwrap (profiles vs r/R)
  cnn       - frozen EfficientNet-B0 embedding of the droplet crop
  scalar+radial, cnn+radial, cnn+scalar+radial
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROFILE_KEYS = ["mean_R", "mean_G", "mean_B", "std_L", "p95_L", "hue_ratio"]
# imaging session = date of the raw tiles (file mtime on the J: drive, checked 2026-11-13)
SESSION = {"A": "08-25", "V": "09-10", "Y": "09-10",
           **{l: "10-06" for l in "QRSTUW"}, **{l: "10-07" for l in "JKLMNOP"},
           "H": "10-13", "I": "10-13"}
RAW_BRIGHTNESS = ["ring_score", "ring_strength"]      # computed on the raw image in every mode
NUISANCE = ["radius_px", "ring_score", "coverage", "outside_frac", "h", "w"]
NON_FEATURES = {"letter", "sample", "seg_ok", "cx", "cy", "h", "w", "coverage",
                "inside_canvas", "outside_frac", "outside_ring", "polar_valid"}


def load(feat, min_score, max_outside, min_valid):
    df = pd.read_csv(os.path.join(feat, "features.csv"), dtype={"sample": str})
    n0 = df.groupby("letter").size()
    q = df.seg_ok.astype(bool) & (df.ring_score >= min_score) & (df.outside_frac <= max_outside) \
        & (df.polar_valid >= min_valid)
    df = df[q].reset_index(drop=True)
    qc = pd.DataFrame({"images": n0, "kept": df.groupby("letter").size()}).fillna(0).astype(int)
    prof = []
    for _, r in df.iterrows():
        z = np.load(os.path.join(feat, "per_image", f"{r.letter}_{r['sample']}.npz"))
        # 128 radial bins -> 32 (mean over blocks of 4) per profile
        prof.append(np.concatenate([z[k].reshape(32, 4).mean(1) for k in PROFILE_KEYS]))
    return df, np.array(prof), qc


def models():
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), PCA(0.99, svd_solver="full"),
                                        LogisticRegression(C=0.5, max_iter=4000)),
        "extratrees": lambda: ExtraTreesClassifier(600, max_features="sqrt", n_jobs=-1,
                                                   class_weight="balanced", random_state=0),
    }


def cv_eval(X, y, make, folds):
    pred = np.empty_like(y)
    proba = np.zeros((len(y), len(np.unique(y))))
    for tr, te in folds:
        m = make()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        proba[te] = m.predict_proba(X[te])
    return pred, proba


def main():
    p = argparse.ArgumentParser()
    p.add_argument("feat")
    p.add_argument("out")
    p.add_argument("--min-score", type=float, default=0.12)
    p.add_argument("--max-outside", type=float, default=0.45)
    p.add_argument("--min-valid", type=float, default=0.6)
    p.add_argument("--normalised", action="store_true",
                   help="features came from normalised images: drop raw-brightness features")
    p.add_argument("--session-z", action="store_true",
                   help="z-score every feature within its imaging session")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df, P, qc = load(a.feat, a.min_score, a.max_outside, a.min_valid)
    qc.to_csv(os.path.join(a.out, "qc_counts.csv"))
    print(qc.T.to_string())
    y = df.letter.values
    sess = df.letter.map(SESSION).values
    drop = NON_FEATURES | (set(RAW_BRIGHTNESS) if a.normalised else set())
    scal_cols = [c for c in df.columns if c not in drop]
    S = df[scal_cols].fillna(0).values
    N = df[NUISANCE].fillna(0).values
    sets = {"nuisance": N, "scalar": S, "radial": P, "scalar+radial": np.hstack([S, P])}
    ep = os.path.join(a.feat, "cnn_embed.npz")
    if os.path.exists(ep):
        z = np.load(ep)
        idx = {k: i for i, k in enumerate(z["keys"])}
        keys = (df.letter + "_" + df["sample"]).values
        if all(k in idx for k in keys):
            E = z["X"][[idx[k] for k in keys]]
            sets["cnn"] = E
            sets["cnn+radial"] = np.hstack([E, P])
            sets["cnn+scalar+radial"] = np.hstack([E, S, P])

    if a.session_z:
        def sz(X):
            X = X.astype(np.float64).copy()
            for s in np.unique(sess):
                m = sess == s
                X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)
            return X
        sets = {k: (v if k == "nuisance" else sz(v)) for k, v in sets.items()}

    # within-session test: whiskies imaged on the same day share lamp/slide/exposure,
    # so separating them can't rely on session artefacts
    ws_rows = []
    for s in sorted(np.unique(sess)):
        m = sess == s
        ys = y[m]
        if len(np.unique(ys)) < 2 or np.bincount(np.unique(ys, return_inverse=True)[1]).min() < 5:
            continue
        f_s = list(StratifiedKFold(5, shuffle=True, random_state=0).split(ys, ys))
        for sname in ("scalar+radial", "radial", "cnn"):
            if sname not in sets:
                continue
            for mname, make in models().items():
                pr, _ = cv_eval(sets[sname][m], ys, make, f_s)
                ws_rows.append(dict(session=s, whiskies="".join(sorted(np.unique(ys))),
                                    n=int(m.sum()), chance=1 / len(np.unique(ys)),
                                    features=sname, model=mname,
                                    balanced_acc=balanced_accuracy_score(ys, pr)))
                print(ws_rows[-1], flush=True)
    pd.DataFrame(ws_rows).to_csv(os.path.join(a.out, "within_session.csv"), index=False)

    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(S, y))
    rows, preds = [], {}
    for sname, X in sets.items():
        for mname, make in models().items():
            pr, proba = cv_eval(X, y, make, folds)
            rows.append(dict(features=sname, model=mname, n_features=X.shape[1],
                             accuracy=accuracy_score(y, pr),
                             balanced_acc=balanced_accuracy_score(y, pr),
                             macro_f1=f1_score(y, pr, average="macro")))
            preds[(sname, mname)] = (pr, proba)
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows).sort_values("balanced_acc", ascending=False)
    res.to_csv(os.path.join(a.out, "cv_results.csv"), index=False)

    labels = sorted(np.unique(y))
    best = res.iloc[0]
    pr = preds[(best.features, best.model)][0]
    cm = confusion_matrix(y, pr, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(a.out, "confusion_best.csv"))
    pd.DataFrame({"letter": df.letter, "sample": df["sample"], "pred": pr}).to_csv(
        os.path.join(a.out, "predictions_best.csv"), index=False)

    # interpretability: which features matter (ExtraTrees on scalar+radial, all data)
    et = ExtraTreesClassifier(600, max_features="sqrt", n_jobs=-1, class_weight="balanced",
                              random_state=0).fit(np.hstack([S, P]), y)
    names = scal_cols + [f"{k}@r{(i + 0.5) / 32 * 1.15:.2f}" for k in PROFILE_KEYS for i in range(32)]
    imp = pd.Series(et.feature_importances_, names).sort_values(ascending=False)
    imp.to_csv(os.path.join(a.out, "feature_importance.csv"))
    np.savez(os.path.join(a.out, "for_plots.npz"), P=P, y=y, cm=cm, labels=labels)
    json.dump(dict(best=best.to_dict(), qc=vars(a), n=int(len(y))),
              open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
