"""Where the research scripts keep their working data (not in the repo).

Set the environment variable WHISKY_ML_WORK to a folder of your choice, e.g.
    Windows (once):  setx WHISKY_ML_WORK "D:\\whisky_ml"
    bash:            export WHISKY_ML_WORK=/d/whisky_ml
Without it, ~/whisky_ml is used. Inside it:
    ml_cache/<letter>/<sample>.png   1/4-size image copies (made by cache_images.py)
    ml_features/                     features (extract_features.py, embed_cnn.py)
    ml_features_norm/                the same with --norm
"""
import os
import sys

WORK = os.path.abspath(os.path.expanduser(os.environ.get("WHISKY_ML_WORK") or "~/whisky_ml"))
IMAGE_CACHE = os.path.join(WORK, "ml_cache")
FEATURES = os.path.join(WORK, "ml_features")
FEATURES_NORM = os.path.join(WORK, "ml_features_norm")


def need(path, made_by):
    """Exit with a clear message if a working folder doesn't exist yet."""
    if not os.path.isdir(path):
        sys.exit(f"{path} not found.\nMake it with {made_by}, or set WHISKY_ML_WORK to the folder "
                 f"that holds ml_cache / ml_features (currently {WORK}).")
    return path


def image_cache_paths():
    """(paths, labels) of every cached 1/4-size image, letter = folder name."""
    c = need(IMAGE_CACHE, 'cache_images.py "<Khalifa Samples>" "%s" --letters ...' % IMAGE_CACHE)
    paths, labels = [], []
    for L in sorted(os.listdir(c)):
        d = os.path.join(c, L)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".png"):
                    paths.append(os.path.join(d, f))
                    labels.append(L)
    return paths, labels
