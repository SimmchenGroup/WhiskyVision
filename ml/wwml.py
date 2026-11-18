"""Research-script view of the droplet analysis. The code lives in ../ml_core.py
(shared with the app's ML tab, numpy + OpenCV only); this keeps the names and
return types the scripts in ml/ were written against. Results are identical to the
original scipy version (checked 13 Nov 2026: features agree to < 1e-6).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_core as _mc  # noqa: E402

N_R, N_T, R_MAX, WORK = _mc.N_R, _mc.N_T, _mc.R_MAX, _mc.WORK
_ring_map = _mc.ring_map
segment = _mc.segment
crop = _mc.crop
unwrap = _mc.unwrap
radial_profiles = _mc.radial_profiles
scalar_features = _mc.scalar_features


def normalise(rgb, seg, target=0.15):
    """Normalised image only (ml_core.normalise also returns the step-by-step numbers)."""
    return _mc.normalise(rgb, seg, target)[0]
