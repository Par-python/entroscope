"""Divergence between two distributions — KL and Jensen-Shannon.

Distribution-vs-distribution measures: how far apart are two samples? Inputs are
two raw sample arrays/Series; both are histogrammed over a SHARED range so the
resulting probability vectors are aligned and comparable. Base-2 (bits) to match
the rest of the library.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import rel_entr

from . import _core

_EPS = 1e-12


def _binned_probs(p, q, bins):
    """Histogram p and q over SHARED edges; return two aligned probability vectors.

    Edges span the combined [min, max] of both samples so the bins line up. A
    degenerate (zero-width) combined range falls back to a single bin.
    """
    if bins <= 0:
        raise ValueError("bins must be a positive integer")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())
    if hi <= lo:  # all values identical across both samples
        return np.array([1.0]), np.array([1.0])
    edges = np.linspace(lo, hi, bins + 1)
    cp, _ = np.histogram(p, bins=edges)
    cq, _ = np.histogram(q, bins=edges)
    pp = cp / cp.sum()
    qq = cq / cq.sum()
    return pp, qq
