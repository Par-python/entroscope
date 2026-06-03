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


_LN2 = np.log(2.0)


def _kl_bits(pp, qq):
    """KL(pp || qq) in bits, with epsilon smoothing so the result stays finite."""
    pp = pp + _EPS
    qq = qq + _EPS
    pp = pp / pp.sum()
    qq = qq / qq.sum()
    return float(np.sum(rel_entr(pp, qq)) / _LN2)


def kl(p, q, bins=10):
    """Kullback-Leibler divergence KL(p || q) in bits (directional).

    p, q are raw samples; both are binned over a shared range. Epsilon smoothing
    keeps the result finite even when q has an empty bin where p has mass.
    """
    pa, _ = _core.as_array(p)
    qa, _ = _core.as_array(q)
    pp, qq = _binned_probs(pa, qa, bins)
    return _kl_bits(pp, qq)
