"""Transfer-entropy estimators (history length 1) and shared delay-embedding.

Private module: all the math lives here so the public ``transfer`` module stays
thin. Both estimators consume the SAME embedded vectors from ``embed`` — the
embedding is verified in isolation (see tests) so a bug here cannot silently fool
both estimators.

Correctness is established by three independent test authorities (see
tests/test_transfer.py): the bivariate-Gaussian closed form (magnitude), the
Kraskov-2004 analytic mutual information (KSG core), and a hand-checked isolated
embedding test.
"""

import numpy as np
from scipy.special import digamma
from .utils import knn


def embed(x, y, lag=1):
    """Build aligned (y_future, y_past, x_past) sample columns, history length 1.

    For each valid time t (from ``lag`` to n-1): y_future=y[t], y_past=y[t-lag],
    x_past=x[t-lag]. Returns three 1-D arrays of length ``n - lag``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if lag < 1:
        raise ValueError("lag must be >= 1")
    n = len(y)
    if n - lag < 1:
        raise ValueError(f"series too short ({n}) for lag {lag}")
    y_future = y[lag:]
    y_past = y[: n - lag]
    x_past = x[: n - lag]
    return y_future, y_past, x_past


def _hist_prob(*cols, bins):
    """Joint probability table over the given equal-width-binned columns."""
    sample = np.column_stack(cols)
    counts, _ = np.histogramdd(sample, bins=bins)
    total = counts.sum()
    return counts / total if total > 0 else counts


def te_binned(y_future, y_past, x_past, bins=6):
    """Transfer entropy X->Y via equal-width histogram probabilities (base 2)."""
    p_fpx = _hist_prob(y_future, y_past, x_past, bins=bins)   # p(yf, yp, xp)
    p_fp = p_fpx.sum(axis=2)                                  # p(yf, yp)
    p_px = p_fpx.sum(axis=0)                                  # p(yp, xp)
    p_p = p_fpx.sum(axis=(0, 2))                              # p(yp)

    te = 0.0
    nf, npq, nx = p_fpx.shape
    for i in range(nf):
        for j in range(npq):
            for k in range(nx):
                pijk = p_fpx[i, j, k]
                if pijk <= 0:
                    continue
                denom = p_fp[i, j] * p_px[j, k]
                numer = pijk * p_p[j]
                if denom <= 0 or numer <= 0:
                    continue
                te += pijk * np.log2(numer / denom)
    return float(te)


_LN2 = np.log(2.0)


def te_ksg(y_future, y_past, x_past, k=4):
    """Transfer entropy X->Y via the KSG (Kraskov) k-NN estimator, in bits.

    Joint space Z = (y_future, y_past, x_past). eps = Chebyshev distance to the
    k-th neighbor in Z; marginal neighbor counts within eps give the digamma
    terms. Returned in bits (nats / ln 2) to match the library's base-2 output.
    """
    yf = np.asarray(y_future, dtype=float).reshape(-1, 1)
    yp = np.asarray(y_past, dtype=float).reshape(-1, 1)
    xp = np.asarray(x_past, dtype=float).reshape(-1, 1)

    z = np.column_stack([yf, yp, xp])
    eps = knn.kth_neighbor_distance(z, k=k)

    sp_yp = yp
    sp_fp = np.column_stack([yf, yp])
    sp_px = np.column_stack([yp, xp])

    n_yp = knn.count_within_radius(sp_yp, eps)
    n_fp = knn.count_within_radius(sp_fp, eps)
    n_px = knn.count_within_radius(sp_px, eps)

    terms = digamma(n_yp + 1) - digamma(n_fp + 1) - digamma(n_px + 1)
    te_nats = digamma(k) + np.mean(terms)
    return float(te_nats / _LN2)


def ksg_mutual_information(a, b, k=4):
    """KSG (Kraskov algorithm 1) mutual information between a and b, in NATS.

    a, b are (n, da) and (n, db). Returns I(a; b) in nats so it can be checked
    against the analytic -0.5 ln(1 - r^2) for correlated Gaussians.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    joint = np.column_stack([a, b])
    eps = knn.kth_neighbor_distance(joint, k=k)
    n_a = knn.count_within_radius(a, eps)
    n_b = knn.count_within_radius(b, eps)
    mi = digamma(k) + digamma(n) - np.mean(digamma(n_a + 1) + digamma(n_b + 1))
    return float(mi)
