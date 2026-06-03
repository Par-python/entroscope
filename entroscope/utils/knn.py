"""k-nearest-neighbor helpers for KSG-style estimators (Chebyshev / max norm)."""

import numpy as np
from scipy.spatial import cKDTree


def kth_neighbor_distance(points, k):
    """Chebyshev distance to the k-th nearest neighbor of each point.

    `points` is (n, d). Returns length-n array. Self (distance 0) is excluded by
    querying k+1 neighbors and dropping the first.
    """
    points = np.asarray(points, dtype=float)
    tree = cKDTree(points)
    # query returns the point itself first (distance 0); take the (k+1)-th column.
    dists, _ = tree.query(points, k=k + 1, p=np.inf)
    return np.asarray(dists)[:, k]


def count_within_radius(points, radii):
    """For each point, count OTHER points with Chebyshev distance strictly < radius.

    `points` is (n, d); `radii` is length n. Excludes the point itself.
    """
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    tree = cKDTree(points)
    counts = np.empty(len(points), dtype=int)
    for i, (p, r) in enumerate(zip(points, radii)):
        # ball_point with p=inf, count neighbors strictly inside r, minus self.
        idx = tree.query_ball_point(p, r=r, p=np.inf)
        # exclude self and any point exactly at radius r (strict inequality).
        c = 0
        for j in idx:
            if j == i:
                continue
            if np.max(np.abs(points[j] - p)) < r:
                c += 1
        counts[i] = c
    return counts
