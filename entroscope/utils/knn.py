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
    # query_ball_point counts distance <= r; shrinking each radius to the next
    # float below it gives the strict inequality. Subtract 1 for the point itself.
    inner = np.nextafter(radii, 0)
    counts = np.asarray(
        tree.query_ball_point(points, r=inner, p=np.inf, return_length=True), dtype=int
    )
    # Nothing is strictly within a zero radius, but duplicates sit at distance 0.
    return np.where(radii > 0, counts - 1, 0)
