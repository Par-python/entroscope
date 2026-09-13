import numpy as np
import pytest

from entroscope.utils.knn import count_within_radius, kth_neighbor_distance


def _brute_count(points, radii):
    """O(n^2) reference: other points at Chebyshev distance strictly < radius."""
    dist = np.max(np.abs(points[:, None, :] - points[None, :, :]), axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.sum(dist < radii[:, None], axis=1)


@pytest.mark.parametrize(
    "points",
    [
        np.random.RandomState(0).randn(400, 2),
        np.round(np.random.RandomState(1).randn(400, 3), 1),  # many exact ties
        np.random.RandomState(2).randint(0, 4, (300, 2)).astype(float),  # duplicates
    ],
    ids=["continuous", "rounded", "duplicates"],
)
def test_count_within_radius_matches_brute_force(points):
    rng = np.random.RandomState(3)
    for radii in (
        kth_neighbor_distance(points, 4),  # boundary points sit exactly at r
        np.zeros(len(points)),
        rng.rand(len(points)),
    ):
        np.testing.assert_array_equal(
            count_within_radius(points, radii), _brute_count(points, radii)
        )
