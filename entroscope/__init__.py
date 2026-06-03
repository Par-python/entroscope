"""entroscope: the definitive entropy toolkit for time series data."""

from . import (
    shannon,
    permutation,
    spectral,
    sample,
    approximate,
    differential,
    multiscale,
    transfer,
    divergence,
)
from .utils import plot

__version__ = "0.2.0"
__all__ = [
    "shannon",
    "permutation",
    "spectral",
    "sample",
    "approximate",
    "differential",
    "multiscale",
    "transfer",
    "divergence",
    "plot",
]
