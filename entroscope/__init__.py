"""entroscope: the definitive entropy toolkit for time series data."""

from . import (
    approximate,
    differential,
    divergence,
    multiscale,
    permutation,
    sample,
    shannon,
    spectral,
    transfer,
)
from .utils import plot

__version__ = "0.3.0"
__all__ = [
    "approximate",
    "differential",
    "divergence",
    "multiscale",
    "permutation",
    "plot",
    "sample",
    "shannon",
    "spectral",
    "transfer",
]
