"""entroscope: the definitive entropy toolkit for time series data."""

from . import shannon, permutation, spectral, sample, approximate, differential, multiscale
from .utils import plot

__version__ = "0.1.1"
__all__ = [
    "shannon",
    "permutation",
    "spectral",
    "sample",
    "approximate",
    "differential",
    "multiscale",
    "plot",
]
