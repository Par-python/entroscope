"""Normalization helpers: scale an entropy value into [0, 1]."""


def by_max(value, max_value):
    """Return value / max_value clipped to [0, 1]; 0.0 if max_value == 0."""
    if max_value == 0:
        return 0.0
    ratio = value / max_value
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return float(ratio)
