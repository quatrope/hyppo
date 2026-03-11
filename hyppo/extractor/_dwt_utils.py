"""Shared utilities for DWT-based extractors."""

import numpy as np


def calculate_swt_padding(shape, levels):
    """Calculate padding needed for Stationary Wavelet Transform.

    Returns (padding_per_axis, needs_padding).
    padding_per_axis is a tuple of ints, one per dimension in shape.
    """
    divisor = 2 ** levels
    padding = tuple((divisor - s % divisor) % divisor for s in shape)
    needs_padding = any(p > 0 for p in padding)
    return padding, needs_padding


def apply_swt_padding(data, padding, needs_padding):
    """Apply reflect padding to data array if needed."""
    if not needs_padding:
        return data
    pad_width = tuple((0, p) for p in padding)
    return np.pad(data, pad_width, mode="reflect")
