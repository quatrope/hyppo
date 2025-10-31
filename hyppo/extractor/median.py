"""Median spectral value feature extractor for hyperspectral images."""

import numpy as np

from hyppo.core import HSI
from .base import Extractor


class MedianExtractor(Extractor):
    """Extractor that computes the median value across spectral bands for each pixel."""

    def __init__(self) -> None:
        """Initialize median extractor."""
        super().__init__()

    def _extract(self, data: HSI, **inputs):
        reflectance = data.reflectance
        mask = data.mask.reshape(data.height, data.width)

        # Reshape to (n_pixels, n_bands) for easier spectral processing
        n_pixels = data.height * data.width
        pixels = reflectance.reshape(n_pixels, data.n_bands)
        mask = mask.reshape(n_pixels)

        # Compute median along spectral dimension
        result = np.zeros(pixels.shape[0], dtype=np.float32)

        # Only compute for valid pixels
        valid_pixels = pixels[mask]
        if valid_pixels.size > 0:
            result[mask] = np.median(valid_pixels, axis=1)

        # Set invalid pixels to NaN
        result[~mask] = np.nan

        return {
            "features": result,
        }
