"""Standard deviation spectral feature extractor for hyperspectral images."""

import numpy as np

from hyppo.core import HSI
from .base import Extractor


class StdExtractor(Extractor):
    """Extractor that computes the standard deviation across spectral bands for each pixel."""

    def __init__(self) -> None:
        """Initialize standard deviation extractor."""
        super().__init__()

    def _extract(self, data: HSI, **inputs):

        reflectance = data.reflectance
        # Reshape to (n_pixels, n_bands) for easier spectral processing
        n_pixels = data.height * data.width
        pixels = reflectance.reshape(n_pixels, data.n_bands)
        mask = data.mask.reshape(n_pixels)

        # Compute standard deviation along spectral dimension
        result = np.zeros(pixels.shape[0], dtype=np.float32)

        # Only compute for valid pixels
        valid_pixels = pixels[mask]
        if valid_pixels.size > 0:
            result[mask] = np.std(valid_pixels, axis=1)

        # Set invalid pixels to NaN
        result[~mask] = np.nan

        return {
            "features": result,
        }
