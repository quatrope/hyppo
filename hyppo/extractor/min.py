import numpy as np

from hyppo.core import HSI
from .base import Extractor


class MinExtractor(Extractor):
    """Extractor that computes the minimum value across spectral bands for each pixel."""

    def __init__(self) -> None:
        super().__init__()

    def _extract(self, data: HSI, **inputs):

        reflectance = data.reflectance
        # Reshape to (n_pixels, n_bands) for easier spectral processing
        n_pixels = data.height * data.width
        pixels = reflectance.reshape(n_pixels, data.n_bands)
        mask = data.mask.reshape(n_pixels)

        # Compute minimum along spectral dimension
        result = np.zeros(pixels.shape[0], dtype=np.float32)

        # Only compute for valid pixels
        valid_pixels = pixels[mask]
        if valid_pixels.size > 0:
            result[mask] = np.min(valid_pixels, axis=1)

        # Set invalid pixels to NaN
        result[~mask] = np.nan

        return {
            "features": result,
        }
