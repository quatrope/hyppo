import numpy as np
from hyppo.hsi import HSI
from .base import Extractor


class MeanExtractor(Extractor):
    """Extractor that computes the mean value across spectral bands for each pixel."""

    def __init__(self) -> None:
        super().__init__()

    def extract(self, data: HSI):
        reflectance = data.reflectance
        mask = data.mask.reshape(data.height, data.width)

        # Reshape to (n_pixels, n_bands) for easier spectral processing
        n_pixels = data.height * data.width
        pixels = reflectance.reshape(n_pixels, data.n_bands)
        mask = mask.reshape(n_pixels)

        # Compute mean along spectral dimension
        result = np.zeros(pixels.shape[0], dtype=np.float32)

        # Only compute for valid pixels
        valid_pixels = pixels[mask]
        if valid_pixels.size > 0:
            result[mask] = np.mean(valid_pixels, axis=1)

        # Set invalid pixels to NaN
        result[~mask] = np.nan

        return {
            "mean": result,
        }
