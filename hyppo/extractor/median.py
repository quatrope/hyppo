import numpy as np
from .spectral import SpectralExtractor


class MedianExtractor(SpectralExtractor):
    """Extractor that computes the median value across spectral bands for each pixel."""

    def __init__(self) -> None:
        super().__init__()

    def extract_spectral(
        self, pixels: np.ndarray, mask: np.ndarray, wavelengths: np.ndarray
    ) -> np.ndarray:
        # Compute median along spectral dimension
        result = np.zeros(pixels.shape[0], dtype=np.float32)

        # Only compute for valid pixels
        valid_pixels = pixels[mask]
        if valid_pixels.size > 0:
            result[mask] = np.median(valid_pixels, axis=1)

        # Set invalid pixels to NaN
        result[~mask] = np.nan

        return result
