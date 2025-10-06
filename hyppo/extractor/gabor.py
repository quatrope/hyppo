import numpy as np
from scipy import ndimage
from typing import Any, Dict, Optional, List, Tuple

from hyppo.core import HSI
from .base import Extractor


class GaborExtractor(Extractor):
    """Extractor that applies Gabor filters to extract texture features.

    Gabor filters are used to extract texture information at different
    frequencies and orientations.
    """

    def __init__(
        self,
        frequencies: Optional[List[float]] = None,
        thetas: Optional[List[float]] = None,
        sigma: float = 3.0,
        band_indices: Optional[List[int]] = None,
        aggregate_bands: bool = True,
    ) -> None:
        super().__init__()

        # Default frequencies and orientations if not provided
        if frequencies is None:
            frequencies = [0.05, 0.1, 0.2]  # Different spatial frequencies
        if thetas is None:
            thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135°

        self.frequencies = frequencies
        self.thetas = thetas
        self.sigma = sigma
        self.aggregate_bands = aggregate_bands

    def _create_gabor_kernel(
        self, frequency: float, theta: float, sigma: float
    ) -> np.ndarray:
        """Create a 2D Gabor kernel.

        Args:
            frequency: Spatial frequency of the sinusoidal component
            theta: Orientation in radians
            sigma: Standard deviation of the Gaussian envelope

        Returns:
            2D Gabor kernel
        """
        # Determine kernel size based on sigma
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center

                # Rotate coordinates
                x_theta = x * np.cos(theta) + y * np.sin(theta)
                y_theta = -x * np.sin(theta) + y * np.cos(theta)

                # Gaussian envelope
                gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))

                # Sinusoidal component
                sinusoid = np.cos(2 * np.pi * frequency * x_theta)

                kernel[i, j] = gaussian * sinusoid

        # Normalize kernel
        kernel = kernel - kernel.mean()
        kernel = kernel / np.abs(kernel).sum()

        return kernel

    def _apply_gabor_filter(
        self, image: np.ndarray, frequency: float, theta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a Gabor filter to an image.

        Args:
            image: 2D image array
            frequency: Spatial frequency
            theta: Orientation in radians

        Returns:
            Tuple of (magnitude response, phase response)
        """
        # Create Gabor kernel
        kernel = self._create_gabor_kernel(frequency, theta, self.sigma)

        # Apply convolution
        filtered_real = ndimage.convolve(image, kernel.real, mode="reflect")
        filtered_imag = ndimage.convolve(image, kernel.imag, mode="reflect")

        # Compute magnitude and phase
        magnitude = np.sqrt(filtered_real**2 + filtered_imag**2)
        phase = np.arctan2(filtered_imag, filtered_real)

        return magnitude, phase

    def _extract(self, data: HSI, **inputs) -> Dict[str, Any]:
        band_indices = data.get_band_indices()
        n_frequencies = len(self.frequencies)
        n_orientations = len(self.thetas)
        n_bands = len(band_indices)

        if self.aggregate_bands:
            # Output shape: (height, width, n_frequencies * n_orientations * 2)
            # Factor of 2 for magnitude and energy features
            n_features = n_frequencies * n_orientations * 2
            features = np.zeros((data.height, data.width, n_features), dtype=np.float32)

            # Process each band and aggregate
            band_features = []
            for band_idx in band_indices:
                band = data.get_band(band_idx)
                band_feat = self._extract_gabor_features_single_band(band)
                band_features.append(band_feat)

            # Average across bands
            features = np.mean(band_features, axis=0)
        else:
            # Output shape: (height, width, n_bands * n_frequencies * n_orientations * 2)
            n_features = n_bands * n_frequencies * n_orientations * 2
            features = np.zeros((data.height, data.width, n_features), dtype=np.float32)

            # Process each band separately
            feat_idx = 0
            for band_idx in band_indices:
                band = data.get_band(band_idx)
                band_features = self._extract_gabor_features_single_band(band)
                n_band_features = band_features.shape[2]
                features[:, :, feat_idx : feat_idx + n_band_features] = band_features
                feat_idx += n_band_features

        # Apply mask
        features[~data.mask] = np.nan

        return {
            "features": features,
        }

    def _extract_gabor_features_single_band(self, band: np.ndarray) -> np.ndarray:
        """Extract Gabor features from a single band.

        Args:
            band: 2D array representing a single spectral band

        Returns:
            Array of shape (height, width, n_features)
        """
        n_frequencies = len(self.frequencies)
        n_orientations = len(self.thetas)
        height, width = band.shape

        # 2 features per filter: magnitude mean and energy (magnitude squared mean)
        n_features = n_frequencies * n_orientations * 2
        features = np.zeros((height, width, n_features), dtype=np.float32)

        feat_idx = 0
        for freq in self.frequencies:
            for theta in self.thetas:
                # Apply Gabor filter
                magnitude, _ = self._apply_gabor_filter(band, freq, theta)

                # Store magnitude (texture intensity)
                features[:, :, feat_idx] = magnitude
                feat_idx += 1

                # Store energy (magnitude squared, texture strength)
                features[:, :, feat_idx] = magnitude**2
                feat_idx += 1

        return features
