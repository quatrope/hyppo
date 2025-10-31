"""Discrete Wavelet Transform 1D feature extractor."""

import numpy as np
import pywt

from hyppo.core import HSI
from .base import Extractor


class DWT1DExtractor(Extractor):
    """
    Discrete Wavelet Transform (1D) feature extractor for hyperspectral images.

    Applies a 1D DWT to each pixel's spectral signature to extract
    multiscale spectral features.

    Parameters
    ----------
    wavelet : str, optional
        Wavelet name to use (default: 'db4').
    mode : str, optional
        Signal extension mode (default: 'symmetric').
    levels : int, optional
        Number of decomposition levels (default: 3).

    References
    ----------
    Bruce, K., Koger, C., & Li, J. (2002). Dimensionality reduction of
    hyperspectral data using discrete wavelet transform feature extraction.
    *IEEE Transactions on Geoscience and Remote Sensing*, 40(10),
    2331–2338. https://doi.org/10.1109/TGRS.2002.804721

    Mallat, S. (1999). A Wavelet Tour of Signal Processing.
    Academic Press.
    """

    def __init__(self, wavelet="db4", mode="symmetric", levels=3):
        """Initialize DWT1D extractor with wavelet parameters."""
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.levels = levels

    @classmethod
    def feature_name(cls):
        """Return feature name for this extractor."""
        return "dwt1d"

    def _extract(self, data: HSI, **inputs):
        """
        Extract 1D DWT features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:
                - "features" : ndarray
                    DWT-transformed array with shape (H, W, n_features).
                - "wavelet" : str
                    Wavelet used.
                - "mode" : str
                    Signal extension mode.
                - "levels" : int
                    Number of decomposition levels.
                - "coeffs_lengths" : list of int
                    Length of coefficients at each decomposition level.
                - "n_features" : int
                    Total number of features per pixel.
                - "original_shape" : tuple
                    Shape of the original HSI (H, W, bands).
        """
        # Prepare data
        reflectance = data.reflectance
        height, width, bands = reflectance.shape
        reflectance_reshaped = reflectance.reshape(-1, bands)

        # Apply DWT to each pixel's spectral signature
        features_list = []

        for i in range(reflectance_reshaped.shape[0]):
            pixel_spectrum = reflectance_reshaped[i, :]

            # Apply 1D DWT to the spectral signature
            coefficients = pywt.wavedec(
                pixel_spectrum, self.wavelet, mode=self.mode, level=self.levels
            )

            # Concatenate all coefficients as features
            pixel_features = np.concatenate(coefficients)
            features_list.append(pixel_features)

        features_2d = np.array(features_list)
        features = features_2d.reshape(height, width, -1)

        # Get coefficient lengths from first pixel for reference
        sample_coefficients = pywt.wavedec(
            reflectance_reshaped[0, :],
            self.wavelet,
            mode=self.mode,
            level=self.levels,
        )
        coefficients_lengths = [len(c) for c in sample_coefficients]

        # TODO: Consider validating max_level using pywt.dwt_max_level()

        return {
            "features": features,
            "wavelet": self.wavelet,
            "mode": self.mode,
            "levels": self.levels,
            "coeffs_lengths": coefficients_lengths,
            "n_features": features.shape[1],
            "original_shape": (height, width, bands),
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{self.wavelet}' not available")

        if self.mode not in pywt.Modes.modes:
            raise ValueError(f"Mode '{self.mode}' not available")

        if not isinstance(self.levels, int) or self.levels <= 0:
            raise ValueError("levels must be a positive integer")


# Wavelet selection affects sensitivity and decomposition:
# - 'haar': detects abrupt changes and jumps in spectrum (ideal for edges).
# - 'db4': captures smooth variations between absorption bands.
# - 'sym5': good balance between smoothness and symmetry.
# - 'coif2': models smooth complex shapes (more computationally costly).
#
# Decomposition level determines signal analysis depth:
# Max level depends on signal length and wavelet filter.
# Can be obtained with pywt.dwt_max_level().
# For signal length 107 and 'db4' (filter length 8),
# max level is 3, which we use.
