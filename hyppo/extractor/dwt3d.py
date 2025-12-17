"""Discrete Wavelet Transform (3D) feature extractor for HSI."""

import numpy as np
import pywt

from hyppo.core import HSI
from .base import Extractor


class DWT3DExtractor(Extractor):
    """
    3D Discrete Wavelet Transform feature extractor.

    Applies 3D Stationary Wavelet Transform (SWT) to the hyperspectral cube
    (Spatial + Spectral dimensions) to extract joint spectral-spatial texture features.

    As described in Qian et al. (2013), 3D DWT decomposes the data into 8 subbands
    capturing correlations across spatial and spectral domains simultaneously.

    Parameters
    ----------
    wavelet : str
        Wavelet name to use (default: 'haar').
    levels : int
        Number of decomposition levels (default: 1).

    References
    ----------
    Qian, Y., Ye, M., & Zhou, J. (2013). Hyperspectral Image Classification
    Based on Structured Sparse Logistic Regression and Three-Dimensional
    Wavelet Texture Features. *IEEE Transactions on Geoscience and Remote Sensing*,
    51(4), 2276-2291.
    """

    def __init__(self, wavelet="haar", levels=1):
        """Initialize DWT3D extractor with wavelet parameters."""
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels

    @classmethod
    def feature_name(cls) -> str:
        """Return the feature name."""
        return "dwt3d"

    def _extract(self, data: HSI, **inputs):
        """
        Extract 3D DWT features from a hyperspectral image.

        Args
        ----
        data : HSI
            Hyperspectral image object

        Returns
        -------
        dict
            - "features": 3D SWT coefficient maps stacked (H, W, n_features)
            - "wavelet": Wavelet used
            - "levels": Number of decomposition levels
            - "n_features": Total number of features per pixel
            - "original_shape": Original HSI shape (H, W, bands)
        """
        reflectance = data.reflectance
        h, w, b = reflectance.shape

        # Calculate required padding for SWT

        divisor = 2**self.levels

        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        pad_b = (divisor - b % divisor) % divisor

        needs_padding = pad_h > 0 or pad_w > 0 or pad_b > 0

        # Apply padding if necessary
        if needs_padding:
            cube_padded = np.pad(
                reflectance,
                ((0, pad_h), (0, pad_w), (0, pad_b)),
                mode="reflect",
            )
        else:
            cube_padded = reflectance

        # Apply 3D Stationary Wavelet Transform
        coeffs = pywt.swtn(
            cube_padded,
            self.wavelet,
            level=self.levels,
            start_level=0,
            axes=(0, 1, 2),
        )

        features_list = []

        # Reverse the order to process
        coeffs = list(reversed(coeffs))

        # Process each decomposition level
        for level_coeffs in coeffs:
            # level_coeffs has keys like 'aaa', 'aad', 'ada', etc.
            # (a=approximation/low, d=detail/high)
            # Sort keys for deterministic output
            subband_keys = sorted(level_coeffs.keys())

            for key in subband_keys:
                subband = level_coeffs[key]

                # Crop back to original size if padding was applied
                if needs_padding:
                    subband = subband[:h, :w, :b]

                features_list.append(subband)

        # Concatenate all subbands along the spectral axis
        # Output shape: (H, W, bands * 8 * levels)
        features = np.concatenate(features_list, axis=-1)

        return {
            "features": features,
            "wavelet": self.wavelet,
            "levels": self.levels,
            "n_features": features.shape[-1],
            "original_shape": (h, w, b),
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.wavelet not in pywt.wavelist():
            raise ValueError(
                f"Wavelet '{self.wavelet}' not available. "
                f"Available wavelets: {pywt.wavelist()}"
            )

        if not isinstance(self.levels, int) or self.levels <= 0:
            raise ValueError("levels must be a positive integer")

        # TODO: See if we can add a validation to check if the minimum
        # dimension supports the requested level
        # SWT requires at least 2^level samples
