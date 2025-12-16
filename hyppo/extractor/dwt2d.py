"""Discrete Wavelet Transform (2D) feature extractor for HSI."""

import numpy as np
import pywt

from hyppo.core import HSI
from .base import Extractor


class DWT2DExtractor(Extractor):
    """
    2D Discrete Wavelet Transform feature extractor for spatial texture analysis.

    Applies 2D DWT band-by-band to extract spatial texture features from
    hyperspectral images. Based on the approach described in Kumar & Dikshit (2015)
    for integrating spectral and textural features.
    Captures spatial texture information per band.

    Uses Stationary Wavelet Transform (SWT) to maintain spatial resolution
    for pixel-wise classification.

    Parameters
    ----------
    wavelet : str
        Wavelet name to use (default: 'haar')
    levels : int
        Number of decomposition levels (default: 1)

    References
    ----------
    Kumar, B., & Dikshit, O. (2015). Integrating spectral and textural
    features for urban land cover classification with hyperspectral data.
    IEEE International Geoscience and Remote Sensing Symposium (IGARSS),
    pp. 1653-1656. doi: 10.1109/IGARSS.2015.7326091

    Notes
    -----
    Each band is decomposed into four subbands:
    - LL (Approximation): Low-frequency spatial information
    - LH (Horizontal): Horizontal edges and textures
    - HL (Vertical): Vertical edges and textures
    - HH (Diagonal): Diagonal edges and textures

    These subbands capture different spatial texture characteristics
    useful for distinguishing land cover types, especially in urban areas.
    """

    def __init__(self, wavelet="haar", levels=1):
        """Initialize DWT2D extractor with wavelet parameters."""
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels

    @classmethod
    def feature_name(cls) -> str:
        """Return the feature name."""
        return "dwt2d"

    def _extract(self, data: HSI, **inputs):
        """
        Extract 2D DWT features from a hyperspectral image.

        Args
        ----
        data : HSI
            Hyperspectral image object

        Returns
        -------
        dict
            - "features": DWT coefficient maps stacked (H, W, n_features)
            - "wavelet": Wavelet used
            - "levels": Number of decomposition levels
            - "n_features": Total number of features per pixel
            - "original_shape": Original HSI shape (H, W, bands)
        """
        reflectance = data.reflectance
        h, w, bands = reflectance.shape

        features_list = []

        # Calculate required divisor for SWT
        divisor = 2**self.levels

        # Calculate padding
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        needs_padding = pad_h > 0 or pad_w > 0

        # Process each band separately
        for band_idx in range(bands):
            band_img = reflectance[:, :, band_idx]

            # Apply padding if necessary
            if needs_padding:
                band_img_padded = np.pad(
                    band_img, ((0, pad_h), (0, pad_w)), mode="reflect"
                )
            else:
                band_img_padded = band_img

            # Perform 2D wavelet decomposition using SWT
            coeffs = pywt.swt2(
                band_img_padded, self.wavelet, level=self.levels
            )

            # Reverse the order to process
            coeffs = list(reversed(coeffs))

            # Extract all subbands from all levels
            for level_coeffs in coeffs:
                cA, (cH, cV, cD) = level_coeffs

                # Crop back to original size if padding was applied
                if needs_padding:
                    cA = cA[:h, :w]
                    cH = cH[:h, :w]
                    cV = cV[:h, :w]
                    cD = cD[:h, :w]

                # Stack the four subbands: LL, LH, HL, HH
                level_features = np.stack([cA, cH, cV, cD], axis=-1)
                features_list.append(level_features)

        # Concatenate all features
        features = np.concatenate(features_list, axis=-1)

        return {
            "features": features,
            "wavelet": self.wavelet,
            "levels": self.levels,
            "n_features": features.shape[-1],
            "original_shape": reflectance.shape,
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

        # TODO: See if we can add a validation to check if the minimum spatial
        # dimension supports the requested level
        # SWT requires at least 2^level samples
