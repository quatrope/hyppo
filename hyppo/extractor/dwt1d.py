"""Discrete Wavelet Transform 1D feature extractor."""

import numpy as np
import pywt

from hyppo.core import HSI
from .base import Extractor
from ._validators import validate_positive_int


class DWT1DExtractor(Extractor):
    """
    Discrete Wavelet Transform (1D) feature extractor for hyperspectral images.

    Applies a 1D DWT to each pixel's spectral signature to extract
    multiscale spectral features.  Implements the approach described in
    Bruce et al. (2002) for hyperspectral dimensionality reduction.


    Parameters
    ----------
    wavelet : str, optional
        Wavelet name to use (default: 'db4'). Lower-order wavelets like
        'haar', 'db2', 'db4' tend to be more stable across applications.
    mode : str, optional
        Signal extension mode (default: 'symmetric').
    levels : int, optional
        Number of decomposition levels (default: 3). If None, uses maximum
        possible level based on signal length as recommended in Bruce(2002).

    References
    ----------
    Bruce, K., Koger, C., & Li, J. (2002). Dimensionality reduction of
    hyperspectral data using discrete wavelet transform feature extraction.
    *IEEE Transactions on Geoscience and Remote Sensing*, 40(10),
    2331–2338. https://doi.org/10.1109/TGRS.2002.804721

    Notes
    -----
    Decomposes each pixel's spectral signature into:
    - cA: Approximation coefficients (low-frequency spectral info)
    - cD_n, ..., cD_1: Detail coefficients at each level (high-frequency)

    The concatenated structure is: [cA_n, cD_n, cD_n-1, ..., cD_1]
    where n is the decomposition level.
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
            - "features": DWT-transformed array (H, W, n_features)
            - "wavelet": Wavelet used
            - "mode": Signal extension mode
            - "level": Number of decomposition levels used
            - "n_features": Total number of features per pixel
            - "original_shape": Original HSI shape (H, W, bands)
        """

        reflectance = data.reflectance
        h, w, bands = reflectance.shape

        # Determine decomposition level
        if self.levels is None:
            # Use maximum level as in Bruce et al. (2002)
            max_level = pywt.dwt_max_level(bands, self.wavelet)
            actual_level = max_level
        else:
            actual_level = self.levels

        # Reshape for pixel-wise processing
        reflectance_reshaped = reflectance.reshape(-1, bands)

        features_list = []

        # Apply DWT to each pixel's spectral signature
        for i in range(reflectance_reshaped.shape[0]):
            pixel_spectrum = reflectance_reshaped[i, :]

            # Apply 1D DWT to the spectral signature
            coefficients = pywt.wavedec(
                pixel_spectrum,
                self.wavelet,
                mode=self.mode,
                level=actual_level,
            )

            # Concatenate all coefficients as features
            # The structure is: [cA_n, cD_n, cD_n-1, ..., cD_1]
            pixel_features = np.concatenate(coefficients)
            features_list.append(pixel_features)

        features_2d = np.array(features_list)
        features = features_2d.reshape(h, w, -1)

        return {
            "features": features,
            "wavelet": self.wavelet,
            "mode": self.mode,
            "levels": self.levels,
            "n_features": features.shape[1],
            "original_shape": (h, w, bands),
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.wavelet not in pywt.wavelist():
            raise ValueError(
                f"Wavelet '{self.wavelet}' not available. "
                f"Available wavelets: {pywt.wavelist()}"
            )

        if self.mode not in pywt.Modes.modes:
            raise ValueError(f"Mode '{self.mode}' not available")

        if self.levels is not None:
            validate_positive_int(self.levels, "levels")

            # Validate that levels don't exceed maximum
            max_level = pywt.dwt_max_level(
                data.reflectance.shape[2], self.wavelet
            )
            if self.levels > max_level:
                raise ValueError(
                    f"levels={self.levels} exceeds maximum "
                    f"level {max_level} "
                    f"for signal length "
                    f"{data.reflectance.shape[2]} "
                    f"and wavelet '{self.wavelet}'"
                )
