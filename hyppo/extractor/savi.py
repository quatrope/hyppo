"""Soil-Adjusted Vegetation Index (SAVI) extractor for hyperspectral images."""

import warnings

import numpy as np

from hyppo.core import HSI
from ._spectral_utils import find_and_validate_bands
from .base import Extractor


class SAVIExtractor(Extractor):
    """
    Soil-Adjusted Vegetation Index (SAVI) extractor for hyperspectral images.

    SAVI is a vegetation index designed to reduce the influence of soil
    brightness in areas with sparse vegetation. It modifies the NDVI by
    introducing a soil brightness correction factor L:

        SAVI = ((NIR - Red) * (1 + L)) / (NIR + Red + L)

    By default, the bands closest to 660 nm (red) and 850 nm (NIR) are used.

    Parameters
    ----------
    red_wavelength : float, default=660
        Target wavelength for the red band in nanometers.
    nir_wavelength : float, default=850
        Target wavelength for the near-infrared (NIR) band in nanometers.
    L : float, default=0.5
        Soil brightness correction factor (typically between 0 and 1).

    References
    ----------
    .. [1] Huete, A. R. (1988). A soil-adjusted vegetation index (SAVI).
           Remote Sensing of Environment, 25(3), 295–309.
           https://doi.org/10.1016/0034-4257(88)90106-X
    """

    def __init__(self, red_wavelength=660, nir_wavelength=850, L=0.5):
        """Initialize SAVI extractor.

        With target wavelengths and soil brightness correction.
        """
        super().__init__()
        self.red_wavelength = red_wavelength
        self.nir_wavelength = nir_wavelength
        self.L = L

    @classmethod
    def feature_name(cls) -> str:
        """Return the feature name."""
        return "savi"

    def _extract(self, data: HSI, **inputs):
        """Compute the SAVI index from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data and
            wavelength information.

        Returns
        -------
        dict
            Dictionary containing:

            - features : ndarray of shape (H, W, 1)
                SAVI index values.
            - red_idx : int
                Index of the red band used.
            - nir_idx : int
                Index of the NIR band used.
            - wavelength_used : tuple of float
                Actual wavelengths used (red, NIR).
            - brightness_correction : float
                L parameter used.
            - original_shape : tuple of int
                Shape of the original HSI cube.
        """
        (red_idx, red), (nir_idx, nir) = find_and_validate_bands(
            data,
            [
                (self.red_wavelength, "Red"),
                (self.nir_wavelength, "NIR"),
            ],
        )

        # Calculate SAVI
        savi = ((nir - red) * (1 + self.L)) / (nir + red + self.L)
        features = savi[:, :, np.newaxis]

        wavelength = data.wavelengths
        return {
            "features": features,
            "red_idx": red_idx,
            "nir_idx": nir_idx,
            "wavelength_used": (wavelength[red_idx], wavelength[nir_idx]),
            "brightness_correction": self.L,
            "original_shape": data.reflectance.shape,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.red_wavelength <= 0:
            raise ValueError("red_wavelength must be positive")
        if self.nir_wavelength <= 0:
            raise ValueError("nir_wavelength must be positive")
        if self.red_wavelength >= self.nir_wavelength:
            warnings.warn("red_wavelength should be less than nir_wavelength")
        if not (0 <= self.L <= 1):
            raise ValueError("L must be between 0 and 1")
