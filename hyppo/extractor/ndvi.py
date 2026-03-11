"""Normalized Difference Vegetation Index (NDVI) extractor.

Provides NDVI feature extraction for hyperspectral images.
"""

import warnings

import numpy as np

from hyppo.core import HSI
from .base import Extractor
from ._spectral_utils import find_and_validate_bands


class NDVIExtractor(Extractor):
    """Normalized Difference Vegetation Index (NDVI) extractor.

    NDVI is a vegetation index widely used to assess vegetation health,
    computed as:

        NDVI = (NIR - Red) / (NIR + Red)

    By default, the bands closest to 660 nm (red) and 850 nm (NIR) are used.

    Parameters
    ----------
    red_wavelength : float, default=660
        Target wavelength for the red band in nanometers.
    nir_wavelength : float, default=850
        Target wavelength for the near-infrared (NIR) band in nanometers.

    References
    ----------
    .. [1] Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974).
           Monitoring vegetation systems in the Great Plains with ERTS.
           NASA/GSFC Final Report.
    """

    def __init__(self, red_wavelength=660, nir_wavelength=850):
        """Initialize NDVI extractor with target wavelengths."""
        super().__init__()
        self.red_wavelength = red_wavelength
        self.nir_wavelength = nir_wavelength

    @classmethod
    def feature_name(cls) -> str:
        """Return the feature name."""
        return "ndvi"

    def _extract(self, data: HSI, **inputs):
        """
        Compute NDVI from a hyperspectral image.

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
                NDVI index values.
            - red_idx : int
                Index of the red band used.
            - nir_idx : int
                Index of the NIR band used.
            - wavelength_used : tuple of float
                Actual wavelengths used (red, NIR).
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

        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-6)
        features = ndvi[:, :, np.newaxis]

        wavelengths = data.wavelengths
        return {
            "features": features,
            "red_idx": red_idx,
            "nir_idx": nir_idx,
            "wavelength_used": (wavelengths[red_idx], wavelengths[nir_idx]),
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
