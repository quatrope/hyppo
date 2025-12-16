"""Normalized Difference Water Index (NDWI) extractor.

Provides NDWI feature extraction for hyperspectral images.
"""

import warnings

import numpy as np

from hyppo.core import HSI
from .base import Extractor


class NDWIExtractor(Extractor):
    """Normalized Difference Water Index (NDWI) extractor.

    NDWI is a water-related vegetation index computed as:

        NDWI = (Green - NIR) / (Green + NIR)

    By default, the bands closest to 560 nm (green) and 850 nm (NIR) are used.

    Parameters
    ----------
    green_wavelength : float, default=560
        Target wavelength for the green band in nanometers.
    nir_wavelength : float, default=850
        Target wavelength for the near-infrared (NIR) band in nanometers.

    References
    ----------
    .. [1] McFeeters, S. K. (1996). The use of the Normalized Difference 
           Water Index (NDWI) in the delineation of open water features. 
           International Journal of Remote Sensing, 17(7), 1425-1432.
    """

    def __init__(self, green_wavelength=560, nir_wavelength=850):
        """Initialize NDWI extractor with target wavelengths."""
        super().__init__()
        self.green_wavelength = green_wavelength
        self.nir_wavelength = nir_wavelength

    def _extract(self, data: HSI, **inputs):
        """
        Compute NDWI from a hyperspectral image.

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
                NDWI index values.
            - green_idx : int
                Index of the green band used.
            - nir_idx : int
                Index of the NIR band used.
            - wavelength_used : tuple of float
                Actual wavelengths used (green, NIR).
            - original_shape : tuple of int
                Shape of the original HSI cube.
        """
        reflectance = data.reflectance
        wavelength = data.wavelengths

        # Check wavelength availability
        if len(wavelength) == 0:
            raise ValueError("No wavelength information available")

        # Find closest band
        green_idx = np.argmin(np.abs(wavelength - self.green_wavelength))
        nir_idx = np.argmin(np.abs(wavelength - self.nir_wavelength))

        green = reflectance[:, :, green_idx].astype(float)
        nir = reflectance[:, :, nir_idx].astype(float)

        # Check if wavelengths are far from target
        green_diff = abs(wavelength[green_idx] - self.green_wavelength)
        nir_diff = abs(wavelength[nir_idx] - self.nir_wavelength)

        if green_diff > 50 or nir_diff > 50:  # 50nm tolerance
            warnings.warn(
                f"Bands far from target wavelengths: "
                f"{'Green' if green_diff > 50 else ''} "
                f"{'NIR' if nir_diff > 50 else ''}".strip()
            )

        # Calculate NDWI
        ndwi = (green - nir) / (green + nir + 1e-6)
        features = ndwi[:, :, np.newaxis]

        return {
            "features": features,
            "green_idx": green_idx,
            "nir_idx": nir_idx,
            "wavelength_used": (wavelength[green_idx], wavelength[nir_idx]),
            "original_shape": reflectance.shape,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.green_wavelength <= 0:
            raise ValueError("green_wavelength must be positive")
        if self.nir_wavelength <= 0:
            raise ValueError("nir_wavelength must be positive")
        if self.green_wavelength >= self.nir_wavelength:
            warnings.warn(
                "green_wavelength should be less than nir_wavelength"
            )
