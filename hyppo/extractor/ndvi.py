from .base import Extractor
from hyppo.core import HSI
import numpy as np
import warnings


class NDVIExtractor(Extractor):
    """
    Normalized Difference Vegetation Index (NDVI) extractor for hyperspectral images.

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
        super().__init__()
        self.red_wavelength = red_wavelength
        self.nir_wavelength = nir_wavelength

    def extract(self, data: HSI, **inputs):
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

            - features : ndarray of shape (H, W)
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
        reflectance = data.reflectance
        wavelengths = data.wavelengths

        # Check wavelength availability
        if len(wavelengths) == 0:
            raise ValueError("No wavelength information available")

        # Find closest band
        red_idx = np.argmin(np.abs(wavelengths - self.red_wavelength))
        nir_idx = np.argmin(np.abs(wavelengths - self.nir_wavelength))

        red = reflectance[:, :, red_idx].astype(float)
        nir = reflectance[:, :, nir_idx].astype(float)

        # Check if wavelengths are far from target
        red_diff = abs(wavelengths[red_idx] - self.red_wavelength)
        nir_diff = abs(wavelengths[nir_idx] - self.nir_wavelength)

        if red_diff > 50 or nir_diff > 50:  # 50nm tolerance
            warnings.warn(
                f"Bands far from target wavelengths: "
                f"{'Red' if red_diff > 50 else ''} "
                f"{'NIR' if nir_diff > 50 else ''}".strip()
            )

        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-6)

        return {
            "features": ndvi,
            "red_idx": red_idx,
            "nir_idx": nir_idx,
            "wavelength_used": (wavelengths[red_idx], wavelengths[nir_idx]),
            "original_shape": reflectance.shape,
        }

    def validate(self):
        """Validate extractor parameters."""
        if self.red_wavelength <= 0:
            raise ValueError("red_wavelength must be positive")
        if self.nir_wavelength <= 0:
            raise ValueError("nir_wavelength must be positive")
        if self.red_wavelength >= self.nir_wavelength:
            warnings.warn("red_wavelength should be less than nir_wavelength")
