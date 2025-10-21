"""Hyperspectral image data structure."""

import numpy as np


class HSI:
    """Represents a hyperspectral image with reflectance and metadata."""

    def __init__(
        self,
        reflectance: np.ndarray,
        wavelengths: np.ndarray,
        mask: np.ndarray | None = None,
        metadata: dict | None = None,
    ):
        """
        Initialize HSI with reflectance data and wavelengths.

        Args:
            reflectance: 3D array (height, width, bands)
            wavelengths: 1D array of wavelength values
            mask: Optional 2D boolean mask
            metadata: Optional metadata dictionary
        """
        self.reflectance = self._validate_reflectance(reflectance)
        self.wavelengths = self._validate_wavelengths(wavelengths)
        self.mask = (
            self._validate_mask(mask)
            if mask is not None
            else np.ones(self.reflectance.shape[:2], dtype=bool)
        )
        self.metadata = metadata or {}

        self._validate_dimensions()

    def _validate_reflectance(self, reflectance) -> np.ndarray:
        """Validate reflectance array."""
        if not isinstance(reflectance, np.ndarray):
            raise TypeError("Reflectance must be a numpy array")
        if reflectance.ndim != 3:
            msg = (
                f"Reflectance must be 3D (height, width, bands), "
                f"got {reflectance.ndim}D"
            )
            raise ValueError(msg)
        return reflectance.astype(np.float32, copy=False)

    def _validate_wavelengths(self, wavelengths) -> np.ndarray:
        """Validate wavelengths array."""
        if not isinstance(wavelengths, np.ndarray):
            raise TypeError("Wavelengths must be a numpy array")
        if wavelengths.ndim != 1:
            msg = f"Wavelengths must be 1D, got {wavelengths.ndim}D"
            raise ValueError(msg)
        return wavelengths.astype(np.float32, copy=False)

    def _validate_mask(self, mask) -> np.ndarray:
        """Validate mask array."""
        if not isinstance(mask, np.ndarray):
            raise TypeError("Mask must be a numpy array")
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got {mask.ndim}D")
        return mask.astype(bool, copy=False)

    def _validate_dimensions(self):
        """Validate dimensions match between arrays."""
        height, width, bands = self.reflectance.shape
        if len(self.wavelengths) != bands:
            msg = (
                f"Number of wavelengths ({len(self.wavelengths)}) "
                f"must match number of bands ({bands})"
            )
            raise ValueError(msg)
        if self.mask.shape != (height, width):
            msg = (
                f"Mask shape {self.mask.shape} must match spatial "
                f"dimensions ({height}, {width})"
            )
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get shape of reflectance array."""
        return self.reflectance.shape

    @property
    def height(self) -> int:
        """Get height of image."""
        return self.reflectance.shape[0]

    @property
    def width(self) -> int:
        """Get width of image."""
        return self.reflectance.shape[1]

    @property
    def n_bands(self) -> int:
        """Get number of spectral bands."""
        return self.reflectance.shape[2]

    def get_band(self, band_idx: int) -> np.ndarray:
        """Get single band as 2D array."""
        if not 0 <= band_idx < self.n_bands:
            msg = f"Band index {band_idx} out of range [0, {self.n_bands})"
            raise IndexError(msg)
        return self.reflectance[:, :, band_idx]

    def get_pixel_spectrum(self, row: int, col: int) -> np.ndarray:
        """Get spectrum at pixel location."""
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(f"Pixel ({row}, {col}) out of bounds")
        return self.reflectance[row, col, :]

    def get_masked_data(self) -> np.ndarray:
        """Get reflectance with mask applied (invalid pixels set to NaN)."""
        masked_reflectance = self.reflectance.copy()
        masked_reflectance[~self.mask] = np.nan
        return masked_reflectance

    def get_band_indices(self) -> list:
        """Get list of band indices."""
        return list(range(self.n_bands))

    def __repr__(self) -> str:
        """Return string representation of HSI."""
        wl_min = self.wavelengths.min()
        wl_max = self.wavelengths.max()
        valid = self.mask.sum()
        total = self.mask.size
        return (
            f"HSI(shape={self.shape}, "
            f"wavelengths={wl_min:.1f}-{wl_max:.1f}nm, "
            f"valid_pixels={valid}/{total})"
        )
