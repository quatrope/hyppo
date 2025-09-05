import numpy as np
from typing import Optional, Tuple, Dict, Any


class HSI:
    def __init__(
        self,
        reflectance: np.ndarray,
        wavelengths: np.ndarray,
        mask: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
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
        if not isinstance(reflectance, np.ndarray):
            raise TypeError("Reflectance must be a numpy array")
        if reflectance.ndim != 3:
            raise ValueError(
                f"Reflectance must be 3D (height, width, bands), got {reflectance.ndim}D"
            )
        return reflectance.astype(np.float32, copy=False)

    def _validate_wavelengths(self, wavelengths) -> np.ndarray:
        if not isinstance(wavelengths, np.ndarray):
            raise TypeError("Wavelengths must be a numpy array")
        if wavelengths.ndim != 1:
            raise ValueError(f"Wavelengths must be 1D, got {wavelengths.ndim}D")
        return wavelengths.astype(np.float32, copy=False)

    def _validate_mask(self, mask) -> np.ndarray:
        if not isinstance(mask, np.ndarray):
            raise TypeError("Mask must be a numpy array")
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got {mask.ndim}D")
        return mask.astype(bool, copy=False)

    def _validate_dimensions(self):
        height, width, bands = self.reflectance.shape
        if len(self.wavelengths) != bands:
            raise ValueError(
                f"Number of wavelengths ({len(self.wavelengths)}) must match "
                f"number of bands ({bands})"
            )
        if self.mask.shape != (height, width):
            raise ValueError(
                f"Mask shape {self.mask.shape} must match spatial dimensions "
                f"({height}, {width})"
            )

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.reflectance.shape

    @property
    def height(self) -> int:
        return self.reflectance.shape[0]

    @property
    def width(self) -> int:
        return self.reflectance.shape[1]

    @property
    def n_bands(self) -> int:
        return self.reflectance.shape[2]

    def get_band(self, band_idx: int) -> np.ndarray:
        if not 0 <= band_idx < self.n_bands:
            raise IndexError(f"Band index {band_idx} out of range [0, {self.n_bands})")
        return self.reflectance[:, :, band_idx]

    def get_pixel_spectrum(self, row: int, col: int) -> np.ndarray:
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(f"Pixel ({row}, {col}) out of bounds")
        return self.reflectance[row, col, :]

    def get_masked_data(self) -> np.ndarray:
        masked_reflectance = self.reflectance.copy()
        masked_reflectance[~self.mask] = np.nan
        return masked_reflectance

    def get_band_indices(self) -> list:
        return list(range(self.n_bands))

    def __repr__(self) -> str:
        return (
            f"HSI(shape={self.shape}, "
            f"wavelengths={self.wavelengths.min():.1f}-{self.wavelengths.max():.1f}nm, "
            f"valid_pixels={self.mask.sum()}/{self.mask.size})"
        )
