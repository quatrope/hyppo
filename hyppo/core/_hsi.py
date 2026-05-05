"""Hyperspectral image data structure."""

import numpy as np

from hyppo.core._hsi_plot import HSIPlotAccessor


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

        Parameters
        ----------
        reflectance : 3D array (height, width, bands)
        wavelengths : 1D array of wavelength values
        mask : Optional 2D boolean mask
        metadata : Optional metadata dictionary
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

    def describe(self) -> dict:
        """Get summary of HSI dimensions and metadata."""
        return {
            "height": int(self.height),
            "width": int(self.width),
            "bands": int(self.n_bands),
            "wavelength_min_nm": float(self.wavelengths.min()),
            "wavelength_max_nm": float(self.wavelengths.max()),
            "valid_pixels": int(self.mask.sum()),
            "total_pixels": int(self.mask.size),
        }

    def pseudo_rgb(
        self,
        r_nm: float = 650.0,
        g_nm: float = 550.0,
        b_nm: float = 450.0,
    ) -> "HSI":
        """Build new HSI with the 3 bands closest to RGB wavelengths."""
        bands = [
            int(np.argmin(np.abs(self.wavelengths - w)))
            for w in (r_nm, g_nm, b_nm)
        ]
        return HSI(
            reflectance=self.reflectance[:, :, bands],
            wavelengths=self.wavelengths[bands],
            mask=self.mask.copy(),
            metadata=dict(self.metadata),
        )

    def crop(
        self,
        rows: slice | None = None,
        cols: slice | None = None,
    ) -> "HSI":
        """Return crop defined by row and column slices."""
        if rows is None and cols is None:
            return self
        row_sel = rows if rows is not None else slice(None)
        col_sel = cols if cols is not None else slice(None)
        return HSI(
            reflectance=self.reflectance[row_sel, col_sel, :],
            wavelengths=self.wavelengths,
            mask=self.mask[row_sel, col_sel],
            metadata=dict(self.metadata),
        )

    def crop_center(self, size: int | None = None) -> "HSI":
        """Return centered square crop of given size, or self if not needed."""
        if size is None or (size >= self.height and size >= self.width):
            return self
        h0 = (self.height - size) // 2
        w0 = (self.width - size) // 2
        return self.crop(
            rows=slice(h0, h0 + size),
            cols=slice(w0, w0 + size),
        )

    @property
    def plot(self) -> "HSIPlotAccessor":
        """Plotting accessor for this HSI."""
        if not hasattr(self, "_plot"):
            self._plot = HSIPlotAccessor(self)
        return self._plot

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
