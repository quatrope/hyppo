"""
Tests for the HSI (Hyperspectral Image) module.
"""

import numpy as np
import pytest
from hyppo.core import HSI


class TestHSI:
    """Tests for HSI class."""

    def test_initialization_basic(self):
        """Test basic HSI initialization."""
        reflectance = np.random.rand(10, 10, 50)
        wavelengths = np.linspace(400, 1000, 50)

        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)

        assert hsi.reflectance.shape == (10, 10, 50)
        assert hsi.wavelengths.shape == (50,)
        assert np.allclose(hsi.reflectance, reflectance)
        assert np.allclose(hsi.wavelengths, wavelengths)

    def test_initialization_with_mask(self):
        """Test HSI initialization with mask."""
        reflectance = np.random.rand(10, 10, 50)
        wavelengths = np.linspace(400, 1000, 50)
        mask = np.ones((10, 10), dtype=bool)
        mask[5:, :] = False

        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths, mask=mask)

        assert hsi.mask.shape == (10, 10)
        assert hsi.mask.sum() == 50

    def test_initialization_with_metadata(self):
        """Test HSI initialization with metadata."""
        reflectance = np.random.rand(10, 10, 50)
        wavelengths = np.linspace(400, 1000, 50)
        metadata = {"source": "test", "date": "2025-01-01"}

        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths, metadata=metadata)

        assert hsi.metadata == metadata

    def test_reflectance_not_array(self):
        """Test error when reflectance is not numpy array."""
        with pytest.raises(TypeError, match="Reflectance must be a numpy array"):
            HSI(reflectance=[[1, 2], [3, 4]], wavelengths=np.array([400, 500]))

    def test_reflectance_wrong_dimensions(self):
        """Test error when reflectance is not 3D."""
        with pytest.raises(ValueError, match="Reflectance must be 3D"):
            HSI(reflectance=np.random.rand(10, 50), wavelengths=np.linspace(400, 1000, 50))

    def test_wavelengths_not_array(self):
        """Test error when wavelengths is not numpy array."""
        with pytest.raises(TypeError, match="Wavelengths must be a numpy array"):
            HSI(reflectance=np.random.rand(10, 10, 50), wavelengths=[400, 500, 600])

    def test_wavelengths_wrong_dimensions(self):
        """Test error when wavelengths is not 1D."""
        with pytest.raises(ValueError, match="Wavelengths must be 1D"):
            HSI(reflectance=np.random.rand(10, 10, 50), wavelengths=np.random.rand(50, 1))

    def test_mask_not_array(self):
        """Test error when mask is not numpy array."""
        with pytest.raises(TypeError, match="Mask must be a numpy array"):
            HSI(
                reflectance=np.random.rand(10, 10, 50),
                wavelengths=np.linspace(400, 1000, 50),
                mask=[[True, False], [False, True]],
            )

    def test_mask_wrong_dimensions(self):
        """Test error when mask is not 2D."""
        with pytest.raises(ValueError, match="Mask must be 2D"):
            HSI(
                reflectance=np.random.rand(10, 10, 50),
                wavelengths=np.linspace(400, 1000, 50),
                mask=np.ones((10,), dtype=bool),
            )

    def test_wavelength_band_mismatch(self):
        """Test error when wavelengths don't match bands."""
        with pytest.raises(ValueError, match="Number of wavelengths .* must match number of bands"):
            HSI(reflectance=np.random.rand(10, 10, 50), wavelengths=np.linspace(400, 1000, 30))

    def test_mask_shape_mismatch(self):
        """Test error when mask shape doesn't match spatial dimensions."""
        with pytest.raises(ValueError, match="Mask shape .* must match spatial dimensions"):
            HSI(
                reflectance=np.random.rand(10, 10, 50),
                wavelengths=np.linspace(400, 1000, 50),
                mask=np.ones((5, 5), dtype=bool),
            )

    def test_shape_property(self, sample_hsi):
        """Test shape property."""
        assert sample_hsi.shape == sample_hsi.reflectance.shape

    def test_height_property(self, sample_hsi):
        """Test height property."""
        assert sample_hsi.height == sample_hsi.reflectance.shape[0]

    def test_width_property(self, sample_hsi):
        """Test width property."""
        assert sample_hsi.width == sample_hsi.reflectance.shape[1]

    def test_n_bands_property(self, sample_hsi):
        """Test n_bands property."""
        assert sample_hsi.n_bands == sample_hsi.reflectance.shape[2]

    def test_get_band(self, sample_hsi):
        """Test get_band method."""
        band = sample_hsi.get_band(0)

        assert band.shape == (sample_hsi.height, sample_hsi.width)
        assert np.array_equal(band, sample_hsi.reflectance[:, :, 0])

    def test_get_band_invalid_negative(self, sample_hsi):
        """Test get_band with negative index."""
        with pytest.raises(IndexError, match="Band index .* out of range"):
            sample_hsi.get_band(-1)

    def test_get_band_invalid_high(self, sample_hsi):
        """Test get_band with too high index."""
        with pytest.raises(IndexError, match="Band index .* out of range"):
            sample_hsi.get_band(sample_hsi.n_bands)

    def test_get_pixel_spectrum(self, sample_hsi):
        """Test get_pixel_spectrum method."""
        spectrum = sample_hsi.get_pixel_spectrum(0, 0)

        assert spectrum.shape == (sample_hsi.n_bands,)
        assert np.array_equal(spectrum, sample_hsi.reflectance[0, 0, :])

    def test_get_pixel_spectrum_invalid_row(self, sample_hsi):
        """Test get_pixel_spectrum with invalid row."""
        with pytest.raises(IndexError, match="Pixel .* out of bounds"):
            sample_hsi.get_pixel_spectrum(sample_hsi.height, 0)

    def test_get_pixel_spectrum_invalid_col(self, sample_hsi):
        """Test get_pixel_spectrum with invalid column."""
        with pytest.raises(IndexError, match="Pixel .* out of bounds"):
            sample_hsi.get_pixel_spectrum(0, sample_hsi.width)

    def test_get_masked_data(self, sample_hsi):
        """Test get_masked_data method."""
        masked = sample_hsi.get_masked_data()

        assert masked.shape == sample_hsi.shape
        assert np.all(np.isfinite(masked[sample_hsi.mask]))

    def test_get_masked_data_with_partial_mask(self):
        """Test get_masked_data with partial mask."""
        reflectance = np.ones((5, 5, 10))
        wavelengths = np.linspace(400, 1000, 10)
        mask = np.ones((5, 5), dtype=bool)
        mask[2:, :] = False

        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths, mask=mask)
        masked = hsi.get_masked_data()

        assert np.all(np.isfinite(masked[mask]))
        assert np.all(np.isnan(masked[~mask]))

    def test_get_band_indices(self, sample_hsi):
        """Test get_band_indices method."""
        indices = sample_hsi.get_band_indices()

        assert len(indices) == sample_hsi.n_bands
        assert indices == list(range(sample_hsi.n_bands))

    def test_repr(self, sample_hsi):
        """Test __repr__ method."""
        repr_str = repr(sample_hsi)

        assert "HSI(" in repr_str
        assert f"shape={sample_hsi.shape}" in repr_str
        assert "wavelengths=" in repr_str
        assert "valid_pixels=" in repr_str

    def test_small_hsi_fixture(self, small_hsi):
        """Test small HSI fixture."""
        assert small_hsi.reflectance.shape == (3, 3, 5)
        assert len(small_hsi.wavelengths) == 5
        assert np.array_equal(small_hsi.wavelengths, [450, 550, 650, 750, 850])
