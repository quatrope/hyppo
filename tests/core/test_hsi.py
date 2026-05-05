"""Tests for the HSI (Hyperspectral Image) module."""

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.core._hsi_plot import HSIPlotAccessor

matplotlib.use("Agg")


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

        hsi = HSI(
            reflectance=reflectance, wavelengths=wavelengths, metadata=metadata
        )

        assert hsi.metadata == metadata

    def test_reflectance_not_array(self):
        """Test error when reflectance is not numpy array."""
        with pytest.raises(
            TypeError, match="Reflectance must be a numpy array"
        ):
            HSI(reflectance=[[1, 2], [3, 4]], wavelengths=np.array([400, 500]))

    def test_reflectance_wrong_dimensions(self):
        """Test error when reflectance is not 3D."""
        with pytest.raises(ValueError, match="Reflectance must be 3D"):
            HSI(
                reflectance=np.random.rand(10, 50),
                wavelengths=np.linspace(400, 1000, 50),
            )

    def test_wavelengths_not_array(self):
        """Test error when wavelengths is not numpy array."""
        with pytest.raises(
            TypeError, match="Wavelengths must be a numpy array"
        ):
            HSI(
                reflectance=np.random.rand(10, 10, 50),
                wavelengths=[400, 500, 600],
            )

    def test_wavelengths_wrong_dimensions(self):
        """Test error when wavelengths is not 1D."""
        with pytest.raises(ValueError, match="Wavelengths must be 1D"):
            HSI(
                reflectance=np.random.rand(10, 10, 50),
                wavelengths=np.random.rand(50, 1),
            )

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
        with pytest.raises(
            ValueError,
            match="Number of wavelengths .* must match number of bands",
        ):
            HSI(
                reflectance=np.random.rand(10, 10, 50),
                wavelengths=np.linspace(400, 1000, 30),
            )

    def test_mask_shape_mismatch(self):
        """Test error when mask shape doesn't match spatial dimensions."""
        with pytest.raises(
            ValueError, match="Mask shape .* must match spatial dimensions"
        ):
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


class TestHSIDescribe:
    """Tests for HSI.describe method."""

    def test_describe_returns_expected_keys(self, small_hsi):
        """Test that describe returns all documented keys."""
        # Act:
        desc = small_hsi.describe()

        # Assert:
        expected_keys = {
            "height",
            "width",
            "bands",
            "wavelength_min_nm",
            "wavelength_max_nm",
            "valid_pixels",
            "total_pixels",
        }
        assert set(desc.keys()) == expected_keys

    def test_describe_dimensions(self, small_hsi):
        """Test describe reports height/width/bands instead of tuple."""
        # Act:
        desc = small_hsi.describe()

        # Assert:
        assert desc["height"] == 3
        assert desc["width"] == 3
        assert desc["bands"] == 5

    def test_describe_wavelength_range(self, small_hsi):
        """Test describe reports wavelength min/max in nm."""
        # Act:
        desc = small_hsi.describe()

        # Assert:
        assert desc["wavelength_min_nm"] == 450.0
        assert desc["wavelength_max_nm"] == 850.0

    def test_describe_valid_pixels(self, small_hsi):
        """Test describe reports valid pixel count from mask."""
        # Act:
        desc = small_hsi.describe()

        # Assert:
        assert desc["valid_pixels"] == 9
        assert desc["total_pixels"] == 9

    def test_describe_with_partial_mask(self):
        """Test describe reflects partial mask in valid_pixels."""
        # Arrange:
        reflectance = np.ones((4, 4, 3), dtype=np.float32)
        wavelengths = np.array([400, 500, 600], dtype=np.float32)
        mask = np.ones((4, 4), dtype=bool)
        mask[2:, :] = False
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths, mask=mask)

        # Act:
        desc = hsi.describe()

        # Assert:
        assert desc["valid_pixels"] == 8
        assert desc["total_pixels"] == 16

    def test_describe_returns_python_scalars(self, small_hsi):
        """Test describe values are Python scalars, not numpy types."""
        # Act:
        desc = small_hsi.describe()

        # Assert:
        assert type(desc["height"]) is int
        assert type(desc["width"]) is int
        assert type(desc["bands"]) is int
        assert type(desc["valid_pixels"]) is int
        assert type(desc["total_pixels"]) is int
        assert type(desc["wavelength_min_nm"]) is float
        assert type(desc["wavelength_max_nm"]) is float


class TestHSIPseudoRGB:
    """Tests for HSI.pseudo_rgb method."""

    def test_pseudo_rgb_default_wavelengths(self, small_hsi):
        """Test default R=650, G=550, B=450 nm selection."""
        # Arrange: small_hsi wavelengths = [450, 550, 650, 750, 850]
        # Act:
        rgb = small_hsi.pseudo_rgb()

        # Assert:
        assert isinstance(rgb, HSI)
        assert rgb.n_bands == 3
        assert np.array_equal(rgb.wavelengths, [650.0, 550.0, 450.0])
        expected = small_hsi.reflectance[:, :, [2, 1, 0]]
        assert np.array_equal(rgb.reflectance, expected)

    def test_pseudo_rgb_preserves_spatial_shape(self, small_hsi):
        """Test pseudo_rgb keeps height and width."""
        # Act:
        rgb = small_hsi.pseudo_rgb()

        # Assert:
        assert rgb.height == small_hsi.height
        assert rgb.width == small_hsi.width

    def test_pseudo_rgb_inexact_wavelengths(self, small_hsi):
        """Test that nearest band is picked when wavelength is inexact."""
        # Act: r_nm=640 should round to 650 (idx 2)
        rgb = small_hsi.pseudo_rgb(r_nm=640, g_nm=540, b_nm=440)

        # Assert:
        assert np.array_equal(rgb.wavelengths, [650.0, 550.0, 450.0])

    def test_pseudo_rgb_custom_wavelengths(self, small_hsi):
        """Test custom wavelength selection."""
        # Act:
        rgb = small_hsi.pseudo_rgb(r_nm=850, g_nm=750, b_nm=650)

        # Assert:
        assert np.array_equal(rgb.wavelengths, [850.0, 750.0, 650.0])

    def test_pseudo_rgb_propagates_mask(self):
        """Test mask is propagated to new HSI."""
        # Arrange:
        reflectance = np.random.rand(4, 4, 5).astype(np.float32)
        wavelengths = np.array([450, 550, 650, 750, 850], dtype=np.float32)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths, mask=mask)

        # Act:
        rgb = hsi.pseudo_rgb()

        # Assert:
        assert np.array_equal(rgb.mask, mask)

    def test_pseudo_rgb_propagates_metadata(self, small_hsi):
        """Test metadata is propagated."""
        # Arrange:
        reflectance = np.random.rand(3, 3, 5).astype(np.float32)
        wavelengths = np.array([450, 550, 650, 750, 850], dtype=np.float32)
        metadata = {"sensor": "NEON", "date": "2025-01-01"}
        hsi = HSI(
            reflectance=reflectance,
            wavelengths=wavelengths,
            metadata=metadata,
        )

        # Act:
        rgb = hsi.pseudo_rgb()

        # Assert:
        assert rgb.metadata == metadata

    def test_pseudo_rgb_returns_new_instance(self, small_hsi):
        """Test pseudo_rgb does not mutate original HSI."""
        # Arrange:
        original_shape = small_hsi.shape

        # Act:
        rgb = small_hsi.pseudo_rgb()

        # Assert:
        assert rgb is not small_hsi
        assert small_hsi.shape == original_shape
        assert small_hsi.n_bands == 5


class TestHSICropCenter:
    """Tests for HSI.crop_center method."""

    def test_crop_center_basic(self, sample_hsi):
        """Test crop_center produces correct centered shape."""
        # Act:
        cropped = sample_hsi.crop_center(4)

        # Assert:
        assert cropped.shape == (4, 4, 50)
        expected = sample_hsi.reflectance[3:7, 3:7, :]
        assert np.array_equal(cropped.reflectance, expected)

    def test_crop_center_none_returns_self(self, sample_hsi):
        """Test crop_center with None returns same instance."""
        # Act:
        cropped = sample_hsi.crop_center(None)

        # Assert:
        assert cropped is sample_hsi

    def test_crop_center_size_larger_than_image_returns_self(self, sample_hsi):
        """Test crop_center larger than image returns same instance."""
        # Act:
        cropped = sample_hsi.crop_center(20)

        # Assert:
        assert cropped is sample_hsi

    def test_crop_center_preserves_wavelengths(self, sample_hsi):
        """Test crop_center preserves the wavelengths array."""
        # Act:
        cropped = sample_hsi.crop_center(4)

        # Assert:
        assert np.array_equal(cropped.wavelengths, sample_hsi.wavelengths)

    def test_crop_center_crops_mask(self):
        """Test mask is cropped to the same region."""
        # Arrange:
        reflectance = np.random.rand(10, 10, 5).astype(np.float32)
        wavelengths = np.array([400, 500, 600, 700, 800], dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths, mask=mask)

        # Act:
        cropped = hsi.crop_center(4)

        # Assert:
        assert cropped.mask.shape == (4, 4)
        assert cropped.mask.all()

    def test_crop_center_propagates_metadata(self):
        """Test metadata is propagated to centered crop."""
        # Arrange:
        reflectance = np.random.rand(10, 10, 5).astype(np.float32)
        wavelengths = np.array([400, 500, 600, 700, 800], dtype=np.float32)
        metadata = {"sensor": "NEON"}
        hsi = HSI(
            reflectance=reflectance,
            wavelengths=wavelengths,
            metadata=metadata,
        )

        # Act:
        cropped = hsi.crop_center(4)

        # Assert:
        assert cropped.metadata == metadata

    def test_crop_center_default_size_returns_self(self, sample_hsi):
        """Test crop_center without args returns same instance."""
        # Act:
        cropped = sample_hsi.crop_center()

        # Assert:
        assert cropped is sample_hsi


class TestHSICrop:
    """Tests for HSI.crop method (slice-based bbox)."""

    def test_crop_rows_and_cols(self, sample_hsi):
        """Test crop with both row and column slices."""
        # Act:
        cropped = sample_hsi.crop(rows=slice(2, 6), cols=slice(3, 7))

        # Assert:
        assert cropped.shape == (4, 4, 50)
        expected = sample_hsi.reflectance[2:6, 3:7, :]
        assert np.array_equal(cropped.reflectance, expected)

    def test_crop_rows_only(self, sample_hsi):
        """Test crop with only rows slice keeps all columns."""
        # Act:
        cropped = sample_hsi.crop(rows=slice(2, 6))

        # Assert:
        assert cropped.shape == (4, sample_hsi.width, 50)
        expected = sample_hsi.reflectance[2:6, :, :]
        assert np.array_equal(cropped.reflectance, expected)

    def test_crop_cols_only(self, sample_hsi):
        """Test crop with only cols slice keeps all rows."""
        # Act:
        cropped = sample_hsi.crop(cols=slice(3, 7))

        # Assert:
        assert cropped.shape == (sample_hsi.height, 4, 50)
        expected = sample_hsi.reflectance[:, 3:7, :]
        assert np.array_equal(cropped.reflectance, expected)

    def test_crop_no_args_returns_self(self, sample_hsi):
        """Test crop with no slices returns same instance."""
        # Act:
        cropped = sample_hsi.crop()

        # Assert:
        assert cropped is sample_hsi

    def test_crop_preserves_wavelengths(self, sample_hsi):
        """Test crop preserves the wavelengths array."""
        # Act:
        cropped = sample_hsi.crop(rows=slice(2, 6), cols=slice(3, 7))

        # Assert:
        assert np.array_equal(cropped.wavelengths, sample_hsi.wavelengths)

    def test_crop_crops_mask(self):
        """Test mask is cropped to the same bbox."""
        # Arrange:
        reflectance = np.random.rand(10, 10, 5).astype(np.float32)
        wavelengths = np.array([400, 500, 600, 700, 800], dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 3:7] = True
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths, mask=mask)

        # Act:
        cropped = hsi.crop(rows=slice(2, 6), cols=slice(3, 7))

        # Assert:
        assert cropped.mask.shape == (4, 4)
        assert cropped.mask.all()

    def test_crop_propagates_metadata(self):
        """Test metadata is propagated to cropped HSI."""
        # Arrange:
        reflectance = np.random.rand(10, 10, 5).astype(np.float32)
        wavelengths = np.array([400, 500, 600, 700, 800], dtype=np.float32)
        metadata = {"sensor": "NEON"}
        hsi = HSI(
            reflectance=reflectance,
            wavelengths=wavelengths,
            metadata=metadata,
        )

        # Act:
        cropped = hsi.crop(rows=slice(0, 5), cols=slice(0, 5))

        # Assert:
        assert cropped.metadata == metadata

    def test_crop_negative_indices(self, sample_hsi):
        """Test crop accepts numpy-style negative indices."""
        # Act:
        cropped = sample_hsi.crop(rows=slice(-3, None), cols=slice(-3, None))

        # Assert:
        assert cropped.shape == (3, 3, 50)
        expected = sample_hsi.reflectance[-3:, -3:, :]
        assert np.array_equal(cropped.reflectance, expected)

    def test_crop_returns_new_instance(self, sample_hsi):
        """Test crop with slices returns new HSI, not self."""
        # Act:
        cropped = sample_hsi.crop(rows=slice(0, 4))

        # Assert:
        assert cropped is not sample_hsi


class TestHSIPlot:
    """Tests for HSI.plot accessor."""

    def test_plot_returns_accessor(self, sample_hsi):
        """Test plot property returns HSIPlotAccessor instance."""
        # Act:
        accessor = sample_hsi.plot

        # Assert:
        assert isinstance(accessor, HSIPlotAccessor)

    def test_plot_is_cached(self, sample_hsi):
        """Test plot returns same instance on repeated access."""
        # Act:
        first = sample_hsi.plot
        second = sample_hsi.plot

        # Assert:
        assert first is second

    def test_plot_pseudo_rgb_returns_axes(self, sample_hsi):
        """Test plot.pseudo_rgb returns matplotlib Axes."""
        # Act:
        ax = sample_hsi.plot.pseudo_rgb()

        # Assert:
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_pseudo_rgb_calls_imshow_with_normalized_rgb(
        self, sample_hsi
    ):
        """Test imshow is called with (H, W, 3) array in [0, 1]."""
        # Arrange:
        with patch.object(matplotlib.axes.Axes, "imshow") as mock_imshow:
            # Act:
            sample_hsi.plot.pseudo_rgb()

            # Assert:
            assert mock_imshow.called
            args, _ = mock_imshow.call_args
            rgb = args[0]
            assert rgb.shape == (sample_hsi.height, sample_hsi.width, 3)
            assert rgb.min() >= 0.0
            assert rgb.max() <= 1.0

    def test_plot_pseudo_rgb_uses_provided_axes(self, sample_hsi):
        """Test that an explicit ax parameter is used."""
        # Arrange:
        fig, ax = plt.subplots()

        # Act:
        returned = sample_hsi.plot.pseudo_rgb(ax=ax)

        # Assert:
        assert returned is ax
        plt.close(fig)

    def test_plot_pseudo_rgb_custom_wavelengths(self, small_hsi):
        """Test custom wavelengths flow through to band selection."""
        # Arrange:
        with patch.object(matplotlib.axes.Axes, "imshow") as mock_imshow:
            # Act:
            small_hsi.plot.pseudo_rgb(r_nm=850, g_nm=750, b_nm=650)

            # Assert: shape sanity
            args, _ = mock_imshow.call_args
            rgb = args[0]
            assert rgb.shape == (small_hsi.height, small_hsi.width, 3)

    def test_plot_pseudo_rgb_custom_title(self, sample_hsi):
        """Test explicit title is used instead of default."""
        # Arrange:
        fig, ax = plt.subplots()

        # Act:
        sample_hsi.plot.pseudo_rgb(ax=ax, title="Custom Title")

        # Assert:
        assert ax.get_title() == "Custom Title"
        plt.close(fig)
