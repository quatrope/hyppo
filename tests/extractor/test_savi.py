"""Tests for SAVIExtractor."""

import warnings

import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.extractor.savi import SAVIExtractor


class TestSAVIExtractor:
    """Test cases for SAVIExtractor."""

    @pytest.mark.parametrize(
        "red_val,nir_val,L",
        [
            (0.08, 0.50, 0.5),  # Dense vegetation, standard L
            (0.25, 0.30, 0.5),  # Sparse vegetation
            (0.10, 0.40, 0.0),  # L=0 reduces to NDVI-like
            (0.10, 0.40, 1.0),  # L=1 for very sparse vegetation
            (0.20, 0.20, 0.5),  # Equal reflectance
        ],
    )
    def test_savi_formula_reference(self, red_val, nir_val, L):
        """Test SAVI calculation matches the mathematical definition.

        SAVI = ((NIR - Red) * (1 + L)) / (NIR + Red + L)

        Reference: Huete (1988) - A soil-adjusted vegetation index (SAVI).
        Remote Sensing of Environment, 25(3), 295-309.
        """
        # Arrange: Create HSI with known reflectance values
        wavelengths = np.array([660.0, 850.0])
        reflectance = np.full((2, 2, 2), 0.0, dtype=np.float32)
        reflectance[:, :, 0] = red_val
        reflectance[:, :, 1] = nir_val
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)

        expected_savi = ((nir_val - red_val) * (1 + L)) / (
            nir_val + red_val + L
        )
        extractor = SAVIExtractor(red_wavelength=660, nir_wavelength=850, L=L)

        # Act
        result = extractor.extract(hsi)

        # Assert
        np.testing.assert_allclose(
            result["features"],
            expected_savi,
            rtol=1e-5,
        )

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = SAVIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "red_idx" in result
        assert "nir_idx" in result
        assert "brightness_correction" in result
        assert "wavelength_used" in result
        assert "original_shape" in result

        # Assert: Verify default parameter values
        assert result["brightness_correction"] == 0.5

        # Assert: Verify feature shape (3D with single band for index)
        features = result["features"]
        assert features.shape == (small_hsi.height, small_hsi.width, 1)
        assert features.ndim == 3

    def test_extract_with_custom_wavelengths(self, small_hsi):
        """Test extraction with custom wavelengths."""
        # Arrange: Create extractor with custom wavelengths
        red_wavelength = 650
        nir_wavelength = 850
        extractor = SAVIExtractor(
            red_wavelength=red_wavelength, nir_wavelength=nir_wavelength
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["wavelength_used"][0] >= 0
        assert result["wavelength_used"][1] >= 0

    def test_extract_with_custom_L(self, small_hsi):
        """Test extraction with custom L parameter."""
        # Arrange: Create extractor with custom L
        L = 0.25
        extractor = SAVIExtractor(L=L)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify custom L parameter
        assert result["brightness_correction"] == L

    def test_savi_calculation(self, small_hsi):
        """Test that SAVI is calculated correctly."""
        # Arrange: Create extractor
        extractor = SAVIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: SAVI values should be in reasonable range
        features = result["features"]
        assert np.all(features >= -2)  # Allow for edge cases
        assert np.all(features <= 2)

    def test_closest_band_selection(self, small_hsi):
        """Test that closest bands to target wavelengths are selected."""
        # Arrange: Create extractor
        extractor = SAVIExtractor(red_wavelength=660, nir_wavelength=850)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify band indices are valid
        assert 0 <= result["red_idx"] < small_hsi.reflectance.shape[2]
        assert 0 <= result["nir_idx"] < small_hsi.reflectance.shape[2]

    def test_wavelength_warning_red_band(self, small_hsi):
        """Test warning when red band is far from target."""
        # Arrange: Create extractor with wavelength far from available
        extractor = SAVIExtractor(red_wavelength=300, nir_wavelength=850)

        # Act & Assert: Verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extractor.extract(small_hsi)
            assert "features" in result
            assert len(w) >= 1
            assert "far from target" in str(w[0].message).lower()

    def test_wavelength_warning_nir_band(self, small_hsi):
        """Test warning when NIR band is far from target."""
        # Arrange: Create extractor with wavelength far from available
        extractor = SAVIExtractor(red_wavelength=660, nir_wavelength=1500)

        # Act & Assert: Verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extractor.extract(small_hsi)
            assert "features" in result
            assert len(w) >= 1
            assert "far from target" in str(w[0].message).lower()

    def test_validate_negative_red_wavelength(self, small_hsi):
        """Test validation fails with negative red wavelength."""
        # Arrange: Create extractor with negative wavelength
        extractor = SAVIExtractor(red_wavelength=-100)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="red_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_zero_red_wavelength(self, small_hsi):
        """Test validation fails with zero red wavelength."""
        # Arrange: Create extractor with zero wavelength
        extractor = SAVIExtractor(red_wavelength=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="red_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_negative_nir_wavelength(self, small_hsi):
        """Test validation fails with negative NIR wavelength."""
        # Arrange: Create extractor with negative wavelength
        extractor = SAVIExtractor(nir_wavelength=-100)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="nir_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_zero_nir_wavelength(self, small_hsi):
        """Test validation fails with zero NIR wavelength."""
        # Arrange: Create extractor with zero wavelength
        extractor = SAVIExtractor(nir_wavelength=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="nir_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_red_greater_than_nir(self, small_hsi):
        """Test warning when red wavelength is greater than NIR."""
        # Arrange: Create extractor with inverted wavelengths
        extractor = SAVIExtractor(red_wavelength=900, nir_wavelength=600)

        # Act & Assert: Verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extractor.extract(small_hsi)
            assert len(w) >= 1
            assert "should be less than" in str(w[0].message).lower()

    def test_validate_negative_L(self, small_hsi):
        """Test validation fails with negative L."""
        # Arrange: Create extractor with negative L
        extractor = SAVIExtractor(L=-0.5)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="L must be between 0 and 1"):
            extractor.extract(small_hsi)

    def test_validate_L_greater_than_one(self, small_hsi):
        """Test validation fails with L > 1."""
        # Arrange: Create extractor with L > 1
        extractor = SAVIExtractor(L=1.5)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="L must be between 0 and 1"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("L", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_different_L_values(self, small_hsi, L):
        """Test extraction with different L values."""
        # Arrange: Create extractor with specific L
        extractor = SAVIExtractor(L=L)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct L used
        assert result["brightness_correction"] == L

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = SAVIExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "savi"

    def test_original_shape_preserved(self, small_hsi):
        """Test that original shape is recorded."""
        # Arrange: Create extractor
        extractor = SAVIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify original shape
        assert result["original_shape"] == small_hsi.shape

    def test_wavelength_used_recorded(self, small_hsi):
        """Test that actual wavelengths used are recorded."""
        # Arrange: Create extractor
        extractor = SAVIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify wavelengths are valid
        red_wl, nir_wl = result["wavelength_used"]
        assert red_wl > 0
        assert nir_wl > 0
        assert isinstance(red_wl, (int, float, np.number))
        assert isinstance(nir_wl, (int, float, np.number))

    def test_low_vegetation_L_zero(self, small_hsi):
        """Test SAVI with L=0 (high vegetation, similar to NDVI)."""
        # Arrange: Create extractor with L=0
        extractor = SAVIExtractor(L=0.0)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["brightness_correction"] == 0.0

    def test_high_soil_background_L_one(self, small_hsi):
        """Test SAVI with L=1 (high soil background)."""
        # Arrange: Create extractor with L=1
        extractor = SAVIExtractor(L=1.0)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["brightness_correction"] == 1.0

    def test_intermediate_vegetation_L_half(self, small_hsi):
        """Test SAVI with L=0.5 (intermediate vegetation)."""
        # Arrange: Create extractor with L=0.5 (default)
        extractor = SAVIExtractor(L=0.5)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["brightness_correction"] == 0.5

    @pytest.mark.parametrize(
        "red_wl,nir_wl",
        [
            (650, 800),
            (660, 850),
            (670, 900),
        ],
    )
    def test_different_wavelength_combinations(
        self, small_hsi, red_wl, nir_wl
    ):
        """Test extraction with different wavelength combinations."""
        # Arrange: Create extractor with specific wavelengths
        extractor = SAVIExtractor(red_wavelength=red_wl, nir_wavelength=nir_wl)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["features"].shape == (
            small_hsi.height,
            small_hsi.width,
            1,
        )

    def test_validate_empty_wavelengths(self):
        """Test validation fails with empty wavelengths."""
        # Arrange
        hsi = HSI(
            reflectance=np.zeros((3, 3, 0), dtype=np.float32),
            wavelengths=np.array([], dtype=np.float32),
        )
        extractor = SAVIExtractor()

        # Act & Assert
        with pytest.raises(ValueError, match="No wavelength information"):
            extractor.extract(hsi)
