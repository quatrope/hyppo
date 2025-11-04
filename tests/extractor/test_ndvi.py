"""Tests for NDVIExtractor."""

import warnings

import numpy as np
import pytest

from hyppo.extractor.ndvi import NDVIExtractor


class TestNDVIExtractor:
    """Test cases for NDVIExtractor."""

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result(self, sample_hsi):
        """Test results match reference values from literature."""
        # TODO: Implement validation against reference paper results
        # Rouse et al. (1974) - Monitoring vegetation systems
        pass

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = NDVIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "red_idx" in result
        assert "nir_idx" in result
        assert "wavelength_used" in result
        assert "original_shape" in result

        # Assert: Verify feature shape (2D for index)
        features = result["features"]
        assert features.shape == (small_hsi.height, small_hsi.width)
        assert features.ndim == 2

    def test_extract_with_custom_wavelengths(self, small_hsi):
        """Test extraction with custom wavelengths."""
        # Arrange: Create extractor with custom wavelengths
        red_wavelength = 650
        nir_wavelength = 850
        extractor = NDVIExtractor(
            red_wavelength=red_wavelength, nir_wavelength=nir_wavelength
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["wavelength_used"][0] >= 0
        assert result["wavelength_used"][1] >= 0

    def test_ndvi_calculation(self, small_hsi):
        """Test that NDVI is calculated correctly."""
        # Arrange: Create extractor
        extractor = NDVIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: NDVI values should be in range [-1, 1]
        features = result["features"]
        assert np.all(features >= -1.1)  # Allow small numerical errors
        assert np.all(features <= 1.1)

    def test_closest_band_selection(self, small_hsi):
        """Test that closest bands to target wavelengths are selected."""
        # Arrange: Create extractor
        extractor = NDVIExtractor(red_wavelength=660, nir_wavelength=850)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify band indices are valid
        assert 0 <= result["red_idx"] < small_hsi.reflectance.shape[2]
        assert 0 <= result["nir_idx"] < small_hsi.reflectance.shape[2]

    def test_wavelength_warning_red_band(self, small_hsi):
        """Test warning when red band is far from target."""
        # Arrange: Create extractor with wavelength far from available
        extractor = NDVIExtractor(red_wavelength=300, nir_wavelength=850)

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
        extractor = NDVIExtractor(red_wavelength=660, nir_wavelength=1500)

        # Act & Assert: Verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extractor.extract(small_hsi)
            assert "features" in result
            assert len(w) >= 1
            assert "far from target" in str(w[0].message).lower()

    def test_wavelength_warning_both_bands(self, small_hsi):
        """Test warning when both bands are far from target."""
        # Arrange: Create extractor with both wavelengths far from available
        extractor = NDVIExtractor(red_wavelength=1500, nir_wavelength=2000)

        # Act & Assert: Verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extractor.extract(small_hsi)
            assert len(w) >= 1

    def test_validate_negative_red_wavelength(self, small_hsi):
        """Test validation fails with negative red wavelength."""
        # Arrange: Create extractor with negative wavelength
        extractor = NDVIExtractor(red_wavelength=-100)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="red_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_zero_red_wavelength(self, small_hsi):
        """Test validation fails with zero red wavelength."""
        # Arrange: Create extractor with zero wavelength
        extractor = NDVIExtractor(red_wavelength=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="red_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_negative_nir_wavelength(self, small_hsi):
        """Test validation fails with negative NIR wavelength."""
        # Arrange: Create extractor with negative wavelength
        extractor = NDVIExtractor(nir_wavelength=-100)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="nir_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_zero_nir_wavelength(self, small_hsi):
        """Test validation fails with zero NIR wavelength."""
        # Arrange: Create extractor with zero wavelength
        extractor = NDVIExtractor(nir_wavelength=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="nir_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_red_greater_than_nir(self, small_hsi):
        """Test warning when red wavelength is greater than NIR."""
        # Arrange: Create extractor with inverted wavelengths
        extractor = NDVIExtractor(red_wavelength=900, nir_wavelength=600)

        # Act & Assert: Verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extractor.extract(small_hsi)
            assert len(w) >= 1
            assert "should be less than" in str(w[0].message).lower()

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = NDVIExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "n_d_v_i"

    def test_original_shape_preserved(self, small_hsi):
        """Test that original shape is recorded."""
        # Arrange: Create extractor
        extractor = NDVIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify original shape
        assert result["original_shape"] == small_hsi.shape

    def test_wavelength_used_recorded(self, small_hsi):
        """Test that actual wavelengths used are recorded."""
        # Arrange: Create extractor
        extractor = NDVIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify wavelengths are valid
        red_wl, nir_wl = result["wavelength_used"]
        assert red_wl > 0
        assert nir_wl > 0
        assert isinstance(red_wl, (int, float, np.number))
        assert isinstance(nir_wl, (int, float, np.number))

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
        extractor = NDVIExtractor(red_wavelength=red_wl, nir_wavelength=nir_wl)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["features"].shape == (small_hsi.height, small_hsi.width)
