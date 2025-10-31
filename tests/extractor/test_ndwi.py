"""Tests for NDWIExtractor."""

import warnings

import numpy as np
import pytest

from hyppo.extractor.ndwi import NDWIExtractor


class TestNDWIExtractor:
    """Test cases for NDWIExtractor."""

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result(self, sample_hsi):
        """Test results match reference values from literature."""
        # TODO: Implement validation against reference paper results
        # Gao (1996) - NDWI for remote sensing of vegetation liquid water
        # McFeeters (1996) - The use of NDWI for the delineation of open water features
        pass

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = NDWIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "green_idx" in result
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
        green_wavelength = 560
        nir_wavelength = 850
        extractor = NDWIExtractor(
            green_wavelength=green_wavelength, nir_wavelength=nir_wavelength
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["wavelength_used"][0] >= 0
        assert result["wavelength_used"][1] >= 0

    def test_ndwi_calculation(self, small_hsi):
        """Test that NDWI is calculated correctly."""
        # Arrange: Create extractor
        extractor = NDWIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: NDWI values should be in range [-1, 1]
        features = result["features"]
        assert np.all(features >= -1.1)  # Allow small numerical errors
        assert np.all(features <= 1.1)

    def test_closest_band_selection(self, small_hsi):
        """Test that closest bands to target wavelengths are selected."""
        # Arrange: Create extractor
        extractor = NDWIExtractor(green_wavelength=560, nir_wavelength=850)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify band indices are valid
        assert 0 <= result["green_idx"] < small_hsi.reflectance.shape[2]
        assert 0 <= result["nir_idx"] < small_hsi.reflectance.shape[2]

    def test_wavelength_warning_green_band(self, small_hsi):
        """Test warning when green band is far from target."""
        # Arrange: Create extractor with wavelength far from available
        extractor = NDWIExtractor(green_wavelength=300, nir_wavelength=850)

        # Act & Assert: Verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extractor.extract(small_hsi)
            assert len(w) >= 1
            assert "far from target" in str(w[0].message).lower()

    def test_wavelength_warning_nir_band(self, small_hsi):
        """Test warning when NIR band is far from target."""
        # Arrange: Create extractor with wavelength far from available
        extractor = NDWIExtractor(green_wavelength=560, nir_wavelength=1500)

        # Act & Assert: Verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extractor.extract(small_hsi)
            assert len(w) >= 1
            assert "far from target" in str(w[0].message).lower()

    def test_wavelength_warning_both_bands(self, small_hsi):
        """Test warning when both bands are far from target."""
        # Arrange: Create extractor with both wavelengths far from available
        extractor = NDWIExtractor(green_wavelength=1500, nir_wavelength=2000)

        # Act & Assert: Verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extractor.extract(small_hsi)
            assert len(w) >= 1

    def test_validate_negative_green_wavelength(self, small_hsi):
        """Test validation fails with negative green wavelength."""
        # Arrange: Create extractor with negative wavelength
        extractor = NDWIExtractor(green_wavelength=-100)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="green_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_zero_green_wavelength(self, small_hsi):
        """Test validation fails with zero green wavelength."""
        # Arrange: Create extractor with zero wavelength
        extractor = NDWIExtractor(green_wavelength=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="green_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_negative_nir_wavelength(self, small_hsi):
        """Test validation fails with negative NIR wavelength."""
        # Arrange: Create extractor with negative wavelength
        extractor = NDWIExtractor(nir_wavelength=-100)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="nir_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_zero_nir_wavelength(self, small_hsi):
        """Test validation fails with zero NIR wavelength."""
        # Arrange: Create extractor with zero wavelength
        extractor = NDWIExtractor(nir_wavelength=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="nir_wavelength must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_green_greater_than_nir(self, small_hsi):
        """Test warning when green wavelength is greater than NIR."""
        # Arrange: Create extractor with inverted wavelengths
        extractor = NDWIExtractor(green_wavelength=900, nir_wavelength=500)

        # Act & Assert: Verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extractor.extract(small_hsi)
            assert len(w) >= 1
            assert "should be less than" in str(w[0].message).lower()

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = NDWIExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "n_d_w_i"

    def test_original_shape_preserved(self, small_hsi):
        """Test that original shape is recorded."""
        # Arrange: Create extractor
        extractor = NDWIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify original shape
        assert result["original_shape"] == small_hsi.shape

    def test_wavelength_used_recorded(self, small_hsi):
        """Test that actual wavelengths used are recorded."""
        # Arrange: Create extractor
        extractor = NDWIExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify wavelengths are valid
        green_wl, nir_wl = result["wavelength_used"]
        assert green_wl > 0
        assert nir_wl > 0
        assert isinstance(green_wl, (int, float, np.number))
        assert isinstance(nir_wl, (int, float, np.number))

    @pytest.mark.parametrize(
        "green_wl,nir_wl",
        [
            (540, 800),
            (560, 850),
            (580, 900),
        ],
    )
    def test_different_wavelength_combinations(
        self, small_hsi, green_wl, nir_wl
    ):
        """Test extraction with different wavelength combinations."""
        # Arrange: Create extractor with specific wavelengths
        extractor = NDWIExtractor(
            green_wavelength=green_wl, nir_wavelength=nir_wl
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["features"].shape == (small_hsi.height, small_hsi.width)

    def test_mcfeeters_variant(self, small_hsi):
        """Test McFeeters NDWI variant (green-NIR)."""
        # Arrange: Create extractor with typical McFeeters wavelengths
        extractor = NDWIExtractor(green_wavelength=560, nir_wavelength=850)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify calculation
        features = result["features"]
        assert features.shape == (small_hsi.height, small_hsi.width)
        assert np.all(np.isfinite(features) | np.isnan(features))

    def test_gao_variant_behavior(self, small_hsi):
        """Test that Gao NDWI variant can be approximated with NIR and SWIR."""
        # Arrange: Use NIR wavelength for both (since we don't have SWIR in small_hsi)
        extractor = NDWIExtractor(green_wavelength=850, nir_wavelength=800)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        features = result["features"]
        assert features.shape == (small_hsi.height, small_hsi.width)
