"""Tests for LBPExtractor."""

import pytest
import numpy as np
from hyppo.extractor.lbp import LBPExtractor


class TestLBPExtractor:
    """Test cases for LBPExtractor."""

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = LBPExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "bands_used" in result
        assert "radius" in result
        assert "n_points" in result
        assert "method" in result
        assert "original_shape" in result
        assert "n_features" in result

        # Assert: Verify default parameter values
        assert result["radius"] == 3
        assert result["n_points"] == 24  # 8 * radius
        assert result["method"] == "uniform"

        # Assert: Verify feature shape
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom parameters."""
        # Arrange: Create extractor with custom parameters
        bands = [0, 1]
        radius = 2
        n_points = 8
        method = "ror"
        extractor = LBPExtractor(
            bands=bands,
            radius=radius,
            n_points=n_points,
            method=method,
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify custom parameters
        assert result["bands_used"] == bands
        assert result["radius"] == radius
        assert result["n_points"] == n_points
        assert result["method"] == method

    def test_all_bands_default(self, small_hsi):
        """Test that default behavior processes all bands."""
        # Arrange: Create extractor without specifying bands
        extractor = LBPExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify all bands used
        assert len(result["bands_used"]) == small_hsi.reflectance.shape[2]

    def test_specific_bands(self, small_hsi):
        """Test extraction with specific bands."""
        # Arrange: Create extractor with specific bands
        bands = [0, 2]
        extractor = LBPExtractor(bands=bands)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct bands used
        assert result["bands_used"] == bands
        assert result["n_features"] == len(bands)

    def test_normalize_band(self, small_hsi):
        """Test band normalization."""
        # Arrange: Create extractor
        extractor = LBPExtractor()
        band = small_hsi.reflectance[:, :, 0]

        # Act: Normalize band
        normalized = extractor._normalize_band(band)

        # Assert: Verify normalization
        assert normalized.dtype == np.uint8
        assert normalized.min() >= 0
        assert normalized.max() <= 255

    def test_normalize_band_constant(self, small_hsi):
        """Test normalization of constant band."""
        # Arrange: Create extractor and constant band
        extractor = LBPExtractor()
        band = np.ones((10, 10)) * 5.0

        # Act: Normalize band
        normalized = extractor._normalize_band(band)

        # Assert: Verify all zeros for constant band
        assert np.all(normalized == 0)

    def test_compute_lbp_responses(self, small_hsi):
        """Test LBP response computation."""
        # Arrange: Create extractor
        extractor = LBPExtractor(bands=[0, 1])
        X = small_hsi.reflectance

        # Act: Compute responses
        responses, bands_used = extractor._compute_lbp_responses(X)

        # Assert: Verify responses
        assert len(responses) == 2
        assert bands_used == [0, 1]
        for band_idx in bands_used:
            assert band_idx in responses
            assert responses[band_idx].shape == (X.shape[0], X.shape[1])

    def test_compute_lbp_responses_all_bands(self, small_hsi):
        """Test LBP response computation for all bands."""
        # Arrange: Create extractor without band specification
        extractor = LBPExtractor()
        X = small_hsi.reflectance

        # Act: Compute responses
        responses, bands_used = extractor._compute_lbp_responses(X)

        # Assert: Verify all bands processed
        assert len(responses) == X.shape[2]
        assert len(bands_used) == X.shape[2]

    def test_validate_invalid_bands(self, small_hsi):
        """Test validation fails with invalid bands."""
        # Arrange: Create extractor with invalid bands
        extractor = LBPExtractor(bands="invalid") # type: ignore

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="bands must be None or a non-empty list"):
            extractor.extract(small_hsi)

    def test_validate_empty_bands(self, small_hsi):
        """Test validation fails with empty bands list."""
        # Arrange: Create extractor with empty bands
        extractor = LBPExtractor(bands=[])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="bands must be None or a non-empty list"):
            extractor.extract(small_hsi)

    def test_validate_invalid_radius(self, small_hsi):
        """Test validation fails with invalid radius."""
        # Arrange: Create extractor with invalid radius
        extractor = LBPExtractor(radius=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="radius must be a positive number"):
            extractor.extract(small_hsi)

    def test_validate_negative_radius(self, small_hsi):
        """Test validation fails with negative radius."""
        # Arrange: Create extractor with negative radius
        extractor = LBPExtractor(radius=-1)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="radius must be a positive number"):
            extractor.extract(small_hsi)

    def test_validate_invalid_n_points(self, small_hsi):
        """Test validation fails with invalid n_points."""
        # Arrange: Create extractor with invalid n_points
        extractor = LBPExtractor(n_points=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="n_points must be a positive integer"):
            extractor.extract(small_hsi)

    def test_validate_n_points_too_small(self, small_hsi):
        """Test validation fails with n_points < 3."""
        # Arrange: Create extractor with n_points < 3
        extractor = LBPExtractor(n_points=2)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="n_points must be at least 3"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("method", ["default", "ror", "uniform", "nri_uniform", "var"])
    def test_different_methods(self, small_hsi, method):
        """Test extraction with different LBP methods."""
        # Arrange: Create extractor with specific method
        extractor = LBPExtractor(method=method)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct method
        assert result["method"] == method

    def test_validate_invalid_method(self):
        """Test that invalid method raises error during initialization."""
        # Act & Assert: Verify invalid method raises ValueError
        with pytest.raises(ValueError, match="method must be one of"):
            LBPExtractor(method="invalid_method")

    @pytest.mark.parametrize("radius", [1, 2, 3, 5])
    def test_different_radius_values(self, small_hsi, radius):
        """Test extraction with different radius values."""
        # Arrange: Create extractor with specific radius
        extractor = LBPExtractor(radius=radius)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct radius
        assert result["radius"] == radius

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = LBPExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "l_b_p"

    def test_n_points_default_calculation(self, small_hsi):
        """Test that n_points defaults to 8 * radius."""
        # Arrange: Create extractor with only radius specified
        radius = 4
        extractor = LBPExtractor(radius=radius, n_points=None)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify n_points = 8 * radius
        assert result["n_points"] == 8 * radius

    def test_n_features_matches_bands(self, small_hsi):
        """Test that n_features matches number of bands processed."""
        # Arrange: Create extractor with specific bands
        bands = [0, 1, 2]
        extractor = LBPExtractor(bands=bands)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify n_features matches number of bands
        assert result["n_features"] == len(bands)

    def test_original_shape_preserved(self, small_hsi):
        """Test that original spatial shape is recorded."""
        # Arrange: Create extractor
        extractor = LBPExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify original shape
        assert result["original_shape"] == (small_hsi.height, small_hsi.width)

    def test_band_index_validation(self, small_hsi):
        """Test that out-of-range band indices raise error."""
        # Arrange: Create extractor with out-of-range band
        extractor = LBPExtractor(bands=[0, 999])

        # Act & Assert: Verify out-of-range band raises ValueError
        with pytest.raises(ValueError, match="Band index .* is out of range"):
            extractor.extract(small_hsi)

    def test_negative_band_index(self, small_hsi):
        """Test that negative band indices raise error."""
        # Arrange: Create extractor with negative band index
        extractor = LBPExtractor(bands=[-1])

        # Act & Assert: Verify negative band raises ValueError
        with pytest.raises(ValueError, match="Band index .* is out of range"):
            extractor.extract(small_hsi)
