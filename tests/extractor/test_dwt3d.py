"""Tests for DWT3DExtractor."""

import pytest

from hyppo.extractor.dwt3d import DWT3DExtractor


class TestDWT3DExtractor:
    """Test cases for DWT3DExtractor."""

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result_1(self, sample_hsi):
        """Test results match reference values from literature."""
        # TODO: Implement validation against reference paper results
        # Qian, Ye, and Zhou (2012): Decomposed hyperspectral images
        # at different scales, orientations using 3-D wavelets.
        pass

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result_2(self, sample_hsi):
        """Test wavelet decomposition follows established theory."""
        # TODO: Implement validation against theoretical results
        # Ye et al. (2014): Extracted 3-D DWT coefficients to acquire
        # spectral–spatial information for classification.
        pass

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = DWT3DExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "wavelet" in result
        assert "levels" in result
        assert "n_features" in result
        assert "original_shape" in result

        # Assert: Verify default parameter values
        assert result["wavelet"] == "haar"
        assert result["levels"] == 1

        # Assert: Verify feature shape matches (H, W, n_features)
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

        # Assert: Verify n_features matches actual feature dimension
        assert result["n_features"] == features.shape[2]

        # Assert: Verify original shape preserved
        assert result["original_shape"] == small_hsi.shape

    def test_validate_invalid_wavelet(self, small_hsi):
        """Test validation fails with invalid wavelet name."""
        # Arrange: Create extractor with invalid wavelet
        extractor = DWT3DExtractor(wavelet="invalid_wavelet_name")

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="Wavelet .* not available"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("levels", [-1, 0, 2.5])
    def test_validate_invalid_levels(self, small_hsi, levels):
        """Test validation fails with invalid decomposition levels."""
        # Arrange: Create extractor with invalid levels
        extractor = DWT3DExtractor(levels=levels)  # type: ignore

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="levels must be a positive integer"
        ):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("wavelet", ["haar", "db4", "sym5", "coif2"])
    def test_extract_different_wavelets(self, small_hsi, wavelet):
        """Test extraction with different wavelets."""
        # Arrange: Create extractor with specific wavelet
        extractor = DWT3DExtractor(wavelet=wavelet)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction with correct parameters
        assert result["wavelet"] == wavelet
        assert "features" in result

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = DWT3DExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "dwt3d"

    def test_spatial_resolution_preserved(self, small_hsi):
        """Test that output features maintain original spatial resolution."""
        # Arrange: Create extractor
        extractor = DWT3DExtractor(levels=1)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify spatial dimensions preserved
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width

    @pytest.mark.parametrize("levels", [1, 2])
    def test_extract_different_levels(self, small_hsi, levels):
        """Test extraction with different decomposition levels."""
        # Arrange: Create extractor with specific levels
        extractor = DWT3DExtractor(levels=levels)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct levels in result
        assert result["levels"] == levels
        assert "features" in result
        assert result["features"].shape[0] == small_hsi.height
        assert result["features"].shape[1] == small_hsi.width
