"""Tests for DWT2DExtractor."""

import pytest

from hyppo.extractor.dwt2d import DWT2DExtractor


class TestDWT2DExtractor:
    """Test cases for DWT2DExtractor."""

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result_1(self, sample_hsi):
        """Test results match reference values from literature."""
        # TODO: Implement validation against reference paper results
        # Gormus, A., Canagarajah, C. N., & Achim, A. (2012). Hyperspectral image
        # classification using 2-D wavelet decomposition and spatial-spectral
        # information fusion.
        pass

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result_2(self, sample_hsi):
        """Test wavelet decomposition follows established theory."""
        # TODO: Implement validation against theoretical results
        # Quesada-Barriuso, J., Arguello, H., & Heras, P. (2014). Feature extraction
        # from hyperspectral images using 2-D discrete wavelet transform.
        pass

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result_3(self, sample_hsi):
        """Test texture feature extraction methodology."""
        # TODO: Implement validation against theoretical results
        # Kumar, P., & Dikshit, O. (2015a). Texture feature extraction for
        # hyperspectral image classification using 2-D DWT.
        pass

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = DWT2DExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "wavelet" in result
        assert "mode" in result
        assert "levels" in result
        assert "n_features" in result
        assert "original_shape" in result

        # Assert: Verify default parameter values
        assert result["wavelet"] == "db4"
        assert result["mode"] == "symmetric"
        assert result["levels"] == 2

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
        extractor = DWT2DExtractor(wavelet="invalid_wavelet_name")

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="Wavelet .* not available"):
            extractor.extract(small_hsi)

    def test_validate_invalid_mode(self, small_hsi):
        """Test validation fails with invalid signal extension mode."""
        # Arrange: Create extractor with invalid mode
        extractor = DWT2DExtractor(mode="invalid_mode")

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="Mode .* not available"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("levels", [-1, 0, 2.5])
    def test_validate_invalid_levels(self, small_hsi, levels):
        """Test validation fails with invalid decomposition levels."""
        # Arrange: Create extractor with invalid levels
        extractor = DWT2DExtractor(levels=levels)  # type: ignore

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="levels must be a positive integer"
        ):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize(
        "wavelet,mode",
        [
            ("haar", "symmetric"),
            ("haar", "periodic"),
            ("db4", "symmetric"),
            ("db4", "zero"),
            ("sym5", "constant"),
            ("coif2", "periodic"),
        ],
    )
    def test_extract_wavelet_mode_combinations(self, small_hsi, wavelet, mode):
        """Test extraction with cross-product of wavelets and modes."""
        # Arrange: Create extractor with specific wavelet and mode
        extractor = DWT2DExtractor(wavelet=wavelet, mode=mode)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction with correct parameters
        assert result["wavelet"] == wavelet
        assert result["mode"] == mode
        assert "features" in result

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = DWT2DExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "dwt2d"

    def test_spatial_resolution_preserved(self, small_hsi):
        """Test that output features maintain original spatial resolution."""
        # Arrange: Create extractor
        extractor = DWT2DExtractor(levels=2)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify spatial dimensions preserved
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width

    @pytest.mark.parametrize("levels", [1, 2, 3])
    def test_extract_different_levels(self, small_hsi, levels):
        """Test extraction with different decomposition levels."""
        # Arrange: Create extractor with specific levels
        extractor = DWT2DExtractor(levels=levels)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct levels in result
        assert result["levels"] == levels
        assert "features" in result
        assert result["features"].shape[0] == small_hsi.height
        assert result["features"].shape[1] == small_hsi.width
