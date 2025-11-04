"""Tests for DWT1DExtractor."""

import pytest

from hyppo.extractor.dwt1d import DWT1DExtractor


class TestDWT1DExtractor:
    """Test cases for DWT1DExtractor."""

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result_1(self, sample_hsi):
        """Test results match reference values from literature."""
        # TODO: Implement validation against reference paper results
        # Bruce, K., Koger, C., & Li, J. (2002).
        # Dimensionality reduction of hyperspectral data using DWT.
        pass

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result_2(self, sample_hsi):
        """Test wavelet decomposition follows established theory."""
        # TODO: Implement validation against theoretical results
        # Mallat, S. (1999). A Wavelet Tour of Signal Processing.
        pass

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = DWT1DExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "wavelet" in result
        assert "mode" in result
        assert "levels" in result
        assert "coeffs_lengths" in result
        assert "n_features" in result
        assert "original_shape" in result

        # Assert: Verify default parameter values
        assert result["wavelet"] == "db4"
        assert result["mode"] == "symmetric"
        assert result["levels"] == 3

        # Assert: Verify feature shape matches (H, W, n_features)
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

        # Assert: Verify coeffs_lengths is list with correct structure
        coeffs_lengths = result["coeffs_lengths"]
        assert isinstance(coeffs_lengths, list)
        assert len(coeffs_lengths) == 4  # levels + 1 (approximation + details)

        # Assert: Verify n_features matches actual feature dimension
        assert result["n_features"] == features.shape[1]

        # Assert: Verify original shape preserved
        assert result["original_shape"] == small_hsi.shape

    def test_validate_invalid_wavelet(self, small_hsi):
        """Test validation fails with invalid wavelet name."""
        # Arrange: Create extractor with invalid wavelet
        extractor = DWT1DExtractor(wavelet="invalid_wavelet_name")

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="Wavelet .* not available"):
            extractor.extract(small_hsi)

    def test_validate_invalid_mode(self, small_hsi):
        """Test validation fails with invalid signal extension mode."""
        # Arrange: Create extractor with invalid mode
        extractor = DWT1DExtractor(mode="invalid_mode")

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="Mode .* not available"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("levels", [-1, 0, 2.5])
    def test_validate_invalid_levels(self, small_hsi, levels):
        """Test validation fails with invalid decomposition levels."""
        # Arrange: Create extractor with invalid levels
        extractor = DWT1DExtractor(levels=levels)  # type: ignore

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
        extractor = DWT1DExtractor(wavelet=wavelet, mode=mode)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction with correct parameters
        assert result["wavelet"] == wavelet
        assert result["mode"] == mode
        assert "features" in result

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = DWT1DExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "dwt1d"
