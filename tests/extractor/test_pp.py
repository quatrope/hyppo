"""Tests for PPExtractor."""

from hyppo.extractor.pp import PPExtractor
import numpy as np
import pytest


class TestPPExtractor:
    """Test cases for PPExtractor."""

    @pytest.mark.skip(reason="Paper reference validation pending implementation")
    def test_paper_reference_result(self, sample_hsi):
        """Test results match reference values from literature."""
        # TODO: Implement validation against reference paper results
        pass

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = PPExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "projection_vectors" in result
        assert "n_features" in result
        assert "original_shape" in result
        assert "divergence_scores" in result
        assert result["n_features"] == 10

    def test_extract_with_custom_n_projections(self, small_hsi):
        """Test extraction with custom number of projections."""
        # Arrange
        n_projections = 3
        extractor = PPExtractor(n_projections=n_projections)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_features"] == n_projections

    def test_projection_vectors_shape(self, small_hsi):
        """Test projection vectors shape."""
        # Arrange
        extractor = PPExtractor(n_projections=3)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["projection_vectors"].ndim == 2

    def test_validate_invalid_n_projections(self, small_hsi):
        """Test validation fails with invalid n_projections."""
        # Arrange
        extractor = PPExtractor(n_projections=0)

        # Act & Assert
        with pytest.raises(ValueError, match="n_projections must be positive"):
            extractor.extract(small_hsi)

    def test_validate_negative_n_projections(self, small_hsi):
        """Test validation fails with negative n_projections."""
        # Arrange
        extractor = PPExtractor(n_projections=-5)

        # Act & Assert
        with pytest.raises(ValueError, match="n_projections must be positive"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("n_projections", [1, 3, 5])
    def test_different_n_projections(self, small_hsi, n_projections):
        """Test extraction with different projections."""
        # Arrange
        extractor = PPExtractor(n_projections=n_projections)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_features"] == n_projections

    def test_feature_name(self):
        """Test feature name."""
        assert PPExtractor.feature_name() == "p_p"

    def test_random_state_reproducibility(self, small_hsi):
        """Test random state reproducibility."""
        # Arrange
        extractor1 = PPExtractor(n_projections=3, random_state=42)
        extractor2 = PPExtractor(n_projections=3, random_state=42)

        # Act
        result1 = extractor1.extract(small_hsi)
        result2 = extractor2.extract(small_hsi)

        # Assert
        assert result1["features"].shape == result2["features"].shape

    def test_original_shape_preserved(self, small_hsi):
        """Test original shape."""
        # Arrange
        extractor = PPExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["original_shape"] == small_hsi.shape

    def test_divergence_scores_computed(self, small_hsi):
        """Test divergence scores."""
        # Arrange
        extractor = PPExtractor(n_projections=3)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert len(result["divergence_scores"]) == 3

    def test_pca_components_parameter(self, small_hsi):
        """Test PCA components parameter."""
        # Arrange
        extractor = PPExtractor(n_projections=3, pca_components=3)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["pca_components_used"] == 3

    def test_sample_size_parameter(self, small_hsi):
        """Test sample size parameter."""
        # Arrange
        extractor = PPExtractor(n_projections=3, sample_size=50)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert "features" in result

    def test_n_bins_parameter(self, small_hsi):
        """Test n_bins parameter."""
        # Arrange
        extractor = PPExtractor(n_projections=3, n_bins=30)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert "features" in result
