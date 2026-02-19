"""Tests for MPExtractor."""

import numpy as np
import pytest

from hyppo.extractor.mp import MPExtractor


class TestMPExtractor:
    """Test cases for MPExtractor."""

    @pytest.mark.skip(
        reason="Paper reference validation pending implementation"
    )
    def test_paper_reference_result(self, sample_hsi):
        """Test results match reference values from literature."""
        # TODO: Implement validation against reference paper results
        # Lv et al. (2014) - Morphological Profiles Based on Differently Shaped
        # Structuring Elements for Classification
        pass

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = MPExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "explained_variance_ratio" in result
        assert "n_components" in result
        assert "shapes" in result
        assert "radii" in result
        assert "n_features" in result
        assert "use_reconstruction" in result

        # Assert: Verify default parameter values
        assert result["n_components"] == 3
        assert result["radii"] == [2, 4, 6, 8]
        assert result["shapes"] == ["disk", "square", "diamond"]
        assert result["use_reconstruction"] is False

        # Assert: Verify feature shape
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom parameters."""
        # Arrange: Create extractor with custom parameters
        extractor = MPExtractor(
            n_components=2,
            radii=[1, 3],
            shapes=["disk"],
            use_reconstruction=True,
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify custom parameters
        assert result["n_components"] == 2
        assert result["radii"] == [1, 3]
        assert result["shapes"] == ["disk"]
        assert result["use_reconstruction"] is True

    @pytest.mark.parametrize("shape", ["disk", "square", "diamond"])
    def test_different_shapes(self, small_hsi, shape):
        """Test extraction with different structuring element shapes."""
        # Arrange: Create extractor with specific shape
        extractor = MPExtractor(shapes=[shape])

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct shape used
        assert result["shapes"] == [shape]
        assert "features" in result

    def test_get_structuring_element_disk(self):
        """Test disk structuring element generation."""
        # Arrange: Create extractor
        extractor = MPExtractor()

        # Act: Get structuring element
        elem = extractor._get_structuring_element("disk", 3)

        # Assert: Verify disk structure
        assert elem.ndim == 2
        assert elem.shape[0] == elem.shape[1]

    def test_get_structuring_element_square(self):
        """Test square structuring element generation."""
        # Arrange: Create extractor
        extractor = MPExtractor()

        # Act: Get structuring element
        elem = extractor._get_structuring_element("square", 3)

        # Assert: Verify square structure
        assert elem.ndim == 2
        assert elem.shape[0] == elem.shape[1]
        assert elem.shape[0] == 7  # 2*3 + 1

    def test_get_structuring_element_diamond(self):
        """Test diamond structuring element generation."""
        # Arrange: Create extractor
        extractor = MPExtractor()

        # Act: Get structuring element
        elem = extractor._get_structuring_element("diamond", 3)

        # Assert: Verify diamond structure
        assert elem.ndim == 2

    def test_get_structuring_element_invalid(self):
        """Test invalid structuring element raises error."""
        # Arrange: Create extractor
        extractor = MPExtractor()

        # Act & Assert: Verify error raised
        with pytest.raises(ValueError, match="Unsupported shape"):
            extractor._get_structuring_element("invalid", 3)

    @pytest.mark.parametrize("radii", [[1], [1, 2], [2, 4, 6, 8]])
    def test_different_radii(self, small_hsi, radii):
        """Test extraction with different radii."""
        # Arrange: Create extractor with specific radii
        extractor = MPExtractor(radii=radii)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct radii used
        assert result["radii"] == sorted(radii)

    def test_n_features_calculation(self, small_hsi):
        """Test that n_features is calculated correctly."""
        # Arrange: Create extractor with known configuration
        n_components = 2
        radii = [1, 2]
        shapes = ["disk"]
        extractor = MPExtractor(
            n_components=n_components, radii=radii, shapes=shapes
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: n_features = n_components * len(shapes) * (2*len(radii) + 1)
        expected = n_components * len(shapes) * (2 * len(radii) + 1)
        assert result["n_features"] == expected

    def test_validate_invalid_radii_type(self, small_hsi):
        """Test validation fails with invalid radii type."""
        # Arrange: Create extractor with invalid radii
        extractor = MPExtractor(radii="invalid")

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises((ValueError, TypeError)):
            extractor.extract(small_hsi)

    def test_validate_empty_radii(self, small_hsi):
        """Test validation fails with empty radii list."""
        # Arrange: Create extractor with empty radii
        extractor = MPExtractor(radii=[])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="radii must be a non-empty"):
            extractor.extract(small_hsi)

    def test_validate_negative_radii(self, small_hsi):
        """Test validation fails with negative radii."""
        # Arrange: Create extractor with negative radii
        extractor = MPExtractor(radii=[-1, 2])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="All radii must be positive"):
            extractor.extract(small_hsi)

    def test_validate_zero_radii(self, small_hsi):
        """Test validation fails with zero radii."""
        # Arrange: Create extractor with zero radii
        extractor = MPExtractor(radii=[0, 1])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="All radii must be positive"):
            extractor.extract(small_hsi)

    def test_validate_invalid_shape(self, small_hsi):
        """Test validation fails with invalid shape."""
        # Arrange: Create extractor with invalid shape
        extractor = MPExtractor(shapes=["invalid"])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="Invalid shape"):
            extractor.extract(small_hsi)

    def test_validate_empty_shapes(self, small_hsi):
        """Test validation fails with empty shapes list."""
        # Arrange: Create extractor with empty shapes
        extractor = MPExtractor(shapes=[])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="shapes must be a non-empty"):
            extractor.extract(small_hsi)

    def test_validate_invalid_n_components(self, small_hsi):
        """Test validation fails with invalid n_components."""
        # Arrange: Create extractor with invalid n_components
        extractor = MPExtractor(n_components=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_n_components_exceeds_bands(self, small_hsi):
        """Test validation fails when n_components exceeds spectral bands."""
        # Arrange: Create extractor with too many components
        extractor = MPExtractor(n_components=100)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="Number of spectral bands"):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = MPExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "m_p"

    def test_spatial_shape_preserved(self, small_hsi):
        """Test that output features preserve spatial dimensions."""
        # Arrange: Create extractor
        extractor = MPExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify spatial dimensions
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width

    def test_use_reconstruction(self, small_hsi):
        """Test extraction with reconstruction-based morphological operations."""
        # Arrange: Create extractors with and without reconstruction
        extractor_standard = MPExtractor(
            shapes=["disk"], radii=[2], use_reconstruction=False
        )
        extractor_recon = MPExtractor(
            shapes=["disk"], radii=[2], use_reconstruction=True
        )

        # Act: Execute both extractions
        result_standard = extractor_standard.extract(small_hsi)
        result_recon = extractor_recon.extract(small_hsi)

        # Assert: Both produce valid results with same shape
        assert result_standard["features"].shape == result_recon["features"].shape
        assert result_standard["use_reconstruction"] is False
        assert result_recon["use_reconstruction"] is True

    def test_pca_variance_explained(self, small_hsi):
        """Test that PCA variance ratios are returned."""
        # Arrange: Create extractor
        extractor = MPExtractor(n_components=2)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify variance explained
        variance_ratio = result["explained_variance_ratio"]
        assert len(variance_ratio) == 2
        assert np.all(variance_ratio >= 0)
        assert np.all(variance_ratio <= 1)
