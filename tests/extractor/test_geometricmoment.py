"""Tests for GeometricMomentExtractor."""

import numpy as np
import pytest

from hyppo.extractor.geometricmoment import GeometricMomentExtractor


class TestGeometricMomentExtractor:
    """Test cases for GeometricMomentExtractor."""

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = GeometricMomentExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "explained_variance_ratio" in result
        assert "n_components" in result
        assert "window_sizes" in result
        assert "max_order" in result

        # Assert: Verify default parameter values
        assert result["n_components"] == 3
        assert result["window_sizes"] == [3, 9, 15]
        assert result["max_order"] == 3

        # Assert: Verify feature shape
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom parameters."""
        # Arrange: Create extractor with custom parameters
        n_components = 2
        max_order = 2
        window_sizes = [3, 5]
        extractor = GeometricMomentExtractor(
            n_components=n_components,
            max_order=max_order,
            window_sizes=window_sizes,
        )

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify custom parameters
        assert result["n_components"] == n_components
        assert result["max_order"] == max_order
        assert result["window_sizes"] == window_sizes

    def test_geometric_moments_computation(self, small_hsi):
        """Test geometric moments computation."""
        # Arrange: Create extractor and test patches
        extractor = GeometricMomentExtractor(max_order=2)
        patches = np.random.rand(10, 5, 5).astype(np.float32)

        # Act: Compute moments
        moments = extractor._geometric_moments(patches)

        # Assert: Verify moments shape
        expected_n_moments = sum(1 for p in range(3) for q in range(3 - p))
        assert moments.shape[0] == 10
        assert moments.shape[1] == expected_n_moments

    def test_extract_moments_multiscale(self, small_hsi):
        """Test multiscale moment extraction."""
        # Arrange: Create extractor
        extractor = GeometricMomentExtractor(window_sizes=[3, 5])
        image = small_hsi.reflectance[:, :, 0]

        # Act: Extract multiscale moments
        moments = extractor._extract_moments_multiscale(image)

        # Assert: Verify output shape
        assert moments.shape[0] == image.shape[0]
        assert moments.shape[1] == image.shape[1]
        assert moments.ndim == 3

    def test_validate_invalid_n_components(self, small_hsi):
        """Test validation fails with invalid n_components."""
        # Arrange: Create extractor with invalid n_components
        extractor = GeometricMomentExtractor(n_components=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_max_order(self, small_hsi):
        """Test validation fails with invalid max_order."""
        # Arrange: Create extractor with negative max_order
        extractor = GeometricMomentExtractor(max_order=-1)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="max_order must be a non-negative integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_window_sizes_empty(self, small_hsi):
        """Test validation fails with empty window_sizes."""
        # Arrange: Create extractor with empty window_sizes
        extractor = GeometricMomentExtractor(window_sizes=[])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="window_sizes must be a non-empty list"
        ):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("invalid_window", [2, 4, 1])
    def test_validate_invalid_window_size_values(
        self, small_hsi, invalid_window
    ):
        """Test validation fails with invalid window size values."""
        # Arrange: Create extractor with invalid window size
        extractor = GeometricMomentExtractor(window_sizes=[invalid_window])

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="Each window size must be an odd integer"
        ):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("n_components", [1, 3, 5])
    def test_different_n_components(self, small_hsi, n_components):
        """Test extraction with different number of components."""
        # Arrange: Create extractor with specific n_components
        extractor = GeometricMomentExtractor(n_components=n_components)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct number of components
        assert result["n_components"] == n_components
        assert len(result["explained_variance_ratio"]) == n_components

    @pytest.mark.parametrize("max_order", [1, 2, 3])
    def test_different_max_orders(self, small_hsi, max_order):
        """Test extraction with different max orders."""
        # Arrange: Create extractor with specific max_order
        extractor = GeometricMomentExtractor(max_order=max_order)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct max order
        assert result["max_order"] == max_order

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = GeometricMomentExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "geometric_moment"

    def test_pca_variance_explained(self, small_hsi):
        """Test that PCA variance is computed."""
        # Arrange: Create extractor
        extractor = GeometricMomentExtractor(n_components=2)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify variance explained
        assert "explained_variance_ratio" in result
        variance_ratio = result["explained_variance_ratio"]
        assert len(variance_ratio) == 2
        assert np.all(variance_ratio >= 0)
        assert np.all(variance_ratio <= 1)

    def test_multiscale_concatenation(self, small_hsi):
        """Test that multiscale features are concatenated properly."""
        # Arrange: Create extractor with multiple window sizes
        window_sizes = [3, 5, 7]
        extractor = GeometricMomentExtractor(window_sizes=window_sizes)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify features from all scales are present
        features = result["features"]
        assert features.shape[2] > 0

    def test_validate_non_integer_n_components(self, small_hsi):
        """Test validation fails with non-integer n_components."""
        # Arrange: Create extractor with float n_components
        extractor = GeometricMomentExtractor(n_components=2.5)  # type: ignore

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_non_integer_max_order(self, small_hsi):
        """Test validation fails with non-integer max_order."""
        # Arrange: Create extractor with float max_order
        extractor = GeometricMomentExtractor(max_order=2.5)  # type: ignore

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(
            ValueError, match="max_order must be a non-negative integer"
        ):
            extractor.extract(small_hsi)
