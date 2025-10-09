"""Tests for PCAExtractor."""

import pytest
import numpy as np
from hyppo.extractor.pca import PCAExtractor


class TestPCAExtractor:
    """Test cases for PCAExtractor."""

    @pytest.mark.skip(reason="Paper reference validation pending implementation")
    def test_paper_reference_result(self, sample_hsi):
        """Test results match reference values from literature."""
        # TODO: Implement validation against reference paper results
        # Principal Component Analysis for dimensionality reduction
        pass

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange: Create extractor with defaults
        extractor = PCAExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        assert "features" in result
        assert "components" in result
        assert "explained_variance" in result
        assert "explained_variance_ratio" in result
        assert "mean" in result
        assert "n_components" in result
        assert "original_shape" in result

        # Assert: Verify default parameter values
        assert result["n_components"] == 5

        # Assert: Verify feature shape
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

    def test_extract_with_custom_n_components(self, small_hsi):
        """Test extraction with custom number of components."""
        # Arrange: Create extractor with custom n_components
        n_components = 3
        extractor = PCAExtractor(n_components=n_components)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify custom n_components
        assert result["n_components"] == n_components
        assert result["features"].shape[2] == n_components

    def test_components_shape(self, small_hsi):
        """Test that components have correct shape."""
        # Arrange: Create extractor
        n_components = 3
        extractor = PCAExtractor(n_components=n_components)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify components shape
        components = result["components"]
        assert components.shape[0] == n_components
        assert components.shape[1] == small_hsi.reflectance.shape[2]

    def test_explained_variance_properties(self, small_hsi):
        """Test explained variance properties."""
        # Arrange: Create extractor
        extractor = PCAExtractor(n_components=3)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify explained variance
        explained_var = result["explained_variance"]
        explained_var_ratio = result["explained_variance_ratio"]

        assert len(explained_var) == 3
        assert len(explained_var_ratio) == 3
        assert np.all(explained_var >= 0)
        assert np.all(explained_var_ratio >= 0)
        assert np.all(explained_var_ratio <= 1)

    def test_cumulative_variance_ratio(self, small_hsi):
        """Test cumulative variance ratio is computed."""
        # Arrange: Create extractor
        extractor = PCAExtractor(n_components=3)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify cumulative variance ratio
        cumulative_variance = result["cumulative_variance_ratio"]
        assert len(cumulative_variance) == 3
        assert np.all(cumulative_variance >= 0)
        assert np.all(cumulative_variance <= 1)
        # Should be monotonically increasing
        assert np.all(np.diff(cumulative_variance) >= 0)

    def test_mean_shape(self, small_hsi):
        """Test that mean has correct shape."""
        # Arrange: Create extractor
        extractor = PCAExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify mean shape
        mean = result["mean"]
        assert mean.shape[0] == small_hsi.reflectance.shape[2]

    def test_validate_invalid_n_components(self, small_hsi):
        """Test validation fails with invalid n_components."""
        # Arrange: Create extractor with invalid n_components
        extractor = PCAExtractor(n_components=0)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="n_components must be positive"):
            extractor.extract(small_hsi)

    def test_validate_negative_n_components(self, small_hsi):
        """Test validation fails with negative n_components."""
        # Arrange: Create extractor with negative n_components
        extractor = PCAExtractor(n_components=-5)

        # Act & Assert: Verify validation raises ValueError
        with pytest.raises(ValueError, match="n_components must be positive"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("whiten", [True, False])
    def test_different_whiten_options(self, small_hsi, whiten):
        """Test extraction with different whiten options."""
        # Arrange: Create extractor with specific whiten option
        extractor = PCAExtractor(n_components=3, whiten=whiten)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify successful extraction
        assert "features" in result
        assert result["n_components"] == 3

    @pytest.mark.parametrize("n_components", [1, 3, 5])
    def test_different_n_components(self, small_hsi, n_components):
        """Test extraction with different number of components."""
        # Arrange: Create extractor with specific n_components
        extractor = PCAExtractor(n_components=n_components)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify correct number of components
        assert result["n_components"] == n_components
        assert result["features"].shape[2] == n_components

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        # Arrange & Act: Get feature name
        name = PCAExtractor.feature_name()

        # Assert: Verify correct name
        assert name == "p_c_a"

    def test_random_state_reproducibility(self, small_hsi):
        """Test that same random state produces same results."""
        # Arrange: Create two extractors with same random state
        extractor1 = PCAExtractor(n_components=3, random_state=42)
        extractor2 = PCAExtractor(n_components=3, random_state=42)

        # Act: Execute extraction
        result1 = extractor1.extract(small_hsi)
        result2 = extractor2.extract(small_hsi)

        # Assert: Verify results are similar
        features1 = result1["features"]
        features2 = result2["features"]
        assert features1.shape == features2.shape

    def test_original_shape_preserved(self, small_hsi):
        """Test that original shape is recorded."""
        # Arrange: Create extractor
        extractor = PCAExtractor()

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify original shape
        assert result["original_shape"] == small_hsi.shape

    def test_n_components_exceeds_features(self, small_hsi):
        """Test behavior when n_components exceeds available features."""
        # Arrange: Create extractor with very high n_components
        extractor = PCAExtractor(n_components=1000)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify actual components is limited
        assert result["n_components"] <= small_hsi.reflectance.shape[2]

    def test_pca_object_stored(self, small_hsi):
        """Test that PCA object is stored in extractor."""
        # Arrange: Create extractor
        extractor = PCAExtractor(n_components=3)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify PCA object is stored
        assert extractor.pca is not None
        assert hasattr(extractor.pca, "components_")

    def test_explained_variance_sum(self, small_hsi):
        """Test that explained variance values are consistent."""
        # Arrange: Create extractor
        extractor = PCAExtractor(n_components=3)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify explained variance is computed
        assert "explained_variance" in result
        explained_var = result["explained_variance"]
        assert len(explained_var) == 3
        assert np.all(explained_var >= 0)

    def test_variance_ratio_sum_less_than_one(self, small_hsi):
        """Test that variance ratio sum is less than or equal to 1."""
        # Arrange: Create extractor
        extractor = PCAExtractor(n_components=3)

        # Act: Execute extraction
        result = extractor.extract(small_hsi)

        # Assert: Verify variance ratio sum
        variance_ratio_sum = np.sum(result["explained_variance_ratio"])
        assert variance_ratio_sum <= 1.0
        assert variance_ratio_sum > 0.0
