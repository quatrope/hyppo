"""Tests for GeometricMomentExtractor."""

import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.extractor.geometricmoment import GeometricMomentExtractor


class TestGeometricMomentExtractor:
    """Test cases for GeometricMomentExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 5x5x8 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(5, 5, 8).astype(np.float32)
        wavelengths = np.linspace(400, 900, 8).astype(np.float32)
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression(self, regression_hsi):
        """Regression test: GeometricMoment output must not change."""
        # Arrange
        extractor = GeometricMomentExtractor(
            n_components=2,
            max_order=2,
            window_sizes=[3],
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        expected_row0 = np.array(
            [
                [
                    1.99521767,
                    0.0,
                    0.42895742,
                    0.0,
                    0.0,
                    1.45023517,
                    -0.69350639,
                    0.0,
                    0.06905615,
                    0.0,
                    0.0,
                    -0.52726471,
                ],
                [
                    0.85928748,
                    0.0,
                    -0.22416729,
                    -0.95579509,
                    0.0,
                    0.13416989,
                    -0.6369716,
                    0.0,
                    -0.61138147,
                    -0.04085589,
                    0.0,
                    -0.37333924,
                ],
                [
                    0.05483723,
                    0.0,
                    -0.97393776,
                    -0.98458534,
                    0.0,
                    0.46564984,
                    -1.59885263,
                    0.0,
                    -0.86192501,
                    -0.86449035,
                    0.0,
                    -1.39175506,
                ],
                [
                    1.16846047,
                    0.0,
                    0.37595657,
                    2.24955343,
                    0.0,
                    1.42792822,
                    -0.37816548,
                    0.0,
                    -0.74650255,
                    1.16415236,
                    0.0,
                    0.74995723,
                ],
                [
                    1.31980532,
                    0.0,
                    0.46607953,
                    0.0,
                    0.0,
                    -0.5189355,
                    -1.29919061,
                    0.0,
                    -1.29710695,
                    0.0,
                    0.0,
                    -2.2562454,
                ],
            ]
        )
        np.testing.assert_allclose(
            result["features"][0, :, :], expected_row0, rtol=1e-5
        )

    def test_m00_equals_sum_of_patch(self):
        """Test M00 moment equals sum of pixel values in the patch.

        For a constant image with value c and window size w,
        M00 = c * w^2 (sum of all pixels).
        """
        # Arrange
        extractor = GeometricMomentExtractor(
            max_order=2,
            normalize_coords=True,
        )
        constant_value = 2.0
        patches = np.full((1, 3, 3), constant_value)

        # Act
        moments = extractor._geometric_moments(patches)

        # Assert: M00 (first moment) = constant * n_pixels
        assert np.isclose(moments[0, 0], constant_value * 9)

    def test_m10_m01_zero_for_symmetric_patch(self):
        """Test M10 and M01 are zero for symmetric constant patches.

        With normalized coordinates centered at 0, a constant patch
        has zero first-order moments.
        """
        # Arrange
        extractor = GeometricMomentExtractor(
            max_order=2,
            normalize_coords=True,
        )
        patches = np.ones((1, 5, 5))

        # Act
        moments = extractor._geometric_moments(patches)

        # Assert: M01 (index 1) and M10 (index 3) are zero
        # Ordering: (p=0,q=0), (p=0,q=1), (p=0,q=2), (p=1,q=0), ...
        assert np.isclose(moments[0, 1], 0.0, atol=1e-10)  # M01
        assert np.isclose(moments[0, 3], 0.0, atol=1e-10)  # M10

    def test_normalize_coords_false(self, small_hsi):
        """Test extraction with normalize_coords=False."""
        # Arrange
        extractor = GeometricMomentExtractor(
            n_components=2,
            max_order=2,
            window_sizes=[3],
            normalize_coords=False,
        )

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["features"].ndim == 3
        assert result["features"].shape[:2] == (
            small_hsi.height,
            small_hsi.width,
        )

    def test_normalize_coords_produces_different_results(self, small_hsi):
        """Test that normalize_coords=True and False give different results."""
        # Arrange
        ext_norm = GeometricMomentExtractor(
            n_components=2,
            max_order=2,
            window_sizes=[3],
            normalize_coords=True,
        )
        ext_no_norm = GeometricMomentExtractor(
            n_components=2,
            max_order=2,
            window_sizes=[3],
            normalize_coords=False,
        )

        # Act
        result_norm = ext_norm.extract(small_hsi)
        result_no_norm = ext_no_norm.extract(small_hsi)

        # Assert
        assert not np.allclose(
            result_norm["features"],
            result_no_norm["features"],
        )

    def test_feature_count_formula(self, small_hsi):
        """Test feature count matches formula."""
        # Arrange
        n_components = 2
        max_order = 3
        window_sizes = [3, 5]
        extractor = GeometricMomentExtractor(
            n_components=n_components,
            max_order=max_order,
            window_sizes=window_sizes,
        )

        # Act
        result = extractor.extract(small_hsi)

        # Assert: n_moments = sum(1 for p,q valid pairs)
        n_moments = sum(
            1 for p in range(max_order + 1) for q in range(max_order + 1 - p)
        )
        expected_features = n_components * len(window_sizes) * n_moments
        assert result["features"].shape[2] == expected_features
        assert result["n_moments_per_scale"] == n_moments

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = GeometricMomentExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        expected_keys = [
            "features",
            "explained_variance_ratio",
            "n_components",
            "window_sizes",
            "max_order",
            "n_moments_per_scale",
        ]
        for key in expected_keys:
            assert key in result

        assert result["n_components"] == 3
        assert result["window_sizes"] == [3, 9, 15]
        assert result["max_order"] == 6

        features = result["features"]
        assert features.shape[:2] == (small_hsi.height, small_hsi.width)
        assert features.ndim == 3

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom parameters."""
        # Arrange
        extractor = GeometricMomentExtractor(
            n_components=2,
            max_order=2,
            window_sizes=[3, 5],
        )

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_components"] == 2
        assert result["max_order"] == 2
        assert result["window_sizes"] == [3, 5]

    @pytest.mark.parametrize("n_components", [1, 3, 5])
    def test_different_n_components(self, small_hsi, n_components):
        """Test extraction with different number of components."""
        # Arrange
        extractor = GeometricMomentExtractor(n_components=n_components)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_components"] == n_components
        assert len(result["explained_variance_ratio"]) == n_components

    @pytest.mark.parametrize("max_order", [1, 2, 4])
    def test_different_max_orders(self, small_hsi, max_order):
        """Test extraction with different max orders."""
        # Arrange
        extractor = GeometricMomentExtractor(max_order=max_order)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["max_order"] == max_order

    def test_pca_variance_explained(self, small_hsi):
        """Test that PCA variance ratios are valid."""
        # Arrange
        extractor = GeometricMomentExtractor(n_components=2)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        variance_ratio = result["explained_variance_ratio"]
        assert len(variance_ratio) == 2
        assert np.all(variance_ratio >= 0)
        assert np.all(variance_ratio <= 1)

    def test_validate_invalid_n_components(self, small_hsi):
        """Test validation fails with invalid n_components."""
        extractor = GeometricMomentExtractor(n_components=0)
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_non_integer_n_components(self, small_hsi):
        """Test validation fails with non-integer n_components."""
        extractor = GeometricMomentExtractor(n_components=2.5)  # type: ignore
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_max_order(self, small_hsi):
        """Test validation fails with negative max_order."""
        extractor = GeometricMomentExtractor(max_order=-1)
        with pytest.raises(
            ValueError, match="max_order must be a non-negative integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_non_integer_max_order(self, small_hsi):
        """Test validation fails with non-integer max_order."""
        extractor = GeometricMomentExtractor(max_order=2.5)  # type: ignore
        with pytest.raises(
            ValueError, match="max_order must be a non-negative integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_empty_window_sizes(self, small_hsi):
        """Test validation fails with empty window_sizes."""
        extractor = GeometricMomentExtractor(window_sizes=[])
        with pytest.raises(
            ValueError, match="window_sizes must be a non-empty list"
        ):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("invalid_window", [1, 2, 4])
    def test_validate_invalid_window_size(self, small_hsi, invalid_window):
        """Test validation fails with invalid window size values."""
        extractor = GeometricMomentExtractor(
            window_sizes=[invalid_window],
        )
        with pytest.raises(
            ValueError, match="Each window size must be an odd integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_n_components_exceeds_bands(self, small_hsi):
        """Test validation fails when n_components exceeds spectral bands."""
        extractor = GeometricMomentExtractor(n_components=100)
        with pytest.raises(ValueError, match="Number of spectral bands"):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert GeometricMomentExtractor.feature_name() == "geometric_moment"
