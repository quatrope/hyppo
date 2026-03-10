"""Tests for ZernikeMomentExtractor."""

import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.extractor.zernikemoment import ZernikeMomentExtractor


class TestZernikeMomentExtractor:
    """Test cases for ZernikeMomentExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 5x5x8 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(5, 5, 8).astype(np.float32)
        wavelengths = np.linspace(400, 900, 8).astype(np.float32)
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression(self, regression_hsi):
        """Regression test: ZernikeMoment output must not change."""
        # Arrange
        extractor = ZernikeMomentExtractor(
            n_components=2, max_order=2, window_sizes=[3],
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        expected_row0 = np.array([
            [5.83703470e-01, 0.0, 1.22115461e+00, 9.75248418e-01,
             4.43472353e-01, 0.0, 2.21736579e+00, 5.69444482e-01],
            [3.70571385e-01, 7.37852909e-02, 1.18976310e-01, 3.42186807e-01,
             1.03215616e-01, 2.17034350e-01, 1.48131480e+00, 2.27313583e-01],
            [1.45270217e-01, 2.68388451e-01, 1.27210722e-01, 1.37470489e+00,
             3.39800163e-01, 2.33354049e-01, 1.25524617e+00, 5.05950432e-01],
            [9.87498268e-02, 1.56948669e-01, 1.29275631e-01, 1.00455894e+00,
             1.63246309e-01, 2.34462147e-01, 1.81331206e-02, 1.42901382e+00],
            [7.27131087e-01, 0.0, 1.40194733e+00, 9.40620071e-01,
             1.47438067e-01, 0.0, 4.96917920e-01, 9.15909755e-01],
        ])
        np.testing.assert_allclose(
            result["features"][0, :, :], expected_row0, atol=1e-5
        )

    def test_magnitudes_non_negative(self, regression_hsi):
        """Test Zernike moment magnitudes are always non-negative."""
        # Arrange
        extractor = ZernikeMomentExtractor(
            n_components=2, max_order=4, window_sizes=[3],
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        assert np.all(result["features"] >= 0)

    def test_rotation_invariance(self):
        """Test Zernike moment magnitudes are rotation invariant.

        Reference: Khotanzad & Hong (1990) - Zernike moments provide
        rotation-invariant features through their magnitude.
        """
        # Arrange: create a simple asymmetric patch and its 90-degree rotation
        extractor = ZernikeMomentExtractor(max_order=4)
        patch = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        rotated = np.rot90(patch)

        patches = np.stack([patch, rotated])

        # Act
        moments = extractor._zernike_moments(patches)

        # Assert: magnitudes should be very similar (not exact due to grid)
        np.testing.assert_allclose(moments[0], moments[1], atol=0.5)

    def test_radial_polynomial_r00(self):
        """Test R_00(r) = 1 for all r."""
        # Arrange
        extractor = ZernikeMomentExtractor()
        r = np.linspace(0, 1, 100)

        # Act
        R00 = extractor._zernike_radial_poly(0, 0, r)

        # Assert
        np.testing.assert_allclose(R00, np.ones_like(r))

    def test_radial_polynomial_r11(self):
        """Test R_11(r) = r."""
        # Arrange
        extractor = ZernikeMomentExtractor()
        r = np.linspace(0, 1, 100)

        # Act
        R11 = extractor._zernike_radial_poly(1, 1, r)

        # Assert
        np.testing.assert_allclose(R11, r)

    def test_radial_polynomial_r20(self):
        """Test R_20(r) = 2r^2 - 1."""
        # Arrange
        extractor = ZernikeMomentExtractor()
        r = np.linspace(0, 1, 100)

        # Act
        R20 = extractor._zernike_radial_poly(2, 0, r)

        # Assert
        expected = 2 * r**2 - 1
        np.testing.assert_allclose(R20, expected)

    def test_feature_count_formula(self, small_hsi):
        """Test feature count matches formula."""
        # Arrange
        n_components = 2
        max_order = 4
        window_sizes = [3, 5]
        extractor = ZernikeMomentExtractor(
            n_components=n_components, max_order=max_order,
            window_sizes=window_sizes,
        )

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        n_moments = sum(
            1
            for p in range(max_order + 1)
            for q in range(0, p + 1)
            if (p - q) % 2 == 0
        )
        expected_features = n_components * len(window_sizes) * n_moments
        assert result["features"].shape[2] == expected_features
        assert result["n_moments_per_scale"] == n_moments

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = ZernikeMomentExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        expected_keys = [
            "features", "explained_variance_ratio", "n_components",
            "window_sizes", "max_order", "n_moments_per_scale",
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
        extractor = ZernikeMomentExtractor(
            n_components=2, max_order=4, window_sizes=[3, 5],
        )

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_components"] == 2
        assert result["max_order"] == 4
        assert result["window_sizes"] == [3, 5]

    @pytest.mark.parametrize("n_components", [1, 3, 5])
    def test_different_n_components(self, small_hsi, n_components):
        """Test extraction with different number of components."""
        # Arrange
        extractor = ZernikeMomentExtractor(n_components=n_components)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_components"] == n_components
        assert len(result["explained_variance_ratio"]) == n_components

    @pytest.mark.parametrize("max_order", [2, 4, 6])
    def test_different_max_orders(self, small_hsi, max_order):
        """Test extraction with different max orders."""
        # Arrange
        extractor = ZernikeMomentExtractor(max_order=max_order)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["max_order"] == max_order

    def test_pca_variance_explained(self, small_hsi):
        """Test that PCA variance ratios are valid."""
        # Arrange
        extractor = ZernikeMomentExtractor(n_components=2)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        variance_ratio = result["explained_variance_ratio"]
        assert len(variance_ratio) == 2
        assert np.all(variance_ratio >= 0)
        assert np.all(variance_ratio <= 1)

    def test_pca_object_stored(self, small_hsi):
        """Test that PCA object is stored in extractor."""
        # Arrange
        extractor = ZernikeMomentExtractor(n_components=3)

        # Act
        extractor.extract(small_hsi)

        # Assert
        assert extractor.pca is not None
        assert hasattr(extractor.pca, "explained_variance_ratio_")

    def test_validate_invalid_n_components(self, small_hsi):
        """Test validation fails with invalid n_components."""
        extractor = ZernikeMomentExtractor(n_components=0)
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_non_integer_n_components(self, small_hsi):
        """Test validation fails with non-integer n_components."""
        extractor = ZernikeMomentExtractor(
            n_components=2.5,  # type: ignore
        )
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_max_order(self, small_hsi):
        """Test validation fails with negative max_order."""
        extractor = ZernikeMomentExtractor(max_order=-1)
        with pytest.raises(
            ValueError, match="max_order must be a non-negative integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_non_integer_max_order(self, small_hsi):
        """Test validation fails with non-integer max_order."""
        extractor = ZernikeMomentExtractor(max_order=2.5)  # type: ignore
        with pytest.raises(
            ValueError, match="max_order must be a non-negative integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_empty_window_sizes(self, small_hsi):
        """Test validation fails with empty window_sizes."""
        extractor = ZernikeMomentExtractor(window_sizes=[])
        with pytest.raises(
            ValueError, match="window_sizes must be a non-empty list"
        ):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("invalid_window", [1, 2, 4])
    def test_validate_invalid_window_size(self, small_hsi, invalid_window):
        """Test validation fails with invalid window size values."""
        extractor = ZernikeMomentExtractor(
            window_sizes=[invalid_window],
        )
        with pytest.raises(
            ValueError, match="Each window size must be an odd integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_n_components_exceeds_bands(self, small_hsi):
        """Test validation fails when n_components exceeds spectral bands."""
        extractor = ZernikeMomentExtractor(n_components=100)
        with pytest.raises(ValueError, match="Number of spectral bands"):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert ZernikeMomentExtractor.feature_name() == "zernike_moment"
