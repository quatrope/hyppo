"""Tests for LegendreMomentExtractor."""

import numpy as np
import pytest
from scipy.special import legendre

from hyppo.core import FeatureSpace, HSI
from hyppo.extractor import PCAExtractor
from hyppo.extractor.legendremoment import LegendreMomentExtractor


class TestLegendreMomentExtractor:
    """Test cases for LegendreMomentExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 5x5x8 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(5, 5, 8).astype(np.float32)
        wavelengths = np.linspace(400, 900, 8).astype(np.float32)
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    @pytest.fixture
    def pca_result(self, small_hsi):
        """Pre-computed PCA result for unit tests."""
        pca_extractor = PCAExtractor(n_components=3)
        return pca_extractor.extract(small_hsi)

    @pytest.fixture
    def pca_result_2(self, small_hsi):
        """Pre-computed PCA result with 2 components."""
        pca_extractor = PCAExtractor(n_components=2)
        return pca_extractor.extract(small_hsi)

    def test_feature_space_integration(self, small_hsi):
        """Test full pipeline with FeatureSpace dependency resolution."""
        # Arrange
        fs = FeatureSpace.from_list(
            [
                PCAExtractor(n_components=3),
                LegendreMomentExtractor(
                    n_components=3,
                    max_order=2,
                    window_sizes=[3],
                ),
            ]
        )

        # Act
        results = fs.extract(small_hsi)

        # Assert
        lm_feature = results["legendre_moment"]
        lm_data = lm_feature.data
        assert "features" in lm_data
        assert lm_data["features"].ndim == 3
        assert lm_data["features"].shape[:2] == (
            small_hsi.height,
            small_hsi.width,
        )
        assert "explained_variance_ratio" in lm_data

    def test_regression(self, regression_hsi):
        """Regression test: LegendreMoment output must not change."""
        # Arrange
        pca_extractor = PCAExtractor(n_components=2)
        pca_result = pca_extractor.extract(regression_hsi)
        extractor = LegendreMomentExtractor(
            n_components=2,
            max_order=2,
            window_sizes=[3],
        )

        # Act
        result = extractor._extract(regression_hsi, pca=pca_result)

        # Assert
        expected_row0 = np.array(
            [
                [
                    9.97608833e-01,
                    0.00000000e00,
                    -3.95977124e-01,
                    0.00000000e00,
                    0.00000000e00,
                    1.31675774e00,
                    -3.46753195e-01,
                    0.00000000e00,
                    5.03492547e-01,
                    0.00000000e00,
                    0.00000000e00,
                    -4.96567950e-01,
                ],
                [
                    4.29643739e-01,
                    0.00000000e00,
                    -8.56296285e-01,
                    -8.27742833e-01,
                    0.00000000e00,
                    -2.55346555e-01,
                    -3.18485800e-01,
                    0.00000000e00,
                    -6.69239948e-01,
                    -3.53822404e-02,
                    0.00000000e00,
                    -2.70030995e-01,
                ],
                [
                    2.74186172e-02,
                    0.00000000e00,
                    -1.66399822e00,
                    -8.52675914e-01,
                    0.00000000e00,
                    7.50263569e-01,
                    -7.99426313e-01,
                    0.00000000e00,
                    -5.51706389e-01,
                    -7.48670600e-01,
                    0.00000000e00,
                    -1.44025840e00,
                ],
                [
                    5.84230237e-01,
                    0.00000000e00,
                    -2.26909347e-02,
                    1.94817042e00,
                    0.00000000e00,
                    1.74151917e00,
                    -1.89082738e-01,
                    0.00000000e00,
                    -1.04052191e00,
                    1.00818552e00,
                    0.00000000e00,
                    1.46911743e00,
                ],
                [
                    6.59902662e-01,
                    0.00000000e00,
                    4.38455338e-02,
                    0.00000000e00,
                    0.00000000e00,
                    -1.60807490e00,
                    -6.49595305e-01,
                    0.00000000e00,
                    -1.44904486e00,
                    0.00000000e00,
                    0.00000000e00,
                    -3.05756894e00,
                ],
            ]
        )
        np.testing.assert_allclose(
            result["features"][0, :, :], expected_row0, atol=1e-5
        )

    def test_legendre_polynomial_orthogonality(self):
        """Test Legendre polynomials P_m and P_n are orthogonal for m != n.

        Reference: Teague (1980) - orthogonality of Legendre polynomials.
        Integral of P_m(x)*P_n(x) over [-1,1] = 0 for m != n.
        """
        # Arrange: evaluate polynomials on fine grid for numerical integration
        x = np.linspace(-1, 1, 10000)
        dx = x[1] - x[0]

        # Act & Assert: check orthogonality for orders 0-4
        for m in range(5):
            for n in range(m + 1, 5):
                Pm = legendre(m)(x)
                Pn = legendre(n)(x)
                integral = np.sum(Pm * Pn) * dx
                assert np.isclose(
                    integral, 0.0, atol=1e-3
                ), f"P_{m} and P_{n} not orthogonal: integral={integral}"

    def test_m00_for_constant_patch(self):
        """Test L00 moment for a constant patch equals value * area * norm."""
        # Arrange
        extractor = LegendreMomentExtractor(max_order=1)
        constant_value = 3.0
        patches = np.full((1, 5, 5), constant_value)

        # Act
        moments = extractor._legendre_moments(patches)

        # Assert: L00 = norm_factor * sum(P0(x)*P0(y)*f)
        # P0 = 1, norm = sqrt(1*1)/2 = 0.5
        expected_l00 = 0.5 * constant_value * 25
        assert np.isclose(moments[0, 0], expected_l00, rtol=1e-5)

    def test_feature_count_formula(self, small_hsi, pca_result_2):
        """Test feature count matches formula."""
        # Arrange
        n_components = 2
        max_order = 3
        window_sizes = [3, 5]
        extractor = LegendreMomentExtractor(
            n_components=n_components,
            max_order=max_order,
            window_sizes=window_sizes,
        )

        # Act
        result = extractor._extract(small_hsi, pca=pca_result_2)

        # Assert
        n_moments = sum(
            1 for p in range(max_order + 1) for q in range(max_order + 1 - p)
        )
        expected_features = n_components * len(window_sizes) * n_moments
        assert result["features"].shape[2] == expected_features
        assert result["n_moments_per_scale"] == n_moments

    def test_extract_basic_with_defaults(self, small_hsi, pca_result):
        """Test extraction with default parameters."""
        # Arrange
        extractor = LegendreMomentExtractor()

        # Act
        result = extractor._extract(small_hsi, pca=pca_result)

        # Assert
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

    def test_extract_with_custom_parameters(self, small_hsi, pca_result_2):
        """Test extraction with custom parameters."""
        # Arrange
        extractor = LegendreMomentExtractor(
            n_components=2,
            max_order=2,
            window_sizes=[3, 5],
        )

        # Act
        result = extractor._extract(small_hsi, pca=pca_result_2)

        # Assert
        assert result["n_components"] == 2
        assert result["max_order"] == 2
        assert result["window_sizes"] == [3, 5]

    @pytest.mark.parametrize("n_components", [1, 3, 5])
    def test_different_n_components(self, small_hsi, n_components):
        """Test extraction with different number of components."""
        # Arrange
        pca_result = PCAExtractor(
            n_components=n_components,
        ).extract(small_hsi)
        extractor = LegendreMomentExtractor(n_components=n_components)

        # Act
        result = extractor._extract(small_hsi, pca=pca_result)

        # Assert
        assert result["n_components"] == n_components
        assert len(result["explained_variance_ratio"]) == n_components

    @pytest.mark.parametrize("max_order", [1, 2, 4])
    def test_different_max_orders(self, small_hsi, pca_result, max_order):
        """Test extraction with different max orders."""
        # Arrange
        extractor = LegendreMomentExtractor(max_order=max_order)

        # Act
        result = extractor._extract(small_hsi, pca=pca_result)

        # Assert
        assert result["max_order"] == max_order

    def test_pca_variance_explained(self, small_hsi, pca_result_2):
        """Test that PCA variance ratios are valid."""
        # Arrange
        extractor = LegendreMomentExtractor(n_components=2)

        # Act
        result = extractor._extract(small_hsi, pca=pca_result_2)

        # Assert
        variance_ratio = result["explained_variance_ratio"]
        assert len(variance_ratio) == 2
        assert np.all(variance_ratio >= 0)
        assert np.all(variance_ratio <= 1)

    def test_validate_invalid_n_components(self, small_hsi):
        """Test validation fails with invalid n_components."""
        extractor = LegendreMomentExtractor(n_components=0)
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_non_integer_n_components(self, small_hsi):
        """Test validation fails with non-integer n_components."""
        extractor = LegendreMomentExtractor(
            n_components=2.5,  # type: ignore
        )
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_max_order(self, small_hsi):
        """Test validation fails with negative max_order."""
        extractor = LegendreMomentExtractor(max_order=-1)
        with pytest.raises(
            ValueError, match="max_order must be a non-negative integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_non_integer_max_order(self, small_hsi):
        """Test validation fails with non-integer max_order."""
        extractor = LegendreMomentExtractor(
            max_order=2.5,  # type: ignore
        )
        with pytest.raises(
            ValueError, match="max_order must be a non-negative integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_empty_window_sizes(self, small_hsi):
        """Test validation fails with empty window_sizes."""
        extractor = LegendreMomentExtractor(window_sizes=[])
        with pytest.raises(
            ValueError, match="window_sizes must be a non-empty list"
        ):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("invalid_window", [1, 2, 4])
    def test_validate_invalid_window_size(self, small_hsi, invalid_window):
        """Test validation fails with invalid window size values."""
        extractor = LegendreMomentExtractor(
            window_sizes=[invalid_window],
        )
        with pytest.raises(
            ValueError, match="Each window size must be an odd integer"
        ):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert LegendreMomentExtractor.feature_name() == "legendre_moment"

    def test_get_input_dependencies(self):
        """Test that PCA is declared as input dependency."""
        deps = LegendreMomentExtractor.get_input_dependencies()
        assert "pca" in deps
        assert deps["pca"]["extractor"] is PCAExtractor
        assert deps["pca"]["required"] is False

    def test_get_input_default(self):
        """Test that default PCA extractor is provided."""
        default = LegendreMomentExtractor.get_input_default("pca")
        assert isinstance(default, PCAExtractor)
        assert default.n_components == 3

    def test_get_input_default_unknown_returns_none(self):
        """Test that unknown input names return None."""
        assert LegendreMomentExtractor.get_input_default("other") is None
