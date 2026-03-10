"""Tests for ICAExtractor."""

from unittest.mock import patch as mock_patch

import numpy as np
import pytest
from sklearn.decomposition import FastICA

from hyppo.core import HSI
from hyppo.extractor.ica import ICAExtractor


class TestICAExtractor:
    """Test cases for ICAExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 5x5x8 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(5, 5, 8).astype(np.float32)
        wavelengths = np.linspace(400, 900, 8).astype(np.float32)
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression(self, regression_hsi):
        """Regression test: ICA output must not change."""
        # Arrange
        extractor = ICAExtractor(n_components=3, random_state=42)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        expected_row0 = np.array([
            [ 0.28009173, -0.8886466,  -1.18073642],
            [-0.99395734, -1.47108257,  1.66003823],
            [-0.43766952, -0.32563734, -0.43849036],
            [-0.82276291, -0.35997638,  0.53501362],
            [-0.32476985, -1.07117653, -1.39925683],
        ])
        np.testing.assert_allclose(
            result["features"][0, :, :], expected_row0, rtol=1e-5
        )

    def test_reference_sklearn(self):
        """Test ICA results match sklearn FastICA directly."""
        # Arrange
        np.random.seed(42)
        h, w, bands = 4, 4, 10
        reflectance = np.random.rand(h, w, bands).astype(np.float32)
        wavelengths = np.linspace(400, 1000, bands).astype(np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)

        n_components = 3
        extractor = ICAExtractor(n_components=n_components, random_state=42)

        # Act
        result = extractor.extract(hsi)

        # Assert: Compare with sklearn directly
        X_flat = reflectance.reshape(-1, bands)
        mean = X_flat.mean(axis=0)
        X_centered = X_flat - mean

        sklearn_ica = FastICA(
            n_components=n_components, whiten="unit-variance",
            random_state=42,
        )
        sklearn_features = sklearn_ica.fit_transform(X_centered)
        sklearn_features = sklearn_features.reshape(h, w, n_components)

        np.testing.assert_allclose(
            result["features"], sklearn_features, rtol=1e-5
        )
        np.testing.assert_allclose(
            result["components"], sklearn_ica.components_, rtol=1e-5
        )
        np.testing.assert_allclose(
            result["mixing_matrix"], sklearn_ica.mixing_, rtol=1e-5
        )

    def test_mathematical_properties(self):
        """Test Components @ Mixing^T = Identity."""
        # Arrange
        np.random.seed(42)
        reflectance = np.random.rand(5, 5, 8).astype(np.float32)
        wavelengths = np.linspace(400, 900, 8).astype(np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        extractor = ICAExtractor(n_components=4, random_state=42)

        # Act
        result = extractor.extract(hsi)

        # Assert
        product = result["components"] @ result["mixing_matrix"]
        np.testing.assert_allclose(product, np.eye(4), atol=1e-5)
        assert result["n_iter"] > 0

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = ICAExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        expected_keys = [
            "features", "components", "mixing_matrix", "mean",
            "n_components", "original_shape", "n_iter",
            "reconstruction_error", "valid_pixel_mask",
        ]
        for key in expected_keys:
            assert key in result

        assert result["n_components"] == 5
        features = result["features"]
        assert features.shape[:2] == (small_hsi.height, small_hsi.width)
        assert features.ndim == 3

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom parameters."""
        # Arrange
        extractor = ICAExtractor(
            n_components=3, whiten="arbitrary-variance", random_state=123,
        )

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_components"] == 3

    def test_components_and_mixing_shapes(self, small_hsi):
        """Test components and mixing matrix shapes."""
        # Arrange
        n_components = 3
        extractor = ICAExtractor(n_components=n_components)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        n_bands = small_hsi.reflectance.shape[2]
        assert result["components"].shape == (n_components, n_bands)
        assert result["mixing_matrix"].shape == (n_bands, n_components)

    def test_reconstruction_error_computed(self, small_hsi):
        """Test reconstruction error is a non-negative number."""
        # Arrange
        extractor = ICAExtractor(n_components=3)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["reconstruction_error"] is not None
        assert result["reconstruction_error"] >= 0

    def test_reconstruction_error_valueerror_fallback(self, small_hsi):
        """Test reconstruction_error is None when inverse_transform fails."""
        # Arrange
        extractor = ICAExtractor(n_components=3)

        # Act: mock inverse_transform to raise ValueError
        with mock_patch.object(
            FastICA, "inverse_transform",
            side_effect=ValueError("mock"),
        ):
            result = extractor.extract(small_hsi)

        # Assert
        assert result["reconstruction_error"] is None

    @pytest.mark.parametrize(
        "whiten", ["unit-variance", "arbitrary-variance"]
    )
    def test_different_whiten_strategies(self, small_hsi, whiten):
        """Test extraction with different whiten strategies."""
        # Arrange
        extractor = ICAExtractor(n_components=2, whiten=whiten)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_components"] == 2

    def test_whiten_false_warning(self, small_hsi):
        """Test whiten=False emits warning and runs successfully."""
        # Arrange
        extractor = ICAExtractor(n_components=2, whiten=False)

        # Act & Assert: warning emitted
        with pytest.warns(UserWarning, match="ill-posed"):
            result = extractor.extract(small_hsi)

        # Assert: extraction succeeds
        assert result["features"].ndim == 3

    @pytest.mark.parametrize("n_components", [1, 3, 5])
    def test_different_n_components(self, small_hsi, n_components):
        """Test extraction with different number of components."""
        # Arrange
        extractor = ICAExtractor(n_components=n_components)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_components"] == n_components
        assert result["features"].shape[2] == n_components

    def test_n_components_exceeds_features(self, small_hsi):
        """Test n_components is clamped to available features."""
        # Arrange
        extractor = ICAExtractor(n_components=1000)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_components"] <= small_hsi.reflectance.shape[2]

    def test_random_state_reproducibility(self, small_hsi):
        """Test same random state produces same results."""
        # Arrange
        ext1 = ICAExtractor(n_components=3, random_state=42)
        ext2 = ICAExtractor(n_components=3, random_state=42)

        # Act
        r1 = ext1.extract(small_hsi)
        r2 = ext2.extract(small_hsi)

        # Assert
        np.testing.assert_allclose(r1["features"], r2["features"])

    def test_original_shape_preserved(self, small_hsi):
        """Test original shape is recorded correctly."""
        # Arrange
        extractor = ICAExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["original_shape"] == small_hsi.shape

    def test_mean_removal(self, regression_hsi):
        """Test that mean is correctly computed and returned."""
        # Arrange
        extractor = ICAExtractor(n_components=3)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: mean matches manual computation
        X_flat = regression_hsi.reflectance.reshape(-1, 8)
        expected_mean = X_flat.mean(axis=0)
        np.testing.assert_allclose(result["mean"], expected_mean, rtol=1e-5)

    def test_ica_object_stored(self, small_hsi):
        """Test that ICA object is stored after extraction."""
        # Arrange
        extractor = ICAExtractor(n_components=3)

        # Act
        extractor.extract(small_hsi)

        # Assert
        assert extractor.ica is not None
        assert hasattr(extractor.ica, "components_")

    def test_valid_pixel_mask_all_valid(self, small_hsi):
        """Test valid_pixel_mask when all pixels are valid."""
        # Arrange
        extractor = ICAExtractor(n_components=3)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert np.all(result["valid_pixel_mask"])

    def test_validate_invalid_n_components(self, small_hsi):
        """Test validation fails with non-positive n_components."""
        extractor = ICAExtractor(n_components=0)
        with pytest.raises(ValueError, match="n_components must be positive"):
            extractor.extract(small_hsi)

    def test_validate_negative_n_components(self, small_hsi):
        """Test validation fails with negative n_components."""
        extractor = ICAExtractor(n_components=-5)
        with pytest.raises(ValueError, match="n_components must be positive"):
            extractor.extract(small_hsi)

    def test_validate_invalid_whiten(self, small_hsi):
        """Test validation fails with invalid whiten value."""
        extractor = ICAExtractor(whiten="invalid")  # type: ignore
        with pytest.raises(ValueError, match="whiten must be one of"):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert ICAExtractor.feature_name() == "ica"
