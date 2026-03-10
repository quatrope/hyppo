"""Tests for PPExtractor."""

from unittest.mock import patch as mock_patch

import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.extractor.pp import PPExtractor


class TestPPExtractor:
    """Test cases for PPExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 5x5x8 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(5, 5, 8).astype(np.float32)
        wavelengths = np.linspace(400, 900, 8).astype(np.float32)
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression(self, regression_hsi):
        """Regression test: PP output must not change."""
        # Arrange
        extractor = PPExtractor(n_projections=3, random_state=42)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        expected_row0 = np.array([
            [-0.79909122,  0.20852612,  0.03628006],
            [ 1.96342134, -0.2751492,   1.53290534],
            [ 0.18659377,  0.77047849,  1.17713058],
            [ 0.92933339,  0.55912536,  0.60846531],
            [ 0.15774821,  1.43000901, -2.0064218],
        ])
        np.testing.assert_allclose(
            result["features"][0, :, :], expected_row0, rtol=1e-5
        )

    def test_projection_vectors_unit_norm(self, regression_hsi):
        """Test projection vectors have unit norm."""
        # Arrange
        extractor = PPExtractor(n_projections=3, random_state=42)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        norms = np.linalg.norm(result["projection_vectors"], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_divergence_scores_non_negative(self, regression_hsi):
        """Test divergence scores are non-negative."""
        # Arrange
        extractor = PPExtractor(n_projections=3, random_state=42)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        for score in result["divergence_scores"]:
            assert score >= 0

    def test_information_divergence_gaussian_is_low(self):
        """Test divergence is low for Gaussian data (near-zero)."""
        # Arrange
        rng = np.random.RandomState(42)
        gaussian_scores = rng.randn(10000)
        extractor = PPExtractor()

        # Act
        divergence = extractor._compute_information_divergence(
            gaussian_scores
        )

        # Assert: divergence from Gaussian should be small
        assert divergence < 0.1

    def test_information_divergence_uniform_is_high(self):
        """Test divergence is higher for uniform (non-Gaussian) data."""
        # Arrange
        rng = np.random.RandomState(42)
        uniform_scores = rng.uniform(-3, 3, size=10000)
        gaussian_scores = rng.randn(10000)
        extractor = PPExtractor()

        # Act
        div_uniform = extractor._compute_information_divergence(
            uniform_scores
        )
        div_gaussian = extractor._compute_information_divergence(
            gaussian_scores
        )

        # Assert: uniform has higher divergence from Gaussian
        assert div_uniform > div_gaussian

    def test_pca_auto_selection(self, small_hsi):
        """Test PCA auto-selects components when pca_components=None."""
        # Arrange
        extractor = PPExtractor(n_projections=2, pca_components=None)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["pca_components_used"] > 0
        assert result["pca_model"] is not None

    def test_pca_explicit_components(self, small_hsi):
        """Test PCA uses explicit component count."""
        # Arrange
        extractor = PPExtractor(n_projections=2, pca_components=3)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["pca_components_used"] == 3

    def test_sample_size_larger_than_data(self, small_hsi):
        """Test when sample_size > n_pixels (no sampling needed)."""
        # Arrange: small_hsi is 3x3=9 pixels, sample_size=1000 > 9
        extractor = PPExtractor(n_projections=2, sample_size=1000)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["features"].ndim == 3

    def test_sample_size_smaller_than_data(self):
        """Test when sample_size < n_pixels (sampling is used)."""
        # Arrange: 10x10=100 pixels, sample_size=20
        rng = np.random.RandomState(42)
        reflectance = rng.rand(10, 10, 5).astype(np.float32)
        wavelengths = np.linspace(400, 800, 5).astype(np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        extractor = PPExtractor(
            n_projections=2, sample_size=20, random_state=42,
        )

        # Act
        result = extractor.extract(hsi)

        # Assert
        assert result["features"].shape == (10, 10, 2)

    def test_zero_norm_candidate_skipped(self):
        """Test that zero-norm candidate vectors are skipped."""
        # Arrange: create data with a zero-row
        rng = np.random.RandomState(42)
        reflectance = rng.rand(4, 4, 5).astype(np.float32)
        reflectance[0, 0, :] = 0.0
        wavelengths = np.linspace(400, 800, 5).astype(np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        extractor = PPExtractor(n_projections=2, random_state=42)

        # Act
        result = extractor.extract(hsi)

        # Assert: extraction succeeds despite zero vector
        assert result["features"].ndim == 3

    def test_divergence_computation_error_skipped(self, small_hsi):
        """Test that ValueError in divergence is skipped gracefully."""
        # Arrange
        extractor = PPExtractor(n_projections=2, random_state=42)

        original_divergence = extractor._compute_information_divergence

        call_count = [0]

        def mock_divergence(scores):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise ValueError("mock divergence failure")
            return original_divergence(scores)

        # Act
        with mock_patch.object(
            extractor, "_compute_information_divergence",
            side_effect=mock_divergence,
        ):
            result = extractor.extract(small_hsi)

        # Assert
        assert result["features"].ndim == 3

    def test_fallback_random_projection(self, small_hsi):
        """Test fallback to random projection when all candidates fail."""
        # Arrange
        extractor = PPExtractor(n_projections=1, random_state=42)

        # Act: mock divergence to always raise
        with mock_patch.object(
            extractor, "_compute_information_divergence",
            side_effect=ValueError("always fail"),
        ):
            with pytest.warns(UserWarning, match="fallback to random"):
                result = extractor.extract(small_hsi)

        # Assert
        assert result["features"].ndim == 3
        assert result["divergence_scores"][0] == 0.0

    def test_no_valid_pixels_raises(self):
        """Test ValueError when all pixels are NaN."""
        # Arrange
        reflectance = np.full((3, 3, 5), np.nan, dtype=np.float32)
        wavelengths = np.linspace(400, 800, 5).astype(np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        extractor = PPExtractor(n_projections=2)

        # Act & Assert
        with pytest.raises(ValueError, match="No valid pixels"):
            extractor.extract(hsi)

    def test_valid_pixel_mask(self, small_hsi):
        """Test valid pixel mask is computed correctly."""
        # Arrange
        extractor = PPExtractor(n_projections=2)

        # Act
        result = extractor.extract(small_hsi)

        # Assert: all valid for clean data
        assert np.all(result["valid_pixel_mask"])
        assert result["valid_pixel_mask"].shape == (
            small_hsi.height, small_hsi.width,
        )

    def test_deflation_produces_different_projections(self, regression_hsi):
        """Test that deflation makes successive projections different."""
        # Arrange
        extractor = PPExtractor(n_projections=3, random_state=42)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: projection vectors should not be identical
        vectors = result["projection_vectors"]
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                assert not np.allclose(vectors[i], vectors[j])

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = PPExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        expected_keys = [
            "features", "projection_vectors", "n_features",
            "original_shape", "divergence_scores",
            "pca_components_used", "pca_model",
            "selected_pixel_indices", "valid_pixel_mask",
        ]
        for key in expected_keys:
            assert key in result

        assert result["n_features"] == 10
        features = result["features"]
        assert features.shape[:2] == (small_hsi.height, small_hsi.width)
        assert features.ndim == 3

    def test_original_shape_preserved(self, small_hsi):
        """Test original shape is recorded correctly."""
        # Arrange
        extractor = PPExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["original_shape"] == small_hsi.shape

    def test_random_state_reproducibility(self, small_hsi):
        """Test same random state produces same results."""
        # Arrange
        extractor1 = PPExtractor(n_projections=3, random_state=42)
        extractor2 = PPExtractor(n_projections=3, random_state=42)

        # Act
        result1 = extractor1.extract(small_hsi)
        result2 = extractor2.extract(small_hsi)

        # Assert
        np.testing.assert_allclose(
            result1["features"], result2["features"]
        )

    def test_validate_invalid_n_projections(self, small_hsi):
        """Test validation fails with non-positive n_projections."""
        extractor = PPExtractor(n_projections=0)
        with pytest.raises(
            ValueError, match="n_projections must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_negative_n_projections(self, small_hsi):
        """Test validation fails with negative n_projections."""
        extractor = PPExtractor(n_projections=-5)
        with pytest.raises(
            ValueError, match="n_projections must be positive"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_n_bins(self, small_hsi):
        """Test validation fails with non-positive n_bins."""
        extractor = PPExtractor(n_bins=0)
        with pytest.raises(ValueError, match="n_bins must be positive"):
            extractor.extract(small_hsi)

    def test_validate_invalid_sample_size(self, small_hsi):
        """Test validation fails with non-positive sample_size."""
        extractor = PPExtractor(sample_size=0)
        with pytest.raises(ValueError, match="sample_size must be positive"):
            extractor.extract(small_hsi)

    def test_validate_invalid_pca_components(self, small_hsi):
        """Test validation fails with non-positive pca_components."""
        extractor = PPExtractor(pca_components=-1)
        with pytest.raises(
            ValueError, match="pca_components must be positive"
        ):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert PPExtractor.feature_name() == "pp"
