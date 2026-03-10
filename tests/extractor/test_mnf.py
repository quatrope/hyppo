"""Tests for MNFExtractor."""

from unittest.mock import patch as mock_patch

import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.extractor.mnf import MNFExtractor


class TestMNFExtractor:
    """Test cases for MNFExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 5x5x8 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(5, 5, 8).astype(np.float32)
        wavelengths = np.linspace(400, 900, 8).astype(np.float32)
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression(self, regression_hsi):
        """Regression test: MNF output must not change."""
        # Arrange
        extractor = MNFExtractor(n_components=3)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        expected_row0 = np.array([
            [ 0.16387971, -3.07541917, -1.66650856],
            [-0.81401273, -0.81344,     0.33103433],
            [-0.40692347,  0.41655914,  0.29755285],
            [ 1.17871348,  0.1943873,  -0.26803931],
            [ 3.42934909, -1.99904697,  0.10944771],
        ])
        np.testing.assert_allclose(
            result["features"][0, :, :], expected_row0, rtol=1e-5
        )

        expected_snr_eigenvalues = np.array(
            [2.40086154, 1.59773334, 1.17495214]
        )
        np.testing.assert_allclose(
            result["snr_eigenvalues"], expected_snr_eigenvalues, rtol=1e-5
        )

    def test_mnf_mathematical_properties(self):
        """Test MNF satisfies mathematical properties.

        Reference: Green et al. (1988) - A transformation for ordering
        multispectral data in terms of image quality.

        Properties verified:
        - SNR eigenvalues are sorted in descending order
        - Whitening matrix is symmetric
        - SNR ratios sum to <= 1
        - Cumulative SNR is monotonically increasing
        - SNR estimates are non-negative
        """
        # Arrange
        np.random.seed(42)
        reflectance = np.random.rand(6, 6, 10).astype(np.float32)
        wavelengths = np.linspace(400, 1000, 10).astype(np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        extractor = MNFExtractor(n_components=5)

        # Act
        result = extractor.extract(hsi)

        # Assert: SNR eigenvalues are descending
        snr_eigenvalues = result["snr_eigenvalues"]
        assert np.all(np.diff(snr_eigenvalues) <= 1e-10)

        # Assert: Whitening matrix is symmetric
        W = result["whitening_matrix"]
        np.testing.assert_allclose(W, W.T, atol=1e-10)

        # Assert: SNR ratios sum to <= 1
        assert result["snr_ratio"].sum() <= 1.0 + 1e-6

        # Assert: Cumulative SNR is monotonically increasing
        cumulative = result["cumulative_snr_ratio"]
        assert np.all(np.diff(cumulative) >= -1e-10)

        # Assert: SNR estimates are non-negative
        assert np.all(result["snr_estimates"] >= 0)

    def test_mnf_noise_whitening(self):
        """Test that MNF correctly whitens the noise.

        After whitening, the noise covariance should be approximately identity.
        """
        # Arrange
        np.random.seed(42)
        h, w, bands = 8, 8, 6
        reflectance = np.random.rand(h, w, bands).astype(np.float32)
        wavelengths = np.linspace(400, 900, bands).astype(np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        extractor = MNFExtractor(n_components=4)

        # Act
        result = extractor.extract(hsi)

        # Recompute whitened noise covariance
        X = reflectance
        diffs_h = X[1:, :, :] - X[:-1, :, :]
        diffs_w = X[:, 1:, :] - X[:, :-1, :]
        diffs = np.concatenate(
            [diffs_h.reshape(-1, bands), diffs_w.reshape(-1, bands)], axis=0
        )
        noise_cov = np.cov(diffs.T) / 2.0

        # Apply whitening
        W = result["whitening_matrix"]
        whitened_noise_cov = W @ noise_cov @ W.T

        # Assert: Whitened noise covariance is approximately identity
        np.testing.assert_allclose(
            whitened_noise_cov, np.eye(bands), atol=1e-5
        )

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = MNFExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        expected_keys = [
            "features", "n_features", "original_shape", "mean",
            "noise_eigenvalues", "snr_eigenvalues", "snr_estimates",
            "snr_ratio", "cumulative_snr_ratio", "whitening_matrix",
            "projection_matrix",
        ]
        for key in expected_keys:
            assert key in result

        # Assert: Verify default n_components
        assert result["n_features"] == 5

        # Assert: Verify feature shape
        features = result["features"]
        assert features.shape[:2] == (small_hsi.height, small_hsi.width)
        assert features.ndim == 3

    def test_extract_with_custom_n_components(self, small_hsi):
        """Test extraction with custom number of components."""
        # Arrange
        extractor = MNFExtractor(n_components=3)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_features"] == 3
        assert result["features"].shape[2] == 3

    @pytest.mark.parametrize("n_components", [1, 3, 5])
    def test_different_n_components(self, small_hsi, n_components):
        """Test extraction with different number of components."""
        # Arrange
        extractor = MNFExtractor(n_components=n_components)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_features"] == n_components
        assert result["features"].shape[2] == n_components

    def test_n_components_exceeds_bands(self, small_hsi):
        """Test n_components is clamped to available bands."""
        # Arrange
        extractor = MNFExtractor(n_components=1000)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_features"] <= small_hsi.n_bands

    def test_projection_matrix_shape(self, small_hsi):
        """Test projection matrix has correct shape."""
        # Arrange
        n_components = 3
        extractor = MNFExtractor(n_components=n_components)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        n_bands = small_hsi.n_bands
        assert result["projection_matrix"].shape == (n_bands, n_components)
        assert result["whitening_matrix"].shape == (n_bands, n_bands)

    def test_mean_shape(self, small_hsi):
        """Test mean has correct shape."""
        # Arrange
        extractor = MNFExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["mean"].shape == (small_hsi.n_bands,)

    def test_original_shape_preserved(self, small_hsi):
        """Test original shape is recorded correctly."""
        # Arrange
        extractor = MNFExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["original_shape"] == small_hsi.shape

    def test_negative_eigenvalues_warning(self):
        """Test warning is emitted when noise eigenvalues are negative."""
        # Arrange
        reflectance = np.random.RandomState(42).rand(
            4, 4, 5
        ).astype(np.float32)
        wavelengths = np.linspace(400, 800, 5).astype(np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        extractor = MNFExtractor(n_components=3)

        # Act & Assert: mock eigh to return negative eigenvalues
        original_eigh = np.linalg.eigh

        call_count = [0]

        def mock_eigh(matrix):
            call_count[0] += 1
            eigvals, eigvecs = original_eigh(matrix)
            if call_count[0] == 1:
                eigvals[0] = -0.5
            return eigvals, eigvecs

        with mock_patch("numpy.linalg.eigh", side_effect=mock_eigh):
            with pytest.warns(UserWarning, match="negative noise eigenvalues"):
                result = extractor.extract(hsi)

        # Assert: extraction still succeeds
        assert result["features"].ndim == 3

    def test_total_snr_zero_fallback(self):
        """Test snr_ratio is zeros when total_snr is zero."""
        # Arrange
        reflectance = np.random.RandomState(42).rand(
            4, 4, 5
        ).astype(np.float32)
        wavelengths = np.linspace(400, 800, 5).astype(np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        extractor = MNFExtractor(n_components=3)

        # Act: mock eigh to return zero eigenvalues on second call
        original_eigh = np.linalg.eigh
        call_count = [0]

        def mock_eigh(matrix):
            call_count[0] += 1
            eigvals, eigvecs = original_eigh(matrix)
            if call_count[0] == 2:
                eigvals[:] = 0.0
            return eigvals, eigvecs

        with mock_patch("numpy.linalg.eigh", side_effect=mock_eigh):
            result = extractor.extract(hsi)

        # Assert: snr_ratio is all zeros
        np.testing.assert_allclose(result["snr_ratio"], np.zeros(3))

    def test_validate_n_components_positive(self, small_hsi):
        """Test validation fails with non-positive n_components."""
        # Arrange
        extractor = MNFExtractor(n_components=0)

        # Act & Assert
        with pytest.raises(ValueError, match="n_components must be positive"):
            extractor.extract(small_hsi)

    def test_validate_negative_n_components(self, small_hsi):
        """Test validation fails with negative n_components."""
        # Arrange
        extractor = MNFExtractor(n_components=-5)

        # Act & Assert
        with pytest.raises(ValueError, match="n_components must be positive"):
            extractor.extract(small_hsi)

    def test_validate_minimum_spatial_size(self):
        """Test validation fails with 1x1 spatial dimensions."""
        # Arrange
        reflectance = np.random.rand(1, 1, 5).astype(np.float32)
        wavelengths = np.array([400, 500, 600, 700, 800], dtype=np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        extractor = MNFExtractor()

        # Act & Assert
        with pytest.raises(ValueError, match="at least 2x2"):
            extractor.extract(hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert MNFExtractor.feature_name() == "mnf"
