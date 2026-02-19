"""Tests for MNFExtractor."""

import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.extractor.mnf import MNFExtractor


class TestMNFExtractor:
    """Test cases for MNFExtractor."""

    def test_mnf_mathematical_properties(self):
        """Test MNF satisfies mathematical properties.

        Reference: Green et al. (1988) - A transformation for ordering
        multispectral data in terms of image quality.

        Properties verified:
        - SNR eigenvalues are sorted in descending order
        - Whitening matrix is symmetric
        - SNR ratios sum to <= 1
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
        assert "features" in result
        assert "n_features" in result
        assert "original_shape" in result
        assert "mean" in result
        assert "noise_eigenvalues" in result
        assert "snr_eigenvalues" in result
        assert "whitening_matrix" in result
        assert "projection_matrix" in result

        # Assert: Verify feature shape
        features = result["features"]
        assert features.shape[0] == small_hsi.height
        assert features.shape[1] == small_hsi.width
        assert features.ndim == 3

    def test_validate_n_components_positive(self, small_hsi):
        """Test validation fails with non-positive n_components."""
        # Arrange
        extractor = MNFExtractor(n_components=0)

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
