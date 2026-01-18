"""
Tests for extractor implementations.

This module contains basic smoke tests for all concrete extractor classes
to verify they can successfully complete extraction without errors.
"""

import numpy as np

from hyppo.extractor import (
    MNFExtractor,
    MPExtractor,
    NDVIExtractor,
    NDWIExtractor,
    PCAExtractor,
    PPExtractor,
    SAVIExtractor,
    ZernikeMomentExtractor,
)


class TestMNFExtractor:
    """Test cases for MNFExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that MNF extraction completes successfully."""
        # Arrange: Create extractor
        extractor = MNFExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "explained_variance_ratio",
            "explained_variance",
            "components",
            "mean",
            "n_components",
            "original_shape",
            "cumulative_variance_ratio",
            "noise_eigenvalues",
            "whitening_matrix",
        }


class TestMPExtractor:
    """Test cases for MPExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that MP extraction completes successfully."""
        # Arrange: Create extractor
        extractor = MPExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "bands_used",
            "radii",
            "structuring_element",
            "n_features",
            "original_shape",
        }


class TestNDVIExtractor:
    """Test cases for NDVIExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that NDVI extraction completes successfully."""
        # Arrange: Create extractor
        extractor = NDVIExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "red_idx",
            "nir_idx",
            "wavelength_used",
            "original_shape",
        }


class TestNDWIExtractor:
    """Test cases for NDWIExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that NDWI extraction completes successfully."""
        # Arrange: Create extractor
        extractor = NDWIExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "green_idx",
            "nir_idx",
            "wavelength_used",
            "original_shape",
        }


class TestPCAExtractor:
    """Test cases for PCAExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that PCA extraction completes successfully."""
        # Arrange: Create extractor
        extractor = PCAExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "explained_variance_ratio",
            "explained_variance",
            "components",
            "mean",
            "n_components",
            "original_shape",
            "cumulative_variance_ratio",
        }


class TestPPExtractor:
    """Test cases for PPExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that PP extraction completes successfully."""
        # Arrange: Create extractor
        extractor = PPExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "n_features",
            "original_shape",
            "projection_vectors",
            "divergence_scores",
            "pca_components_used",
            "pca_model",
            "selected_pixel_indices",
            "valid_pixel_mask",
        }


class TestSAVIExtractor:
    """Test cases for SAVIExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that SAVI extraction completes successfully."""
        # Arrange: Create extractor
        extractor = SAVIExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "red_idx",
            "nir_idx",
            "wavelength_used",
            "brightness_correction",
            "original_shape",
        }


class TestZernikeMomentExtractor:
    """Test cases for ZernikeMomentExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that ZernikeMoment extraction completes successfully."""
        # Arrange: Create extractor
        extractor = ZernikeMomentExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "explained_variance_ratio",
            "n_components",
            "window_sizes",
            "degree",
        }
