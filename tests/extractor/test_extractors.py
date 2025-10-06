"""
Tests for extractor implementations.

This module contains basic smoke tests for all concrete extractor classes
to verify they can successfully complete extraction without errors.
"""

import pytest
import numpy as np
from hyppo.extractor import (
    DWT1DExtractor,
    DWT2DExtractor,
    DWT3DExtractor,
    GaborExtractor,
    GeometricMomentExtractor,
    GLCMExtractor,
    ICAExtractor,
    LBPExtractor,
    LegendreMomentExtractor,
    MaxExtractor,
    MeanExtractor,
    MedianExtractor,
    MinExtractor,
    MNFExtractor,
    MPExtractor,
    NDVIExtractor,
    NDWIExtractor,
    PCAExtractor,
    PPExtractor,
    SAVIExtractor,
    StdExtractor,
    ZernikeMomentExtractor,
)


class TestDWT1DExtractor:
    """Test cases for DWT1DExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that DWT1D extraction completes successfully."""
        # Arrange: Create extractor
        extractor = DWT1DExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "wavelet",
            "mode",
            "levels",
            "coeffs_lengths",
            "n_features",
            "original_shape",
        }


class TestDWT2DExtractor:
    """Test cases for DWT2DExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that DWT2D extraction completes successfully."""
        # Arrange: Create extractor
        extractor = DWT2DExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "wavelet",
            "mode",
            "levels",
            "n_features",
            "original_shape",
        }


class TestDWT3DExtractor:
    """Test cases for DWT3DExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that DWT3D extraction completes successfully."""
        # Arrange: Create extractor
        extractor = DWT3DExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "wavelet",
            "mode",
            "levels",
            "n_features",
            "original_shape",
        }


class TestGaborExtractor:
    """Test cases for GaborExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that Gabor extraction completes successfully."""
        # Arrange: Create extractor
        extractor = GaborExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {"features"}


class TestGeometricMomentExtractor:
    """Test cases for GeometricMomentExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that GeometricMoment extraction completes successfully."""
        # Arrange: Create extractor
        extractor = GeometricMomentExtractor()

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
            "max_order",
        }


class TestGLCMExtractor:
    """Test cases for GLCMExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that GLCM extraction completes successfully."""
        # Arrange: Create extractor
        extractor = GLCMExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "bands_used",
            "distances",
            "angles",
            "properties",
            "levels_used",
            "window_sizes",
            "orientation_mode",
            "n_features_per_scale",
            "n_features_per_band",
            "total_features",
            "original_shape",
        }


class TestICAExtractor:
    """Test cases for ICAExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that ICA extraction completes successfully."""
        # Arrange: Create extractor
        extractor = ICAExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "components",
            "mixing_matrix",
            "mean",
            "n_components",
            "original_shape",
            "n_iter",
            "reconstruction_error",
        }


class TestLBPExtractor:
    """Test cases for LBPExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that LBP extraction completes successfully."""
        # Arrange: Create extractor
        extractor = LBPExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {
            "features",
            "bands_used",
            "radius",
            "n_points",
            "method",
            "original_shape",
            "n_features",
        }


class TestLegendreMomentExtractor:
    """Test cases for LegendreMomentExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that LegendreMoment extraction completes successfully."""
        # Arrange: Create extractor
        extractor = LegendreMomentExtractor()

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
            "max_order",
        }


class TestMaxExtractor:
    """Test cases for MaxExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that Max extraction completes successfully."""
        # Arrange: Create extractor
        extractor = MaxExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {"features"}


class TestMeanExtractor:
    """Test cases for MeanExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that Mean extraction completes successfully."""
        # Arrange: Create extractor
        extractor = MeanExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {"features"}


class TestMedianExtractor:
    """Test cases for MedianExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that Median extraction completes successfully."""
        # Arrange: Create extractor
        extractor = MedianExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {"features"}


class TestMinExtractor:
    """Test cases for MinExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that Min extraction completes successfully."""
        # Arrange: Create extractor
        extractor = MinExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {"features"}


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


class TestStdExtractor:
    """Test cases for StdExtractor."""

    def test_extract_basic(self, sample_hsi):
        """Test that Std extraction completes successfully."""
        # Arrange: Create extractor
        extractor = StdExtractor()

        # Act: Execute extraction
        result = extractor.extract(sample_hsi)

        # Assert: Verify required keys
        assert "features" in result
        assert isinstance(result["features"], np.ndarray)
        assert set(result.keys()) == {"features"}


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
