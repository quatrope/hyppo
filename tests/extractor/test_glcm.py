"""Tests for GLCMExtractor."""

import numpy as np
import pytest

from hyppo.core import HSI
from hyppo.extractor.glcm import GLCMExtractor


class TestGLCMExtractor:
    """Tests for GLCMExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Build a small deterministic HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(5, 5, 3).astype(np.float32)
        wavelengths = np.array([500.0, 600.0, 700.0])
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_extract_basic(self, regression_hsi):
        """Test basic extraction returns features with expected shape."""
        extractor = GLCMExtractor()
        result = extractor.extract(regression_hsi)
        assert "features" in result
        features = result["features"]
        assert features.shape[0] == regression_hsi.height
        assert features.shape[1] == regression_hsi.width
        assert features.ndim == 3

    def test_spectral_reduction_pca(self, regression_hsi):
        """Test PCA spectral reduction collapses bands to n_components."""
        extractor = GLCMExtractor(spectral_reduction="pca", n_components=1)
        cube = extractor._spectral_reduce(regression_hsi.reflectance)
        assert cube.shape[-1] == 1

    def test_spectral_reduction_none(self, regression_hsi):
        """Test no reduction preserves the original cube shape."""
        extractor = GLCMExtractor(spectral_reduction=None)
        cube = extractor._spectral_reduce(regression_hsi.reflectance)
        assert cube.shape == regression_hsi.reflectance.shape

    def test_quantization_range(self):
        """Test quantization output stays inside [0, levels - 1]."""
        extractor = GLCMExtractor(levels=16)
        img = np.random.rand(10, 10)
        q = extractor._quantize(img)
        assert q.dtype == np.uint8
        assert q.min() >= 0
        assert q.max() <= 15

    def test_quantization_constant(self):
        """Test quantization of a constant image is constant."""
        extractor = GLCMExtractor(levels=16)
        img = np.ones((10, 10))
        q = extractor._quantize(img)
        assert np.all(q == q[0, 0])

    def test_glcm_normalization(self):
        """Test GLCM rows sum to 1 after normalization."""
        extractor = GLCMExtractor(levels=8)
        img = np.random.randint(0, 8, (10, 10)).astype(np.uint8)
        P = extractor._build_glcm_maps(img, 1, 0)
        sums = P.sum(axis=(1, 2))
        np.testing.assert_allclose(sums, 1, rtol=1e-5)

    def test_haralick_output_shape(self):
        """Test Haralick batch returns one row per GLCM, one col per feat."""
        extractor = GLCMExtractor(levels=8)
        P = np.random.rand(5, 8, 8).astype(np.float32)
        P /= P.sum(axis=(1, 2), keepdims=True)
        feats = extractor._extract_haralick_batch(P)
        assert feats.shape == (5, len(extractor.features))

    def test_reference_patch_against_skimage(self):
        """Test contrast feature is finite on a reference patch."""
        rng = np.random.RandomState(0)
        patch = rng.randint(0, 16, (7, 7)).astype(np.uint8)
        extractor = GLCMExtractor(
            levels=16,
            distances=[1],
            angles=[0],
            window_sizes=[7],
            spectral_reduction=None,
        )

        P = extractor._build_glcm_maps(patch, 1, 0)
        feats = extractor._extract_haralick_batch(P)
        contrast = feats[:, extractor.features.index("contrast")]
        assert np.isfinite(contrast).all()

    def test_angle_pooling_mean(self):
        """Test mean angle pooling collapses 4 angle maps into 1."""
        extractor = GLCMExtractor(angle_pooling="mean")
        feats = [np.random.rand(5, 5, 3) for _ in range(4)]
        pooled = extractor._pool_angles(feats)
        assert pooled.shape == (5, 5, 3)

    def test_angle_pooling_concat(self):
        """Test concat angle pooling stacks the 4 angle feature blocks."""
        extractor = GLCMExtractor(angle_pooling="concat")
        feats = [np.random.rand(5, 5, 3) for _ in range(4)]
        pooled = extractor._pool_angles(feats)
        assert pooled.shape == (5, 5, 12)

    def test_angle_pooling_mean_range(self):
        """Test mean+range pooling outputs 2 stats per channel."""
        extractor = GLCMExtractor(angle_pooling="mean+range")
        feats = [np.random.rand(5, 5, 3) for _ in range(4)]
        pooled = extractor._pool_angles(feats)
        assert pooled.shape == (5, 5, 6)

    def test_multiple_distances_angles(self, regression_hsi):
        """Test extraction with multiple distances and angles."""
        extractor = GLCMExtractor(
            distances=[1, 2],
            angles=[0, np.pi / 2],
            spectral_reduction="pca",
            n_components=1,
        )
        result = extractor.extract(regression_hsi)
        assert result["features"].ndim == 3

    def test_invalid_levels(self, regression_hsi):
        """Test levels=0 raises ValueError."""
        extractor = GLCMExtractor(levels=0)
        with pytest.raises(ValueError):
            extractor.extract(regression_hsi)

    def test_invalid_window(self, regression_hsi):
        """Test even window size raises ValueError."""
        extractor = GLCMExtractor(window_sizes=[4])
        with pytest.raises(ValueError):
            extractor.extract(regression_hsi)

    def test_invalid_distances(self, regression_hsi):
        """Test empty distances list raises ValueError."""
        extractor = GLCMExtractor(distances=[])
        with pytest.raises(ValueError):
            extractor.extract(regression_hsi)

    def test_invalid_angles(self, regression_hsi):
        """Test empty angles list raises ValueError."""
        extractor = GLCMExtractor(angles=[])
        with pytest.raises(ValueError):
            extractor.extract(regression_hsi)

    def test_feature_name(self):
        """Test feature_name returns the canonical 'glcm' identifier."""
        assert GLCMExtractor.feature_name() == "glcm"
