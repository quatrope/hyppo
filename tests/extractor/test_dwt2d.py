"""Tests for DWT2DExtractor."""

import numpy as np
import pytest
import pywt

from hyppo.core import HSI
from hyppo.extractor.dwt2d import DWT2DExtractor


class TestDWT2DExtractor:
    """Test cases for DWT2DExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 4x4x5 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(4, 4, 5).astype(np.float32)
        wavelengths = np.linspace(400, 800, 5).astype(np.float32)
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression(self, regression_hsi):
        """Regression test: DWT2D output must not change."""
        # Arrange
        extractor = DWT2DExtractor(wavelet="haar", levels=1)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        expected_pixel00 = np.array(
            [
                0.9637817,
                -0.4332471,
                0.02261129,
                0.19593434,
                0.6739828,
                0.33481514,
                0.41622537,
                0.47640532,
                1.2022746,
                0.39589548,
                -0.17813599,
                0.04395375,
                1.0792749,
                0.12049852,
                -0.11425456,
                0.1117981,
                0.6833058,
                0.18078543,
                -0.07121718,
                -0.48083675,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            result["features"][0, 0, :], expected_pixel00, rtol=1e-5
        )

    def test_reference_pywt(self, regression_hsi):
        """Test DWT2D matches pywt.swt2 directly for first band."""
        # Arrange
        extractor = DWT2DExtractor(wavelet="haar", levels=1)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: compare LL subband (first 4x4 block) with pywt
        band0 = regression_hsi.reflectance[:, :, 0]
        coeffs = pywt.swt2(band0, "haar", level=1)
        coeffs = list(reversed(coeffs))
        cA, (cH, cV, cD) = coeffs[0]

        h, w = regression_hsi.height, regression_hsi.width
        np.testing.assert_allclose(
            result["features"][:, :, 0], cA[:h, :w], rtol=1e-5
        )
        np.testing.assert_allclose(
            result["features"][:, :, 1], cH[:h, :w], rtol=1e-5
        )

    def test_feature_count_formula(self, regression_hsi):
        """Test n_features = bands * 4 * levels."""
        # Arrange
        levels = 2
        extractor = DWT2DExtractor(wavelet="haar", levels=levels)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        bands = regression_hsi.n_bands
        expected = bands * 4 * levels
        assert result["n_features"] == expected

    def test_padding_branch(self, small_hsi):
        """Test extraction with odd-dimension image (requires padding)."""
        # Arrange: small_hsi is 3x3, divisor=2, needs padding
        extractor = DWT2DExtractor(wavelet="haar", levels=1)

        # Act
        result = extractor.extract(small_hsi)

        # Assert: output preserves original spatial size
        assert result["features"].shape[:2] == (
            small_hsi.height,
            small_hsi.width,
        )

    def test_no_padding_branch(self, regression_hsi):
        """Test extraction with even-dimension image (no padding needed)."""
        # Arrange: 4x4 with levels=1, divisor=2, 4%2=0, no padding
        extractor = DWT2DExtractor(wavelet="haar", levels=1)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        assert result["features"].shape[:2] == (4, 4)

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = DWT2DExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        expected_keys = [
            "features",
            "wavelet",
            "levels",
            "n_features",
            "original_shape",
        ]
        for key in expected_keys:
            assert key in result

        assert result["wavelet"] == "haar"
        assert result["levels"] == 1
        assert result["original_shape"] == small_hsi.shape

        features = result["features"]
        assert features.shape[:2] == (small_hsi.height, small_hsi.width)
        assert features.ndim == 3

    def test_spatial_resolution_preserved(self, small_hsi):
        """Test output features maintain original spatial resolution."""
        # Arrange
        extractor = DWT2DExtractor(levels=2)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["features"].shape[0] == small_hsi.height
        assert result["features"].shape[1] == small_hsi.width

    @pytest.mark.parametrize("wavelet", ["haar", "db4", "sym5"])
    def test_different_wavelets(self, small_hsi, wavelet):
        """Test extraction with different wavelets."""
        # Arrange
        extractor = DWT2DExtractor(wavelet=wavelet)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["wavelet"] == wavelet

    @pytest.mark.parametrize("levels", [1, 2])
    def test_different_levels(self, small_hsi, levels):
        """Test extraction with different decomposition levels."""
        # Arrange
        extractor = DWT2DExtractor(levels=levels)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["levels"] == levels

    def test_validate_invalid_wavelet(self, small_hsi):
        """Test validation fails with invalid wavelet name."""
        extractor = DWT2DExtractor(wavelet="invalid_wavelet_name")
        with pytest.raises(ValueError, match="Wavelet .* not available"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("levels", [-1, 0, 2.5])
    def test_validate_invalid_levels(self, small_hsi, levels):
        """Test validation fails with invalid decomposition levels."""
        extractor = DWT2DExtractor(levels=levels)  # type: ignore
        with pytest.raises(
            ValueError, match="levels must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert DWT2DExtractor.feature_name() == "dwt2d"
