"""Tests for DWT3DExtractor."""

import numpy as np
import pytest
import pywt

from hyppo.core import HSI
from hyppo.extractor.dwt3d import DWT3DExtractor


class TestDWT3DExtractor:
    """Test cases for DWT3DExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 4x4x5 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(4, 4, 5).astype(np.float32)
        wavelengths = np.linspace(400, 800, 5).astype(np.float32)
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression(self, regression_hsi):
        """Regression test: DWT3D output must not change."""
        # Arrange
        extractor = DWT3DExtractor(wavelet="haar", levels=1)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        expected_pixel00 = np.array(
            [
                1.1580744,
                1.3267143,
                1.613299,
                1.2463326,
                1.2463326,
                0.20491877,
                -0.37355867,
                0.08697391,
                0.2799924,
                -0.2799924,
                0.31030437,
                0.16835462,
                -0.20675135,
                -0.13114832,
                -0.13114832,
                -0.2783272,
                0.42027694,
                -0.04517099,
                -0.03043203,
                0.03043203,
                -0.06960191,
                0.51669043,
                0.36514568,
                0.21303992,
                0.21303992,
                -0.543102,
                -0.04319032,
                0.19473505,
                -0.04262929,
                0.04262929,
                0.47541592,
                0.3679494,
                0.11013319,
                -0.26094973,
                -0.26094973,
                -0.19832292,
                0.30578944,
                -0.0479732,
                0.41905612,
                -0.41905612,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            result["features"][0, 0, :], expected_pixel00, rtol=1e-5
        )

    def test_reference_pywt(self, regression_hsi):
        """Test DWT3D matches pywt.swtn directly."""
        # Arrange
        extractor = DWT3DExtractor(wavelet="haar", levels=1)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: compare aaa subband with pywt
        h, w, b = regression_hsi.shape
        cube = regression_hsi.reflectance

        # Need padding for spectral dim: 5 bands → pad to 6
        pad_b = (2 - 5 % 2) % 2  # = 1
        cube_padded = np.pad(
            cube, ((0, 0), (0, 0), (0, pad_b)), mode="reflect"
        )

        coeffs = pywt.swtn(
            cube_padded,
            "haar",
            level=1,
            start_level=0,
            axes=(0, 1, 2),
        )
        coeffs = list(reversed(coeffs))
        aaa = coeffs[0]["aaa"][:h, :w, :b]

        # First 5 features should be the 'aaa' subband
        np.testing.assert_allclose(
            result["features"][:, :, :b], aaa, rtol=1e-5
        )

    def test_feature_count_formula(self, regression_hsi):
        """Test n_features = bands * 8 * levels."""
        # Arrange
        extractor = DWT3DExtractor(wavelet="haar", levels=1)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: 8 subbands per level, bands features per subband
        bands = regression_hsi.n_bands
        expected = bands * 8 * 1
        assert result["n_features"] == expected

    def test_padding_branch(self, small_hsi):
        """Test extraction with odd-dimension image (requires padding)."""
        # Arrange: small_hsi is 3x3x5, all odd → needs padding
        extractor = DWT3DExtractor(wavelet="haar", levels=1)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["features"].shape[:2] == (
            small_hsi.height,
            small_hsi.width,
        )

    def test_no_padding_branch(self):
        """Test extraction with even-dimension image (no padding needed)."""
        # Arrange: 4x4x4, all even, divisor=2 → no padding
        rng = np.random.RandomState(42)
        reflectance = rng.rand(4, 4, 4).astype(np.float32)
        wavelengths = np.linspace(400, 700, 4).astype(np.float32)
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)
        extractor = DWT3DExtractor(wavelet="haar", levels=1)

        # Act
        result = extractor.extract(hsi)

        # Assert
        assert result["features"].shape[:2] == (4, 4)

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = DWT3DExtractor()

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

    @pytest.mark.parametrize("wavelet", ["haar", "db4", "sym5"])
    def test_different_wavelets(self, small_hsi, wavelet):
        """Test extraction with different wavelets."""
        # Arrange
        extractor = DWT3DExtractor(wavelet=wavelet)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["wavelet"] == wavelet

    @pytest.mark.parametrize("levels", [1, 2])
    def test_different_levels(self, small_hsi, levels):
        """Test extraction with different decomposition levels."""
        # Arrange
        extractor = DWT3DExtractor(levels=levels)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["levels"] == levels

    def test_validate_invalid_wavelet(self, small_hsi):
        """Test validation fails with invalid wavelet name."""
        extractor = DWT3DExtractor(wavelet="invalid_wavelet_name")
        with pytest.raises(ValueError, match="Wavelet .* not available"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("levels", [-1, 0, 2.5])
    def test_validate_invalid_levels(self, small_hsi, levels):
        """Test validation fails with invalid decomposition levels."""
        extractor = DWT3DExtractor(levels=levels)  # type: ignore
        with pytest.raises(
            ValueError, match="levels must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert DWT3DExtractor.feature_name() == "dwt3d"
