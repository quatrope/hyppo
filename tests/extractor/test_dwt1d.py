"""Tests for DWT1DExtractor."""

import numpy as np
import pytest
import pywt

from hyppo.core import HSI
from hyppo.extractor.dwt1d import DWT1DExtractor


class TestDWT1DExtractor:
    """Test cases for DWT1DExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 5x5x8 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(5, 5, 8).astype(np.float32)
        wavelengths = np.linspace(400, 900, 8).astype(np.float32)
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression(self, regression_hsi):
        """Regression test: DWT1D output must not change."""
        # Arrange
        extractor = DWT1DExtractor(wavelet="haar", levels=2)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        expected_row0 = np.array(
            [
                [
                    1.3279533e00,
                    6.1813641e-01,
                    -2.6990175e-03,
                    -3.0612329e-01,
                    -4.0741667e-01,
                    9.4282389e-02,
                    1.7061830e-05,
                    -5.7140774e-01,
                ],
                [
                    1.1498410e00,
                    7.0500565e-01,
                    1.5934661e-01,
                    3.3977616e-01,
                    -7.5630486e-02,
                    -6.7127436e-01,
                    4.3847942e-01,
                    -1.1169016e-03,
                ],
                [
                    7.7608645e-01,
                    7.0492661e-01,
                    5.2912265e-02,
                    4.6420127e-02,
                    -1.5592706e-01,
                    9.9501163e-02,
                    3.3400828e-01,
                    -5.2479491e-02,
                ],
                [
                    9.7757703e-01,
                    7.0846689e-01,
                    2.6366884e-01,
                    -6.9601983e-02,
                    -2.3271310e-01,
                    -2.2242795e-01,
                    3.8605493e-01,
                    3.0902031e-01,
                ],
                [
                    1.3939831e00,
                    7.6333565e-01,
                    -3.8004607e-01,
                    -3.6104983e-01,
                    -6.2496501e-01,
                    1.1118168e-01,
                    1.4632985e-01,
                    1.7259100e-01,
                ],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            result["features"][0, :, :], expected_row0, rtol=1e-5
        )

    def test_reference_pywt(self, regression_hsi):
        """Test DWT1D matches pywt.wavedec directly."""
        # Arrange
        extractor = DWT1DExtractor(wavelet="haar", levels=2)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: compare with manual pywt for first pixel
        pixel = regression_hsi.reflectance[0, 0, :]
        coeffs = pywt.wavedec(pixel, "haar", mode="symmetric", level=2)
        expected = np.concatenate(coeffs)
        np.testing.assert_allclose(
            result["features"][0, 0, :], expected, rtol=1e-5
        )

    def test_levels_none_auto_detection(self, regression_hsi):
        """Test levels=None uses maximum decomposition level."""
        # Arrange
        extractor = DWT1DExtractor(wavelet="haar", levels=None)

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        max_level = pywt.dwt_max_level(8, "haar")
        assert result["features"].ndim == 3
        assert result["levels"] is None

        # Verify it uses max_level by comparing with explicit max_level
        extractor_explicit = DWT1DExtractor(
            wavelet="haar",
            levels=max_level,
        )
        result_explicit = extractor_explicit.extract(regression_hsi)
        np.testing.assert_allclose(
            result["features"], result_explicit["features"]
        )

    def test_validate_levels_exceeds_max(self, regression_hsi):
        """Test validation fails when levels exceeds maximum."""
        # Arrange: 8 bands with haar → max_level=3, so 4 should fail
        extractor = DWT1DExtractor(wavelet="haar", levels=4)

        # Act & Assert
        with pytest.raises(ValueError, match="exceeds maximum level"):
            extractor.extract(regression_hsi)

    def test_extract_basic_with_defaults(self, large_spectral_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = DWT1DExtractor()

        # Act
        result = extractor.extract(large_spectral_hsi)

        # Assert
        expected_keys = [
            "features",
            "wavelet",
            "mode",
            "levels",
            "n_features",
            "original_shape",
        ]
        for key in expected_keys:
            assert key in result

        assert result["wavelet"] == "db4"
        assert result["mode"] == "symmetric"
        assert result["levels"] == 3
        assert result["original_shape"] == large_spectral_hsi.shape

        features = result["features"]
        assert features.shape[:2] == (
            large_spectral_hsi.height,
            large_spectral_hsi.width,
        )
        assert features.ndim == 3

    @pytest.mark.parametrize(
        "wavelet,mode",
        [
            ("haar", "symmetric"),
            ("haar", "periodic"),
            ("db4", "symmetric"),
            ("db4", "zero"),
            ("sym5", "constant"),
        ],
    )
    def test_extract_wavelet_mode_combinations(
        self, large_spectral_hsi, wavelet, mode
    ):
        """Test extraction with cross-product of wavelets and modes."""
        # Arrange
        extractor = DWT1DExtractor(wavelet=wavelet, mode=mode)

        # Act
        result = extractor.extract(large_spectral_hsi)

        # Assert
        assert result["wavelet"] == wavelet
        assert result["mode"] == mode

    def test_validate_invalid_wavelet(self, small_hsi):
        """Test validation fails with invalid wavelet name."""
        extractor = DWT1DExtractor(wavelet="invalid_wavelet_name")
        with pytest.raises(ValueError, match="Wavelet .* not available"):
            extractor.extract(small_hsi)

    def test_validate_invalid_mode(self, small_hsi):
        """Test validation fails with invalid signal extension mode."""
        extractor = DWT1DExtractor(mode="invalid_mode")
        with pytest.raises(ValueError, match="Mode .* not available"):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("levels", [-1, 0, 2.5])
    def test_validate_invalid_levels(self, small_hsi, levels):
        """Test validation fails with invalid decomposition levels."""
        extractor = DWT1DExtractor(levels=levels)  # type: ignore
        with pytest.raises(
            ValueError, match="levels must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert DWT1DExtractor.feature_name() == "dwt1d"
