"""Tests for MPExtractor."""

import numpy as np
import pytest
from skimage.morphology import (
    closing,
    disk,
    dilation,
    erosion,
    opening,
    reconstruction as morph_reconstruction,
)

from hyppo.core import HSI
from hyppo.extractor.mp import MPExtractor


class TestMPExtractor:
    """Test cases for MPExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 8x8x5 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(8, 8, 5).astype(np.float32)
        wavelengths = np.array([450.0, 550.0, 650.0, 750.0, 850.0])
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression_standard(self, regression_hsi):
        """Regression test: standard morphology output must not change."""
        # Arrange
        extractor = MPExtractor(
            n_components=2, radii=[2], shapes=["disk"],
            use_reconstruction=False,
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: first row of features
        expected_row0 = np.array([
            [-0.11318319,  0.46192884,  0.46192884,
             -0.18764016, -0.11295304, 0.4068605],
            [-0.11318319,  0.1809916,   0.45977724,
             -0.18764016,  0.471562,    0.471562],
            [-0.11318319,  0.7830266,   0.7830266,
             -0.18764016, -0.18764016,  0.43718913],
            [-0.11318319,  0.16598114,  0.4098053,
             -0.18764016, -0.00796717,  0.43718913],
            [-0.50394624, -0.32658696,  0.1787864,
             -0.28407675, -0.10926706,  0.43718913],
            [-0.50394624, -0.1775336,  -0.1775336,
             -0.28407675, -0.28407675,  0.43718913],
            [-0.6796765,  -0.6796765,  -0.1775336,
             -0.28407675,  0.550995,    0.550995],
            [-0.7141775,  -0.54331136, -0.1775336,
             -0.28407675, -0.04963359,  0.54758525],
        ], dtype=np.float32)
        np.testing.assert_allclose(
            result["features"][0, :, :], expected_row0, rtol=1e-5
        )

    def test_regression_reconstruction(self, regression_hsi):
        """Regression test: reconstruction morphology output must not change."""
        # Arrange
        extractor = MPExtractor(
            n_components=2, radii=[2], shapes=["disk"],
            use_reconstruction=True,
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: first row of features
        expected_row0 = np.array([
            [-0.11318319,  0.46192884,  0.46192884,
             -0.18764016, -0.11295304, 0.29060462],
            [-0.11318319,  0.1809916,   0.1809916,
             -0.18764016,  0.471562,    0.471562],
            [-0.11318319,  0.7830266,   0.7830266,
             -0.18764016, -0.18764016,  0.29060462],
            [-0.11318319,  0.16598114,  0.16598114,
             -0.18764016, -0.00796717,  0.29060462],
            [-0.32658696, -0.32658696, -0.1775336,
             -0.18764016, -0.10926706,  0.29060462],
            [-0.1775336,  -0.1775336,  -0.1775336,
             -0.28407675, -0.28407675,  0.29060462],
            [-0.6796765,  -0.6796765,  -0.1775336,
             -0.18764016,  0.550995,    0.550995],
            [-0.54331136, -0.54331136, -0.1775336,
             -0.18764016, -0.04963359,  0.29060462],
        ], dtype=np.float32)
        np.testing.assert_allclose(
            result["features"][0, :, :], expected_row0, rtol=1e-5
        )

    def test_opening_matches_skimage(self):
        """Test _opening_by_reconstruction matches skimage operations."""
        # Arrange
        rng = np.random.RandomState(0)
        img = rng.rand(10, 10).astype(np.float64)
        se = disk(2)
        extractor = MPExtractor()

        # Act
        result = extractor._opening_by_reconstruction(img, se)

        # Assert: matches erosion + dilation reconstruction
        eroded = erosion(img, se)
        expected = morph_reconstruction(eroded, img, method="dilation")
        np.testing.assert_allclose(result, expected)

    def test_closing_matches_skimage(self):
        """Test _closing_by_reconstruction matches skimage operations."""
        # Arrange
        rng = np.random.RandomState(0)
        img = rng.rand(10, 10).astype(np.float64)
        se = disk(2)
        extractor = MPExtractor()

        # Act
        result = extractor._closing_by_reconstruction(img, se)

        # Assert: matches dilation + erosion reconstruction
        dilated = dilation(img, se)
        expected = morph_reconstruction(dilated, img, method="erosion")
        np.testing.assert_allclose(result, expected)

    def test_opening_anti_extensive_closing_extensive(self):
        """Test opening(f) <= f <= closing(f) for all pixels."""
        # Arrange
        rng = np.random.RandomState(7)
        img = rng.rand(12, 12).astype(np.float64)
        extractor = MPExtractor(radii=[2], shapes=["disk"])

        # Act
        profile = extractor._compute_morphological_profile(img, "disk")

        # Assert: [opening, original, closing]
        opening_vals = profile[:, :, 0]
        original_vals = profile[:, :, 1]
        closing_vals = profile[:, :, 2]
        assert np.all(opening_vals <= original_vals + 1e-10)
        assert np.all(original_vals <= closing_vals + 1e-10)

    def test_opening_anti_extensive_closing_extensive_reconstruction(self):
        """Same property holds for reconstruction-based operations."""
        # Arrange
        rng = np.random.RandomState(7)
        img = rng.rand(12, 12).astype(np.float64)
        extractor = MPExtractor(
            radii=[2], shapes=["disk"], use_reconstruction=True,
        )

        # Act
        profile = extractor._compute_morphological_profile(img, "disk")

        # Assert
        opening_vals = profile[:, :, 0]
        original_vals = profile[:, :, 1]
        closing_vals = profile[:, :, 2]
        assert np.all(opening_vals <= original_vals + 1e-10)
        assert np.all(original_vals <= closing_vals + 1e-10)

    def test_profile_ordering(self):
        """Test profile structure: [Open_n,...,Open_1, Original, Close_1,...,Close_n]."""
        # Arrange
        rng = np.random.RandomState(0)
        img = rng.rand(10, 10).astype(np.float64)
        extractor = MPExtractor(radii=[2, 4], shapes=["disk"])

        # Act
        profile = extractor._compute_morphological_profile(img, "disk")

        # Assert: shape is (H, W, 2*2+1) = (10, 10, 5)
        assert profile.shape == (10, 10, 5)

        # Center must be original image
        np.testing.assert_allclose(profile[:, :, 2], img, atol=1e-7)

        # Openings are anti-extensive and ordered by radius
        # open_r4 <= open_r2 <= original (larger SE removes more)
        assert np.all(profile[:, :, 0] <= profile[:, :, 1] + 1e-10)
        assert np.all(profile[:, :, 1] <= profile[:, :, 2] + 1e-10)

        # Closings are extensive and ordered by radius
        # original <= close_r2 <= close_r4
        assert np.all(profile[:, :, 2] <= profile[:, :, 3] + 1e-10)
        assert np.all(profile[:, :, 3] <= profile[:, :, 4] + 1e-10)

    def test_line_structuring_element(self):
        """Test line structuring element shape and extraction."""
        # Arrange
        extractor = MPExtractor()

        # Act
        elem = extractor._get_structuring_element("line", 3)

        # Assert: horizontal line of width 2*3+1=7, height 1
        assert elem.shape == (1, 7)
        assert np.all(elem == 1)

    def test_line_shape_extraction(self, small_hsi):
        """Test full extraction with line shape."""
        # Arrange
        extractor = MPExtractor(shapes=["line"], radii=[1])

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["shapes"] == ["line"]
        assert result["features"].ndim == 3

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = MPExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure and defaults
        expected_keys = [
            "features", "explained_variance_ratio", "n_components",
            "shapes", "radii", "n_features", "use_reconstruction",
        ]
        for key in expected_keys:
            assert key in result

        assert result["n_components"] == 3
        assert result["radii"] == [2, 4, 6, 8]
        assert result["shapes"] == ["disk", "square", "diamond"]
        assert result["use_reconstruction"] is False

        features = result["features"]
        assert features.shape[:2] == (small_hsi.height, small_hsi.width)
        assert features.ndim == 3

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom parameters."""
        # Arrange
        extractor = MPExtractor(
            n_components=2, radii=[1, 3], shapes=["disk"],
            use_reconstruction=True,
        )

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["n_components"] == 2
        assert result["radii"] == [1, 3]
        assert result["shapes"] == ["disk"]
        assert result["use_reconstruction"] is True

    @pytest.mark.parametrize("shape", ["disk", "square", "diamond"])
    def test_different_shapes(self, small_hsi, shape):
        """Test extraction with different structuring element shapes."""
        # Arrange
        extractor = MPExtractor(shapes=[shape])

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["shapes"] == [shape]
        assert result["features"].ndim == 3

    def test_get_structuring_element_disk(self):
        """Test disk structuring element generation."""
        # Arrange
        extractor = MPExtractor()

        # Act
        elem = extractor._get_structuring_element("disk", 3)

        # Assert
        assert elem.ndim == 2
        assert elem.shape[0] == elem.shape[1]

    def test_get_structuring_element_square(self):
        """Test square structuring element is (2r+1) x (2r+1)."""
        # Arrange
        extractor = MPExtractor()

        # Act
        elem = extractor._get_structuring_element("square", 3)

        # Assert
        assert elem.shape == (7, 7)
        assert np.all(elem == 1)

    def test_get_structuring_element_diamond(self):
        """Test diamond structuring element generation."""
        # Arrange
        extractor = MPExtractor()

        # Act
        elem = extractor._get_structuring_element("diamond", 3)

        # Assert
        assert elem.ndim == 2

    def test_get_structuring_element_invalid(self):
        """Test invalid structuring element raises error."""
        extractor = MPExtractor()
        with pytest.raises(ValueError, match="Unsupported shape"):
            extractor._get_structuring_element("invalid", 3)

    @pytest.mark.parametrize("radii", [[1], [1, 2], [2, 4, 6, 8]])
    def test_different_radii(self, small_hsi, radii):
        """Test extraction with different radii."""
        # Arrange
        extractor = MPExtractor(radii=radii)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["radii"] == sorted(radii)

    def test_n_features_calculation(self, small_hsi):
        """Test that n_features matches formula and actual shape."""
        # Arrange
        n_components = 2
        radii = [1, 2]
        shapes = ["disk"]
        extractor = MPExtractor(
            n_components=n_components, radii=radii, shapes=shapes,
        )

        # Act
        result = extractor.extract(small_hsi)

        # Assert: n_features = n_components * len(shapes) * (2*len(radii) + 1)
        expected = n_components * len(shapes) * (2 * len(radii) + 1)
        assert result["n_features"] == expected
        assert result["features"].shape[-1] == expected

    def test_use_reconstruction_produces_different_results(self, small_hsi):
        """Test that reconstruction mode differs from standard morphology."""
        # Arrange
        extractor_std = MPExtractor(
            shapes=["disk"], radii=[2], use_reconstruction=False,
        )
        extractor_rec = MPExtractor(
            shapes=["disk"], radii=[2], use_reconstruction=True,
        )

        # Act
        result_std = extractor_std.extract(small_hsi)
        result_rec = extractor_rec.extract(small_hsi)

        # Assert: same shape but different values
        assert result_std["features"].shape == result_rec["features"].shape
        assert not np.allclose(
            result_std["features"], result_rec["features"]
        )

    def test_pca_variance_explained(self, small_hsi):
        """Test that PCA variance ratios are valid."""
        # Arrange
        extractor = MPExtractor(n_components=2)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        variance_ratio = result["explained_variance_ratio"]
        assert len(variance_ratio) == 2
        assert np.all(variance_ratio >= 0)
        assert np.all(variance_ratio <= 1)

    def test_validate_invalid_radii_type(self, small_hsi):
        """Test validation fails with invalid radii type."""
        extractor = MPExtractor(radii="invalid")
        with pytest.raises((ValueError, TypeError)):
            extractor.extract(small_hsi)

    def test_validate_empty_radii(self, small_hsi):
        """Test validation fails with empty radii list."""
        extractor = MPExtractor(radii=[])
        with pytest.raises(ValueError, match="radii must be a non-empty"):
            extractor.extract(small_hsi)

    def test_validate_negative_radii(self, small_hsi):
        """Test validation fails with negative radii."""
        extractor = MPExtractor(radii=[-1, 2])
        with pytest.raises(ValueError, match="All radii must be positive"):
            extractor.extract(small_hsi)

    def test_validate_zero_radii(self, small_hsi):
        """Test validation fails with zero radii."""
        extractor = MPExtractor(radii=[0, 1])
        with pytest.raises(ValueError, match="All radii must be positive"):
            extractor.extract(small_hsi)

    def test_validate_invalid_shape(self, small_hsi):
        """Test validation fails with invalid shape."""
        extractor = MPExtractor(shapes=["invalid"])
        with pytest.raises(ValueError, match="Invalid shape"):
            extractor.extract(small_hsi)

    def test_validate_empty_shapes(self, small_hsi):
        """Test validation fails with empty shapes list."""
        extractor = MPExtractor(shapes=[])
        with pytest.raises(ValueError, match="shapes must be a non-empty"):
            extractor.extract(small_hsi)

    def test_validate_invalid_n_components(self, small_hsi):
        """Test validation fails with invalid n_components."""
        extractor = MPExtractor(n_components=0)
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_n_components_exceeds_bands(self, small_hsi):
        """Test validation fails when n_components exceeds spectral bands."""
        extractor = MPExtractor(n_components=100)
        with pytest.raises(ValueError, match="Number of spectral bands"):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert MPExtractor.feature_name() == "m_p"
