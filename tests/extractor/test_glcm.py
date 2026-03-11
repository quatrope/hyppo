"""Tests for GLCMExtractor."""

from unittest.mock import patch as mock_patch

import numpy as np
import pytest
from skimage.feature import graycomatrix, graycoprops

from hyppo.core import HSI
from hyppo.extractor.glcm import GLCMExtractor


class TestGLCMExtractor:
    """Test cases for GLCMExtractor."""

    @pytest.fixture
    def regression_hsi(self):
        """Deterministic 5x5x3 HSI for regression tests."""
        rng = np.random.RandomState(42)
        reflectance = rng.rand(5, 5, 3).astype(np.float32)
        wavelengths = np.array([500.0, 600.0, 700.0])
        return HSI(reflectance=reflectance, wavelengths=wavelengths)

    def test_regression_separate_mode(self, regression_hsi):
        """Regression test: separate mode output must not change."""
        # Arrange
        extractor = GLCMExtractor(
            bands=[0],
            properties=["contrast"],
            angles=[0],
            distances=[1],
            window_sizes=[3],
            levels=16,
            orientation_mode="separate",
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        expected = np.array(
            [
                [16.0, 26.833334, 40.833332, 23.0, 2.0],
                [19.0, 42.333332, 52.833332, 26.833334, 13.666667],
                [15.0, 28.333334, 24.833334, 18.333334, 28.666666],
                [9.666667, 24.5, 49.166668, 81.166664, 103.333336],
                [2.6666667, 5.8333335, 34.0, 83.333336, 107.666664],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            result["features"][:, :, 0], expected, rtol=1e-5
        )

    def test_regression_look_direction_mode(self, regression_hsi):
        """Regression test: look_direction mode output must not change."""
        # Arrange
        extractor = GLCMExtractor(
            bands=[0],
            properties=["contrast"],
            distances=[1],
            window_sizes=[3],
            levels=16,
            orientation_mode="look_direction",
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert: first row, 3 direction features
        expected_row0 = np.array(
            [
                [16.0, 9.0, 25.0],
                [26.833334, 7.3333335, 33.75],
                [40.833332, 9.666667, 46.25],
                [23.0, 15.0, 36.25],
                [2.0, 19.0, 22.5],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            result["features"][0, :, :], expected_row0, rtol=1e-5
        )

    def test_regression_average_mode(self, regression_hsi):
        """Regression test: average mode output must not change."""
        # Arrange
        extractor = GLCMExtractor(
            bands=[0],
            properties=["contrast"],
            distances=[1],
            window_sizes=[3],
            levels=16,
            orientation_mode="average",
        )

        # Act
        result = extractor.extract(regression_hsi)

        # Assert
        expected = np.array(
            [
                [18.75, 25.416666, 35.75, 27.625, 16.5],
                [38.625, 43.625, 39.3125, 23.416666, 18.083334],
                [35.416668, 36.979168, 28.145834, 22.125, 26.875],
                [13.625, 23.75, 34.708332, 56.791668, 73.291664],
                [12.75, 24.291666, 36.125, 70.291664, 95.75],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            result["features"][:, :, 0], expected, rtol=1e-5
        )

    def test_reference_uniform_image(self):
        """Test GLCM on uniform image matches skimage directly.

        A uniform image produces a diagonal GLCM, so contrast and
        dissimilarity should be 0.
        """
        # Arrange: Create uniform patch and compute reference GLCM
        patch = np.full((7, 7), 10, dtype=np.uint8)
        levels = 32
        ref_glcm = graycomatrix(
            patch, distances=[1], angles=[0], levels=levels, symmetric=True
        )
        ref_contrast = graycoprops(ref_glcm, "contrast")[0, 0]
        ref_dissimilarity = graycoprops(ref_glcm, "dissimilarity")[0, 0]

        # Act: Compute via extractor
        extractor = GLCMExtractor(
            properties=["contrast", "dissimilarity"],
            levels=levels,
            angles=[0],
            window_sizes=[7],
        )
        patches = patch.reshape(1, 7, 7)
        features = extractor._compute_glcm_features_optimized(patches, levels)

        # Assert: Uniform image → contrast=0, dissimilarity=0
        assert ref_contrast == 0.0
        assert ref_dissimilarity == 0.0
        np.testing.assert_allclose(features[0], [0.0, 0.0], atol=1e-7)

    def test_reference_single_patch(self):
        """Test GLCM features match skimage graycoprops on a known patch."""
        # Arrange: Create patch with texture
        rng = np.random.RandomState(42)
        patch = rng.randint(0, 16, size=(7, 7), dtype=np.uint8)
        levels = 16
        angles = [0, np.pi / 2]
        properties = ["contrast", "correlation"]

        ref_glcm = graycomatrix(
            patch,
            distances=[1],
            angles=angles,
            levels=levels,
            symmetric=True,
        )
        expected = []
        for prop in properties:
            vals = graycoprops(ref_glcm, prop)[0, :]
            expected.extend(vals)
        expected = np.array(expected, dtype=np.float32)

        # Act: Compute via extractor
        extractor = GLCMExtractor(
            properties=properties,
            levels=levels,
            angles=angles,
            window_sizes=[7],
        )
        patches = patch.reshape(1, 7, 7)
        features = extractor._compute_glcm_features_optimized(patches, levels)

        # Assert: Match skimage reference
        np.testing.assert_allclose(features[0], expected, rtol=1e-5)

    def test_low_variation_patch_skipped(self):
        """Test that patches with variation < 2 produce all-zero features."""
        # Arrange: Create patch where max - min < 2
        patch = np.ones((7, 7), dtype=np.uint8) * 5
        patch[0, 0] = 6
        extractor = GLCMExtractor(
            properties=["contrast"], angles=[0], levels=32
        )

        # Act
        features = extractor._compute_glcm_features_optimized(
            patch.reshape(1, 7, 7), 32
        )

        # Assert: All zeros because variation is only 1
        assert np.all(features == 0.0)

    def test_look_direction_averages_diagonals(self):
        """Test look_direction mode averages 45 and 135 degree values."""
        # Arrange: Create textured patch
        rng = np.random.RandomState(42)
        patch = rng.randint(0, 16, size=(7, 7), dtype=np.uint8)
        levels = 16
        properties = ["contrast"]

        angles_look = [0, np.pi / 2, np.pi / 4, 3 * np.pi / 4]
        ref_glcm = graycomatrix(
            patch,
            distances=[1],
            angles=angles_look,
            levels=levels,
            symmetric=True,
        )
        ref_vals = graycoprops(ref_glcm, "contrast")[0, :]
        expected = [ref_vals[0], ref_vals[1], (ref_vals[2] + ref_vals[3]) / 2]

        # Act
        extractor = GLCMExtractor(
            properties=properties,
            levels=levels,
            orientation_mode="look_direction",
            window_sizes=[7],
        )
        features = extractor._compute_glcm_features_optimized(
            patch.reshape(1, 7, 7), levels
        )

        # Assert: 3 features: 0°, 90°, avg(45°,135°)
        np.testing.assert_allclose(features[0], expected, rtol=1e-5)

    def test_average_mode_averages_all_angles(self):
        """Test average mode returns mean across all orientations."""
        # Arrange: Create textured patch
        rng = np.random.RandomState(42)
        patch = rng.randint(0, 16, size=(7, 7), dtype=np.uint8)
        levels = 16
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        properties = ["contrast"]

        ref_glcm = graycomatrix(
            patch,
            distances=[1],
            angles=angles,
            levels=levels,
            symmetric=True,
        )
        ref_vals = graycoprops(ref_glcm, "contrast")[0, :]
        expected_avg = np.mean(ref_vals)

        # Act
        extractor = GLCMExtractor(
            properties=properties,
            levels=levels,
            orientation_mode="average",
            window_sizes=[7],
        )
        features = extractor._compute_glcm_features_optimized(
            patch.reshape(1, 7, 7), levels
        )

        # Assert
        np.testing.assert_allclose(features[0, 0], expected_avg, rtol=1e-5)

    def test_graycoprops_valueerror_fallback_separate(self):
        """Test fallback to zeros when graycoprops raises ValueError."""
        # Arrange
        rng = np.random.RandomState(42)
        patch = rng.randint(0, 16, size=(7, 7), dtype=np.uint8)
        extractor = GLCMExtractor(
            properties=["contrast"],
            angles=[0],
            levels=16,
            orientation_mode="separate",
        )

        # Act: Mock graycoprops to raise ValueError
        with mock_patch(
            "hyppo.extractor.glcm.graycoprops",
            side_effect=ValueError("mock"),
        ):
            features = extractor._compute_glcm_features_optimized(
                patch.reshape(1, 7, 7), 16
            )

        # Assert: Features filled with zeros
        assert np.all(features == 0.0)

    def test_graycoprops_valueerror_fallback_look_direction(self):
        """Test fallback to zeros in look_direction mode."""
        # Arrange
        rng = np.random.RandomState(42)
        patch = rng.randint(0, 16, size=(7, 7), dtype=np.uint8)
        extractor = GLCMExtractor(
            properties=["contrast"],
            levels=16,
            orientation_mode="look_direction",
        )

        # Act
        with mock_patch(
            "hyppo.extractor.glcm.graycoprops",
            side_effect=ValueError("mock"),
        ):
            features = extractor._compute_glcm_features_optimized(
                patch.reshape(1, 7, 7), 16
            )

        # Assert
        assert np.all(features == 0.0)

    def test_graycoprops_valueerror_fallback_average(self):
        """Test fallback to zeros in average mode."""
        # Arrange
        rng = np.random.RandomState(42)
        patch = rng.randint(0, 16, size=(7, 7), dtype=np.uint8)
        extractor = GLCMExtractor(
            properties=["contrast"],
            levels=16,
            orientation_mode="average",
        )

        # Act
        with mock_patch(
            "hyppo.extractor.glcm.graycoprops",
            side_effect=ValueError("mock"),
        ):
            features = extractor._compute_glcm_features_optimized(
                patch.reshape(1, 7, 7), 16
            )

        # Assert
        assert np.all(features == 0.0)

    @pytest.mark.parametrize(
        "orientation_mode", ["separate", "look_direction", "average"]
    )
    def test_graycomatrix_valueerror_fallback(self, orientation_mode):
        """Test fallback when graycomatrix raises ValueError."""
        # Arrange
        rng = np.random.RandomState(42)
        patch = rng.randint(0, 16, size=(7, 7), dtype=np.uint8)
        extractor = GLCMExtractor(
            properties=["contrast"],
            levels=16,
            orientation_mode=orientation_mode,
        )

        # Act
        with mock_patch(
            "hyppo.extractor.glcm.graycomatrix",
            side_effect=ValueError("mock"),
        ):
            features = extractor._compute_glcm_features_optimized(
                patch.reshape(1, 7, 7), 16
            )

        # Assert
        assert np.all(features == 0.0)

    @pytest.mark.parametrize(
        "mode,n_angles,expected_per_dist",
        [
            ("separate", 4, 4),
            ("look_direction", 4, 3),
            ("average", 4, 1),
        ],
    )
    def test_feature_count_per_orientation_mode(
        self, mode, n_angles, expected_per_dist
    ):
        """Test feature count matches orientation mode formula."""
        # Arrange
        properties = ["contrast", "correlation"]
        distances = [1, 2]
        window_sizes = [3]
        bands = [0]
        n_props = len(properties)
        n_dists = len(distances)

        wavelengths = np.linspace(400, 900, 5)
        reflectance = (
            np.random.RandomState(0).rand(10, 10, 5).astype(np.float32)
        )
        hsi = HSI(reflectance=reflectance, wavelengths=wavelengths)

        extractor = GLCMExtractor(
            bands=bands,
            distances=distances,
            properties=properties,
            window_sizes=window_sizes,
            orientation_mode=mode,
        )

        # Act
        result = extractor.extract(hsi)

        # Assert
        expected_per_scale = n_dists * expected_per_dist * n_props
        expected_total = expected_per_scale * len(window_sizes) * len(bands)
        assert result["n_features_per_scale"] == expected_per_scale
        assert result["total_features"] == expected_total
        assert result["features"].shape[-1] == expected_total

    def test_extract_basic_with_defaults(self, small_hsi):
        """Test extraction with default parameters."""
        # Arrange
        extractor = GLCMExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert: Verify output structure
        expected_keys = [
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
        ]
        for key in expected_keys:
            assert key in result

        # Assert: Verify defaults
        assert result["distances"] == [1]
        assert result["properties"] == [
            "contrast",
            "entropy",
            "correlation",
            "dissimilarity",
        ]
        assert result["window_sizes"] == [7]
        assert result["orientation_mode"] == "separate"

        # Assert: Verify feature shape
        features = result["features"]
        assert features.shape[:2] == (small_hsi.height, small_hsi.width)
        assert features.ndim == 3

    def test_extract_with_custom_parameters(self, small_hsi):
        """Test extraction with custom parameters."""
        # Arrange
        bands = [0, 1]
        distances = [1, 2]
        angles = [0, np.pi / 2]
        properties = ["contrast", "correlation"]
        window_sizes = [5]
        extractor = GLCMExtractor(
            bands=bands,
            distances=distances,
            angles=angles,
            properties=properties,
            window_sizes=window_sizes,
        )

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["bands_used"] == bands
        assert result["distances"] == distances
        assert result["properties"] == properties
        assert result["window_sizes"] == window_sizes

    def test_auto_determine_levels_with_explicit(self):
        """Test auto_determine_levels returns explicit levels if set."""
        # Arrange
        extractor = GLCMExtractor(levels=48)
        image = np.random.rand(10, 10) * 100

        # Act
        levels = extractor._auto_determine_levels(image)

        # Assert
        assert levels == 48

    @pytest.mark.parametrize(
        "dynamic_range,expected",
        [
            (20, 32),
            (50, 50),
            (100, 64),
        ],
    )
    def test_auto_determine_levels_by_range(self, dynamic_range, expected):
        """Test level determination across dynamic ranges."""
        # Arrange
        extractor = GLCMExtractor(levels=None)
        image = np.array([[0.0, float(dynamic_range)]])

        # Act
        levels = extractor._auto_determine_levels(image)

        # Assert
        assert levels == expected

    def test_normalize_to_levels(self):
        """Test band normalization maps to [0, levels-1]."""
        # Arrange
        extractor = GLCMExtractor()
        band = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32)
        levels = 32

        # Act
        normalized = extractor._normalize_to_levels(band, levels)

        # Assert
        assert normalized.dtype == np.uint8
        assert normalized.min() == 0
        assert normalized.max() == levels - 1

    def test_normalize_to_levels_constant_band(self):
        """Test normalization of constant band returns all zeros."""
        # Arrange
        extractor = GLCMExtractor()
        band = np.ones((5, 5)) * 5.0

        # Act
        normalized = extractor._normalize_to_levels(band, 32)

        # Assert
        assert np.all(normalized == 0)

    def test_extract_glcm_multiscale(self):
        """Test multiscale concatenates features from all window sizes."""
        # Arrange
        extractor = GLCMExtractor(
            window_sizes=[3, 5],
            properties=["contrast"],
            angles=[0],
        )
        image = (
            np.random.RandomState(0)
            .randint(0, 100, size=(10, 10))
            .astype(np.float32)
        )

        # Act
        features, levels = extractor._extract_glcm_multiscale(image)

        # Assert: 2 scales × 1 distance × 1 angle × 1 property = 2 features
        assert features.shape == (10, 10, 2)
        assert levels > 0

    @pytest.mark.parametrize(
        "orientation_mode", ["separate", "look_direction", "average"]
    )
    def test_different_orientation_modes(self, small_hsi, orientation_mode):
        """Test extraction with different orientation modes."""
        # Arrange
        extractor = GLCMExtractor(orientation_mode=orientation_mode)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert result["orientation_mode"] == orientation_mode
        assert result["features"].ndim == 3

    def test_validate_invalid_bands(self, small_hsi):
        """Test validation fails with invalid bands."""
        extractor = GLCMExtractor(bands="invalid")  # type: ignore
        with pytest.raises(
            ValueError, match="bands must be None or a non-empty list"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_distances(self, small_hsi):
        """Test validation fails with invalid distances."""
        extractor = GLCMExtractor(distances=[])
        with pytest.raises(
            ValueError, match="distances must be a non-empty list"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_angles(self, small_hsi):
        """Test validation fails with invalid angles."""
        extractor = GLCMExtractor(angles=[])
        with pytest.raises(
            ValueError, match="angles must be a non-empty list"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_window_sizes_empty(self, small_hsi):
        """Test validation fails with empty window_sizes."""
        extractor = GLCMExtractor(window_sizes=[])
        with pytest.raises(
            ValueError, match="window_sizes must be a non-empty list"
        ):
            extractor.extract(small_hsi)

    @pytest.mark.parametrize("invalid_window", [2, 4, 1])
    def test_validate_invalid_window_size_values(
        self, small_hsi, invalid_window
    ):
        """Test validation fails with invalid window size values."""
        extractor = GLCMExtractor(window_sizes=[invalid_window])
        with pytest.raises(
            ValueError, match="Each window size must be an odd integer"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_levels(self, small_hsi):
        """Test validation fails with invalid levels."""
        extractor = GLCMExtractor(levels=1)
        with pytest.raises(ValueError, match="levels must be greater than 1"):
            extractor.extract(small_hsi)

    def test_validate_invalid_properties(self, small_hsi):
        """Test validation fails with invalid properties."""
        extractor = GLCMExtractor(properties=[])
        with pytest.raises(
            ValueError, match="properties must be a non-empty list"
        ):
            extractor.extract(small_hsi)

    def test_validate_invalid_orientation_mode(self, small_hsi):
        """Test validation fails with invalid orientation mode."""
        extractor = GLCMExtractor(orientation_mode="invalid")
        with pytest.raises(ValueError, match="orientation_mode must be"):
            extractor.extract(small_hsi)

    def test_feature_name(self):
        """Test that feature name is correctly generated."""
        assert GLCMExtractor.feature_name() == "g_l_c_m"

    def test_explicit_levels(self, small_hsi):
        """Test extraction with explicit level specification."""
        # Arrange
        extractor = GLCMExtractor(levels=32)

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert all(level == 32 for level in result["levels_used"])

    def test_all_bands_default(self, small_hsi):
        """Test that default behavior processes all bands."""
        # Arrange
        extractor = GLCMExtractor()

        # Act
        result = extractor.extract(small_hsi)

        # Assert
        assert len(result["bands_used"]) == small_hsi.reflectance.shape[2]
