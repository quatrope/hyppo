from .base import Extractor
from hyppo.core import HSI
from skimage.util.shape import view_as_windows
from skimage.feature import graycoprops, graycomatrix

import numpy as np


class GLCMExtractor(Extractor):
    def __init__(
        self,
        bands=None,  # Bandas específicas a procesar (None = todas)
        distances=None,  # Pixel distances for GLCM computation
        angles=None,  # Angles in radians for GLCM computation
        symmetric=True,  # Whether to compute symmetric GLCM matrices
        properties=None,  # Properties to calculate
        levels=None,  # Number of levels for quantization (auto-determined if None)
        window_sizes=[7],  # Window sizes for multiscale analysis
        orientation_mode="separate",  # 'separate', 'look_direction', or 'average'
    ):
        super().__init__()

        self.bands = bands

        # Paper recommends δ = 1 for nearest neighbor analysis
        self.distances = distances if distances is not None else [1]

        # Paper uses 0°, 45°, 90°, 135° - all orientations separately
        self.angles = (
            angles if angles is not None else [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        )

        self.symmetric = symmetric

        # Auto-determine levels based on paper's findings (G > 24, typically 32-64)
        self.levels = levels

        # Paper's recommended statistics: Contrast (CON), Entropy (ENT), Correlation (COR)
        # Plus Dissimilarity (DIS) as alternative to Contrast
        self.properties = (
            properties
            if properties is not None
            else ["contrast", "entropy", "correlation", "dissimilarity"]
        )

        # Larger windows for better texture characterization
        self.window_sizes = window_sizes

        # How to handle orientations based on paper's findings
        self.orientation_mode = orientation_mode

    def _auto_determine_levels(self, image):
        """
        Auto-determine quantization levels based on image characteristics
        Following paper's recommendation: G > 24, typically 32-64
        """
        if self.levels is not None:
            return self.levels

        # Calculate image dynamic range
        img_min, img_max = image.min(), image.max()
        dynamic_range = img_max - img_min

        # Determine levels based on dynamic range and paper recommendations
        if dynamic_range <= 32:
            return 32
        elif dynamic_range <= 64:
            return min(64, int(dynamic_range))
        else:
            # For high dynamic range, use 64 levels as recommended
            return 64

    def _normalize_to_levels(self, band, levels):
        """Quantizes the band to the range [0, levels-1]"""
        band_min, band_max = band.min(), band.max()
        if band_max == band_min:
            return np.zeros_like(band, dtype=np.uint8)

        normalized = (band - band_min) / (band_max - band_min) * (levels - 1)
        return normalized.astype(np.uint8)

    def _compute_glcm_features_optimized(self, patches, levels):
        """
        Optimized GLCM computation based on paper's recommendations
        """
        N, h, w = patches.shape

        if self.orientation_mode == "separate":
            # Use all orientations separately (paper's recommendation)
            n_features = len(self.distances) * len(self.angles) * len(self.properties)
        elif self.orientation_mode == "look_direction":
            # Use look direction approach: 0°, 90°, and average of 45°/135°
            n_features = len(self.distances) * 3 * len(self.properties)
        else:  # average
            # Average all orientations
            n_features = len(self.distances) * len(self.properties)

        features = np.zeros((N, n_features), dtype=np.float32)

        # Process in blocks to manage memory
        block_size = 500

        for i in range(0, N, block_size):
            batch = patches[i : i + block_size]
            batch_size = batch.shape[0]

            for j in range(batch_size):
                patch = batch[j]

                # Skip patches with insufficient variation
                if patch.max() - patch.min() < 2:
                    continue

                feature_idx = 0

                for distance in self.distances:
                    try:
                        if self.orientation_mode == "separate":
                            # Compute GLCM for each angle separately
                            glcm = graycomatrix(
                                patch,
                                distances=[distance],
                                angles=self.angles,
                                levels=levels,
                                symmetric=self.symmetric,
                            )

                            # Extract properties for each angle
                            for prop in self.properties:
                                try:
                                    vals = graycoprops(glcm, prop)[
                                        0, :
                                    ]  # All angles for this distance
                                    features[
                                        i + j, feature_idx : feature_idx + len(vals)
                                    ] = vals
                                    feature_idx += len(vals)
                                except Exception:
                                    # Fill with zeros if computation fails
                                    features[
                                        i + j,
                                        feature_idx : feature_idx + len(self.angles),
                                    ] = 0.0
                                    feature_idx += len(self.angles)

                        elif self.orientation_mode == "look_direction":
                            # Paper's look direction approach: 0°, 90°, avg(45°,135°)
                            angles_look = [0, np.pi / 2, np.pi / 4, 3 * np.pi / 4]
                            glcm = graycomatrix(
                                patch,
                                distances=[distance],
                                angles=angles_look,
                                levels=levels,
                                symmetric=self.symmetric,
                            )

                            for prop in self.properties:
                                try:
                                    vals = graycoprops(glcm, prop)[
                                        0, :
                                    ]  # [0°, 90°, 45°, 135°]
                                    # Use 0°, 90°, and average of 45°/135°
                                    look_vals = [
                                        vals[0],
                                        vals[1],
                                        (vals[2] + vals[3]) / 2,
                                    ]
                                    features[i + j, feature_idx : feature_idx + 3] = (
                                        look_vals
                                    )
                                    feature_idx += 3
                                except Exception:
                                    features[i + j, feature_idx : feature_idx + 3] = 0.0
                                    feature_idx += 3

                        else:  # average mode
                            # Average all orientations
                            glcm = graycomatrix(
                                patch,
                                distances=[distance],
                                angles=self.angles,
                                levels=levels,
                                symmetric=self.symmetric,
                            )

                            for prop in self.properties:
                                try:
                                    vals = graycoprops(glcm, prop)[0, :]
                                    avg_val = np.mean(vals)
                                    features[i + j, feature_idx] = avg_val
                                    feature_idx += 1
                                except Exception:
                                    features[i + j, feature_idx] = 0.0
                                    feature_idx += 1

                    except Exception:
                        # Skip this distance if GLCM computation fails
                        if self.orientation_mode == "separate":
                            skip_features = len(self.angles) * len(self.properties)
                        elif self.orientation_mode == "look_direction":
                            skip_features = 3 * len(self.properties)
                        else:
                            skip_features = len(self.properties)

                        features[i + j, feature_idx : feature_idx + skip_features] = 0.0
                        feature_idx += skip_features

        return features

    def _extract_glcm_multiscale(self, image):
        """
        Extract multiscale GLCM features following paper's methodology
        """
        image_height, image_width = image.shape
        all_scales = []

        # Determine quantization levels
        levels = self._auto_determine_levels(image)

        # Normalize image
        image_normalized = self._normalize_to_levels(image, levels)

        for window_size in self.window_sizes:
            # Use larger windows as recommended by the paper
            pad = window_size // 2

            # Use reflection padding to maintain texture characteristics at borders
            padded = np.pad(image_normalized, pad, mode="reflect")

            # Extract windows
            windows = view_as_windows(padded, (window_size, window_size))
            patches = windows.reshape(-1, window_size, window_size)

            # Compute GLCM features with optimized approach
            glcm_features = self._compute_glcm_features_optimized(patches, levels)

            # Reshape to spatial dimensions
            glcm_features = glcm_features.reshape(image_height, image_width, -1)
            all_scales.append(glcm_features)

        # Concatenate all scales
        all_features = np.concatenate(all_scales, axis=-1)
        return all_features, levels

    def _extract(self, data: HSI, **inputs):
        X = data.reflectance  # (H, W, B)
        h, w, b = X.shape

        # Determine which bands to process
        bands_to_process = self.bands if self.bands is not None else list(range(b))

        # Extract GLCM features for each spectral band
        all_features = []
        used_levels = []

        for band_idx in bands_to_process:
            band_img = X[..., band_idx]
            feats, levels = self._extract_glcm_multiscale(band_img)
            all_features.append(feats)
            used_levels.append(levels)

        # Concatenate features from all bands
        features = np.concatenate(all_features, axis=-1)

        # Calculate feature dimensions
        if self.orientation_mode == "separate":
            n_features_per_distance = len(self.angles) * len(self.properties)
        elif self.orientation_mode == "look_direction":
            n_features_per_distance = 3 * len(self.properties)
        else:
            n_features_per_distance = len(self.properties)

        n_features_per_scale = len(self.distances) * n_features_per_distance
        n_features_per_band = n_features_per_scale * len(self.window_sizes)
        total_features = n_features_per_band * len(bands_to_process)

        return {
            "features": features,
            "bands_used": bands_to_process,
            "distances": self.distances,
            "angles": self.angles,
            "properties": self.properties,
            "levels_used": used_levels,
            "window_sizes": self.window_sizes,
            "orientation_mode": self.orientation_mode,
            "n_features_per_scale": n_features_per_scale,
            "n_features_per_band": n_features_per_band,
            "total_features": total_features,
            "original_shape": (h, w),
        }

    def _validate(self, data: HSI, **inputs):
        if self.bands is not None and (
            not isinstance(self.bands, list) or not self.bands
        ):
            raise ValueError("bands must be None or a non-empty list of integers.")
        if not self.distances or not isinstance(self.distances, list):
            raise ValueError("distances must be a non-empty list of integers.")
        if not self.angles or not isinstance(self.angles, list):
            raise ValueError("angles must be a non-empty list of floats (radians).")
        if (
            not isinstance(self.window_sizes, (list, tuple))
            or len(self.window_sizes) == 0
        ):
            raise ValueError("window_sizes must be a non-empty list or tuple.")
        for w in self.window_sizes:
            if not isinstance(w, int) or w < 3 or w % 2 == 0:
                raise ValueError(
                    f"Each window size must be an odd integer ≥ 3. Got: {w}"
                )
        if self.levels is not None and self.levels <= 1:
            raise ValueError(
                "levels must be greater than 1 or None for auto-determination."
            )
        if not isinstance(self.properties, list) or len(self.properties) == 0:
            raise ValueError("properties must be a non-empty list.")
        if self.orientation_mode not in ["separate", "look_direction", "average"]:
            raise ValueError(
                "orientation_mode must be 'separate', 'look_direction', or 'average'."
            )
