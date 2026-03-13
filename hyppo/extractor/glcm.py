"""Fast Gray Level Co-occurrence Matrix (GLCM) extractor."""

import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from skimage.exposure import equalize_hist

from hyppo.core import HSI
from ._validators import (
    validate_positive_int,
    validate_window_sizes,
    validate_non_empty_list,
    validate_sufficient_bands,
)
from .base import Extractor


class GLCMExtractor(Extractor):

    HARALICK_FEATURES = [
        "asm",
        "contrast",
        "correlation",
        "variance",
        "idm",
        "sum_avg",
        "sum_var",
        "sum_entropy",
        "entropy",
        "diff_var",
        "diff_entropy",
        "imc1",
        "imc2",
    ]

    def __init__(
        self,
        distances=None,
        angles=None,
        levels=16,
        window_sizes=None,
        features=None,
        equalize=True,
        spectral_reduction="pca",
        n_components=1,
        angle_pooling="mean",
    ):
        """Initialize GLCM extractor and precompute index tables."""
        super().__init__()

        self.distances = distances if distances is not None else [1]
        self.angles = (
            angles
            if angles is not None
            else [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        )

        self.levels = levels
        self.window_sizes = window_sizes if window_sizes is not None else [7]
        self.features = features or self.HARALICK_FEATURES

        self.equalize = equalize
        self.spectral_reduction = spectral_reduction
        self.n_components = n_components
        self.angle_pooling = angle_pooling

        # Precomputación de tablas de búsqueda (Look-up tables)
        Ng = self.levels
        self._i, self._j = np.meshgrid(
            np.arange(Ng), np.arange(Ng), indexing="ij"
        )

        k_sum = (self._i + self._j).ravel()
        k_diff = np.abs(self._i - self._j).ravel()

        self._ksum_oh = np.zeros((Ng * Ng, 2 * Ng), dtype=np.float32)
        self._ksum_oh[np.arange(Ng * Ng), k_sum] = 1

        self._kdiff_oh = np.zeros((Ng * Ng, Ng), dtype=np.float32)
        self._kdiff_oh[np.arange(Ng * Ng), k_diff] = 1

    @classmethod
    def feature_name(cls):
        """Return the feature name."""
        return "glcm"

    def _spectral_reduce(self, cube):
        """Reduce spectral dimension using PCA."""
        if self.spectral_reduction is None:
            return cube

        if self.spectral_reduction == "pca":
            h, w, b = cube.shape
            X = cube.reshape(-1, b)
            pca = PCA(n_components=self.n_components)
            Xr = pca.fit_transform(X)
            return Xr.reshape(h, w, self.n_components)

        raise ValueError("Invalid spectral reduction")

    def _quantize(self, image):
        """Quantize image to the specified number of gray levels."""
        if self.equalize:
            image = equalize_hist(image)
        else:
            mn, mx = image.min(), image.max()
            if mx > mn:
                image = (image - mn) / (mx - mn)

        return np.floor(image * (self.levels - 1)).astype(np.uint8)

    def _build_glcm_maps(self, quant, dx, dy):
        """Fast GLCM construction using uniform filters."""
        h, w = quant.shape
        Ng = self.levels
        ws = self.window_sizes[0]

        q_shift = np.roll(quant, shift=(-dy, -dx), axis=(0, 1))
        pair_index = quant.astype(np.int32) * Ng + q_shift.astype(np.int32)

        glcm = np.zeros((h, w, Ng * Ng), dtype=np.float32)
        unique_pairs = np.flatnonzero(
            np.bincount(pair_index.ravel(), minlength=Ng * Ng)
        )
        mask = np.empty_like(pair_index, dtype=np.float32)

        for k in unique_pairs:
            np.equal(pair_index, k, out=mask)
            glcm[..., k] = ndimage.uniform_filter(
                mask, size=(ws, ws), mode="reflect"
            )

        glcm = glcm.reshape(h * w, Ng, Ng)
        glcm /= glcm.sum(axis=(1, 2), keepdims=True) + 1e-12

        return glcm

    def _extract_haralick_batch(self, P):
        """Compute Haralick features for a batch of GLCMs."""
        eps = 1e-12
        Ng = self.levels
        n = P.shape[0]

        ii = self._i[np.newaxis].astype(np.float32)
        jj = self._j[np.newaxis].astype(np.float32)

        px = P.sum(axis=2)
        py = P.sum(axis=1)

        ux = (ii * P).sum(axis=(1, 2))
        uy = (jj * P).sum(axis=(1, 2))

        sx = np.sqrt(
            ((ii - ux[:, None, None]) ** 2 * P).sum(axis=(1, 2)) + eps
        )
        sy = np.sqrt(
            ((jj - uy[:, None, None]) ** 2 * P).sum(axis=(1, 2)) + eps
        )

        P_flat = P.reshape(n, Ng * Ng)
        p_sum = P_flat @ self._ksum_oh
        p_diff = P_flat @ self._kdiff_oh

        res = {}
        res["asm"] = (P**2).sum(axis=(1, 2))
        res["contrast"] = ((ii - jj) ** 2 * P).sum(axis=(1, 2))

        num = ((ii - ux[:, None, None]) * (jj - uy[:, None, None]) * P).sum(
            axis=(1, 2)
        )
        res["correlation"] = num / (sx * sy + eps)

        res["variance"] = ((ii - ux[:, None, None]) ** 2 * P).sum(axis=(1, 2))
        res["idm"] = (P / (1 + (ii - jj) ** 2)).sum(axis=(1, 2))
        res["entropy"] = -(P * np.log(P + eps)).sum(axis=(1, 2))

        k = np.arange(2 * Ng, dtype=np.float32)
        res["sum_avg"] = (k * p_sum).sum(axis=1)
        sum_ent = -(p_sum * np.log(p_sum + eps)).sum(axis=1)

        res["sum_entropy"] = sum_ent
        res["sum_var"] = ((k - sum_ent[:, None]) ** 2 * p_sum).sum(axis=1)

        res["diff_entropy"] = -(p_diff * np.log(p_diff + eps)).sum(axis=1)
        res["diff_var"] = p_diff.var(axis=1)

        HXY = res["entropy"]
        HX = -(px * np.log(px + eps)).sum(axis=1)
        HY = -(py * np.log(py + eps)).sum(axis=1)

        res["imc1"] = (HXY - (HX + HY)) / (np.maximum(HX, HY) + eps)
        res["imc2"] = np.sqrt(
            np.maximum(0.0, 1.0 - np.exp(-2 * ((HX + HY) - HXY)))
        )

        return np.stack([res[name] for name in self.features], axis=1).astype(
            np.float32
        )

    def _pool_angles(self, feats):
        """Combine features across orientations."""
        feats = np.array(feats)

        if self.angle_pooling == "mean":
            return feats.mean(axis=0)

        if self.angle_pooling == "concat":
            na, h, w, nf = feats.shape
            return feats.transpose(1, 2, 0, 3).reshape(h, w, na * nf)

        if self.angle_pooling == "mean+range":
            mean = feats.mean(axis=0)
            rng = feats.max(axis=0) - feats.min(axis=0)
            return np.concatenate([mean, rng], axis=-1)

        raise ValueError("Invalid pooling")

    def _extract_from_band(self, band):
        """Extract GLCM features from a single band."""
        h, w = band.shape
        quant = self._quantize(band)
        feats_all = []

        for d in self.distances:
            for angle in self.angles:
                dx = int(round(np.cos(angle) * d))
                dy = int(round(np.sin(angle) * d))

                glcm = self._build_glcm_maps(quant, dx, dy)
                feats = self._extract_haralick_batch(glcm)
                feats_all.append(feats.reshape(h, w, -1))

        return self._pool_angles(feats_all)

    def _extract(self, data: HSI, **inputs):
        """Extract GLCM features from HSI data."""
        cube = self._spectral_reduce(data.reflectance)
        h, w, bands = cube.shape
        feats = []

        for b in range(bands):
            feats.append(self._extract_from_band(cube[:, :, b]))

        features = np.concatenate(feats, axis=-1)

        return {
            "features": features,
            "n_components": bands,
            "window_sizes": self.window_sizes,
            "distances": self.distances,
            "levels": self.levels,
            "angle_pooling": self.angle_pooling,
            "n_features": features.shape[-1],
        }

    def _validate(self, data: HSI, **inputs):

        validate_positive_int(self.levels, "levels")

        if self.levels < 2:
            raise ValueError("levels must be >= 2")

        validate_window_sizes(self.window_sizes)

        for ws in self.window_sizes:
            if ws % 2 == 0:
                raise ValueError("window_sizes must be odd")

        validate_non_empty_list(self.distances, "distances")
        validate_non_empty_list(self.angles, "angles")

        if self.spectral_reduction not in {None, "pca"}:
            raise ValueError("Invalid spectral_reduction")

        if self.spectral_reduction == "pca":
            validate_sufficient_bands(data, self.n_components)

        if self.angle_pooling not in {"mean", "concat", "mean+range"}:
            raise ValueError("Invalid angle_pooling")

        invalid = set(self.features) - set(self.HARALICK_FEATURES)
        if invalid:
            raise ValueError(f"Invalid Haralick features: {invalid}")
