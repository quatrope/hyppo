"""Local Binary Pattern (LBP) texture feature extractor for HSI."""

import numpy as np
from skimage.feature import local_binary_pattern

from hyppo.core import HSI
from ._validators import (
    validate_positive_int,
    validate_positive_number,
    validate_sufficient_bands,
)
from .base import Extractor
from .pca import PCAExtractor

class LBPExtractor(Extractor):
    """
    Local Binary Pattern (LBP) feature extractor for hyperspectral images.

    Supports multiscale LBP computation on either PCA components or
    original spectral bands.

    Parameters
    ----------
    radius : int or list[int], default=3
        Radius of the sampling circle.
    n_points : int or list[int], default=None
        Number of sampling points. If None, defaults to 8 * radius.
    method : str, default="ror"
        LBP method ("default", "ror", "uniform", "nri_uniform", "var").
    n_components : int, default=3
        Number of PCA components to use (only if spectral_mode="pca").
    spectral_mode : {"pca", "bands"}, default="pca"
        Whether to compute LBP on PCA components or raw spectral bands.
    band_indices : list[int], optional
        Specific bands to use if spectral_mode="bands".
    """

    def __init__(
        self,
        radius=3,
        n_points=None,
        method="ror",
        n_components=3,
        spectral_mode="pca",
        band_indices=None,
    ):
        """Initialize LBP extractor and configure multiscale parameters."""
        super().__init__()

        valid_methods = ["default", "ror", "uniform", "nri_uniform", "var"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        if spectral_mode not in ["pca", "bands"]:
            raise ValueError("spectral_mode must be 'pca' or 'bands'")

        self.radius = radius if isinstance(radius, list) else [radius]

        if n_points is None:
            self.n_points = [8 * r for r in self.radius]
        else:
            self.n_points = (
                n_points if isinstance(n_points, list) else [n_points]
            )

        if len(self.radius) != len(self.n_points):
            raise ValueError("radius and n_points must have same length.")

        self.method = method
        self.n_components = n_components
        self.spectral_mode = spectral_mode
        self.band_indices = band_indices

    @classmethod
    def feature_name(cls):
        """Return the feature name."""
        return "lbp"

    @classmethod
    def get_input_dependencies(cls) -> dict:
        """Declare PCA as an input dependency for spectral reduction."""
        return {"pca": {"extractor": PCAExtractor, "required": False}}

    def _compute_lbp_multiscale(self, band):
        """Compute LBP across all scales for a single band/component."""
        band = (
            255 * (band - band.min()) / (band.max() - band.min() + 1e-8)
        ).astype(np.uint8)
        scales = []
        for r, p in zip(self.radius, self.n_points):
            lbp = local_binary_pattern(
                band,
                P=p,
                R=r,
                method=self.method,
            )

            scales.append(lbp)
        return np.stack(scales, axis=-1)

    def _extract(self, data: HSI, **inputs):
        """
        Extract LBP features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.
        **inputs : dict
            Optional keyword arguments.

        Returns
        -------
        dict
            Dictionary containing:
                - "features": np.ndarray, shape (H, W, n_features)
                    LBP features concatenated across scales and components.
                - "explained_variance_ratio": array or None
                    Variance ratio explained by PCA components (PCA mode only).
                - "n_components": int
                    Number of spectral channels (PCA or bands) processed.
                - "spectral_mode": str
                    The mode used: 'pca' or 'bands'.
                - "method": str
                    The LBP mapping method used (e.g., 'ror', 'uniform').
                - "radius": list of int
                    Radii used for the multiscale computation.
                - "n_points": list of int
                    Sampling points used for each scale.
                - "n_scales": int
                    Number of spatial scales processed per channel.
                - "n_features": int
                    Total number of features extracted per pixel.
        """
        if self.spectral_mode == "pca":
            pca_result = inputs.get("pca")

            if pca_result is None:
                # fallback: compute PCA using the extractor
                pca_extractor = PCAExtractor(n_components=self.n_components)
                pca_result = pca_extractor.extract(data)

            pcs = pca_result["features"]

            if self.n_components is not None:
                pcs = pcs[..., : self.n_components]

            spectral_data = pcs
            explained_var = pca_result.get("explained_variance_ratio")
        else:
            cube = data.reflectance
            if self.band_indices is None:
                spectral_data = cube
            else:
                spectral_data = cube[..., self.band_indices]
            explained_var = None

        n_channels = spectral_data.shape[2]

        features = np.concatenate(
            [self._compute_lbp_multiscale(spectral_data[..., i]) for i in range(n_channels)],
            axis=-1,
        )
        return {
            "features": features,
            "method": self.method,
            "radius": self.radius,
            "n_points": self.n_points,
            "spectral_mode": self.spectral_mode,
            "n_channels": n_channels,
            "n_scales": len(self.radius),
            "n_features": features.shape[-1],
            "explained_variance_ratio": explained_var,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate LBP parameters and spectral constraints."""
        validate_positive_int(self.n_components, "n_components")
        for r in self.radius:
            validate_positive_number(r, "radius")
        for p in self.n_points:
            validate_positive_int(p, "n_points")
        if self.spectral_mode == "bands":
            if self.band_indices is not None:
                max_band = data.reflectance.shape[2] - 1
                for b in self.band_indices:
                    if b < 0 or b > max_band:
                        raise ValueError(f"band index {b} out of range")
        if self.spectral_mode == "pca":
            validate_sufficient_bands(data, self.n_components)
