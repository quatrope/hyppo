"""Legendre moment feature extractor for hyperspectral images."""

import numpy as np
from scipy.special import legendre
from skimage.util.shape import view_as_windows

from hyppo.core import HSI
from ._validators import (
    validate_non_negative_int,
    validate_positive_int,
    validate_window_sizes,
)
from .base import Extractor


class LegendreMomentExtractor(Extractor):
    """
    Legendre Moment feature extractor for hyperspectral images (HSI).

    Computes multiscale Legendre moments on the principal components of
    the HSI. For each principal component, the image is processed with
    sliding windows of specified sizes, and Legendre polynomials up to
    `max_order` are used to compute orthogonal moments. The features from
    all scales and components are concatenated into the final feature set.

    This extractor depends on a PCA extractor to provide dimensionality
    reduction. If no PCA result is provided, a default PCAExtractor with
    ``n_components`` matching this extractor's setting is used.

    Parameters
    ----------
    n_components : int, default=3
        Number of PCA components to use for Legendre moment computation.
    max_order : int, default=6
        Maximum order of Legendre polynomials used to compute moments.
    window_sizes : list of int, default=[3, 9, 15]
        List of odd window sizes for multiscale moment computation.

    References
    ----------
    Mirzapour, A., & Ghassemian, H. (2016). Comparison of geometric,
        Zernike, and Legendre moments for hyperspectral images.
    Teague, M. R. (1980). Image analysis via the general theory of moments.
        Journal of the Optical Society of America, 70(8), 920–930.
    Zhou, Y., & Chellappa, R. (2004). Multiscale Legendre moments for
        image representation. Pattern Recognition, 37(7), 1387–1397.

    """

    def __init__(self, n_components=3, max_order=6, window_sizes=[3, 9, 15]):
        """Initialize Legendre moment extractor with parameters."""
        super().__init__()
        self.n_components = n_components
        self.max_order = max_order
        self.window_sizes = window_sizes

    @classmethod
    def get_input_dependencies(cls) -> dict:
        """Declare PCA as an input dependency."""
        from .pca import PCAExtractor

        return {
            "pca": {
                "extractor": PCAExtractor,
                "required": False,
            }
        }

    @classmethod
    def get_input_default(cls, input_name: str):
        """Provide default PCA extractor when none is supplied."""
        if input_name == "pca":
            from .pca import PCAExtractor

            return PCAExtractor(n_components=3)
        return None

    def _build_legendre_kernels(self, height, width):
        """Build 2D orthonormal Legendre polynomial kernels."""
        # Create normalized coordinates in [-1, 1] for the unit square
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)

        # Pre-compute 1D Legendre polynomials
        poly_x = [legendre(p)(x) for p in range(self.max_order + 1)]
        poly_y = [legendre(q)(y) for q in range(self.max_order + 1)]

        # Build 2D orthonormal kernels:
        # h_pq(x,y) = sqrt((2p+1)(2q+1)/4) * P_p(x) * P_q(y)
        kernels = []
        for p in range(self.max_order + 1):
            for q in range(self.max_order + 1 - p):
                # Normalization factor
                norm_factor = np.sqrt((2 * p + 1) * (2 * q + 1)) / 2.0
                # 2D Legendre polynomial: L_pq(x,y) = P_p(x) * P_q(y)
                L_pq = np.outer(poly_y[q], poly_x[p])
                # Apply normalization to get orthonormal basis
                h_pq = norm_factor * L_pq
                kernels.append(h_pq)

        return np.stack(kernels, axis=0)  # (M, h, w)

    def _legendre_moments(self, patches):
        """Compute Legendre moments for a set of patches."""
        N, height, width = patches.shape

        kernels = self._build_legendre_kernels(height, width)
        M = kernels.shape[0]

        # Compute moments
        moments = np.zeros((N, M))
        block_size = 50000

        for i in range(0, N, block_size):
            batch = patches[i : i + block_size]
            moments[i : i + block_size] = np.einsum(
                "bij, kij -> bk", batch, kernels
            )

        return moments

    def _extract_moments_multiscale(self, image):
        """Compute multiscale Legendre moments for a single component."""
        height, width = image.shape
        all_scales = []

        for w in self.window_sizes:
            # Add reflective padding
            pad = w // 2
            padded = np.pad(image, pad, mode="reflect")

            # Extract sliding windows
            windows = view_as_windows(padded, (w, w))
            patches = windows.reshape(-1, w, w)

            # Compute Legendre moments
            moments = self._legendre_moments(patches)
            moments = moments.reshape(height, width, -1)
            all_scales.append(moments)

        # Concatenate all scales
        return np.concatenate(all_scales, axis=-1)

    def _extract(self, data: HSI, **inputs):
        """
        Extract Legendre Moment features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.
        **inputs : dict
            Optional keyword arguments. If ``pca`` is provided, its
            ``"features"`` array is used directly instead of running
            PCA internally.

        Returns
        -------
        dict
            Dictionary containing:
                - "features": np.ndarray, shape (H, W, n_features)
                    Legendre moment features concatenated across scales
                    and components.
                - "explained_variance_ratio": array
                    Variance ratio explained by each PCA component.
                - "n_components": int
                    Number of PCA components used.
                - "window_sizes": list of int
                    Window sizes used for multiscale computation.
                - "max_order": int
                    Maximum order of Legendre moments used.
                - "n_moments_per_scale": int
                    Number of moments computed per scale/component.
        """
        pca_result = inputs.get("pca")

        # Use PCA-reduced components
        pcs = pca_result["features"]
        n_components = pcs.shape[2]

        # Extract moments for each principal component
        all_features = []
        for i in range(n_components):
            feats = self._extract_moments_multiscale(pcs[..., i])
            all_features.append(feats)

        # Concatenate features from all components
        features = np.concatenate(all_features, axis=-1)

        # Calculate number of moments per scale
        n_moments_per_scale = sum(
            1
            for p in range(self.max_order + 1)
            for q in range(self.max_order + 1 - p)
        )

        return {
            "features": features,
            "explained_variance_ratio": (
                pca_result["explained_variance_ratio"]
            ),
            "n_components": n_components,
            "window_sizes": self.window_sizes,
            "max_order": self.max_order,
            "n_moments_per_scale": n_moments_per_scale,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        validate_positive_int(self.n_components, "n_components")
        validate_non_negative_int(self.max_order, "max_order")
        validate_window_sizes(self.window_sizes)
