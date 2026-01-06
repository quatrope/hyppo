"""Legendre moment feature extractor for hyperspectral images."""

import numpy as np
from scipy.special import legendre
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA

from hyppo.core import HSI
from .base import Extractor


class LegendreMomentExtractor(Extractor):
    """
    Legendre Moment feature extractor for hyperspectral images (HSI).

    Computes multiscale Legendre moments on the principal components of
    the HSI. For each principal component, the image is processed with
    sliding windows of specified sizes, and Legendre polynomials up to
    `max_order` are used to compute orthogonal moments. The features from
    all scales and components are concatenated into the final feature set.

    Parameters
    ----------
    n_components : int, default=3
        Number of PCA components to retain before computing Legendre moments.
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

    def _legendre_moments(self, patches):
        """Compute Legendre moments for a set of patches."""
        N, height, width = patches.shape

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

        kernels = np.stack(kernels, axis=0)  # (M, h, w)
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

        Returns
        -------
        dict
            Dictionary containing:
                - "features": np.ndarray, shape (H, W, n_features)
                    Legendre moment features concatenated across scales
                    and components.
                - "explained_variance_ratio": array
                    Variance ratio explained by each PCA component.
                - "n_components": int, number of PCA components used.
                - "window_sizes": list of int, window sizes used for
                    multiscale computation.
                - "max_order": int, maximum Legendre polynomial order used.
        """
        reflectance = data.reflectance
        height, width, bands = reflectance.shape
        reflectance_reshaped = reflectance.reshape(-1, bands)

        # Apply PCA for spectral reduction
        self.pca = PCA(n_components=self.n_components)
        pcs = self.pca.fit_transform(reflectance_reshaped)
        pcs = pcs.reshape(height, width, self.n_components)

        # Extract moments for each principal component
        all_features = []
        for i in range(self.n_components):
            feats = self._extract_moments_multiscale(pcs[..., i])
            all_features.append(feats)

        # Concatenate features from all components
        features = np.concatenate(all_features, axis=-1)

        return {
            "features": features,
            "explained_variance_ratio": self.pca.explained_variance_ratio_,
            "n_components": self.n_components,
            "window_sizes": self.window_sizes,
            "max_order": self.max_order,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError("n_components must be a positive integer.")

        if not isinstance(self.max_order, int) or self.max_order < 0:
            raise ValueError("max_order must be a non-negative integer.")

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
        if data.reflectance.shape[-1] < self.n_components:
            raise ValueError(
                f"Number of spectral bands ({data.reflectance.shape[-1]}) "
                f"is less than n_components ({self.n_components})."
            )
