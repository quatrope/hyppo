"""Geometric moment feature extractor for hyperspectral images."""

import numpy as np
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA

from hyppo.core import HSI
from .base import Extractor
from ._validators import (
    validate_non_negative_int,
    validate_positive_int,
    validate_sufficient_bands,
    validate_window_sizes,
)


class GeometricMomentExtractor(Extractor):
    """
    Geometric Moment feature extractor for hyperspectral images (HSI).

    Computes multiscale geometric (raw) moments on the principal
    components of the HSI. For each principal component, the image is
    processed with sliding windows of specified sizes. Monomials X^p *
    Y^q up to a specified maximum order are used to compute the geometric
    moments within each window. Moments are concatenated across scales
    and components to form the final feature vector.

    Parameters
    ----------
    n_components : int, default=3
        Number of PCA components to retain before computing geometric moments.
    max_order : int, default=6
        Maximum order of geometric moments to compute.
    window_sizes : list of int, default=[3, 9, 15, 27]
        List of odd window sizes for multiscale moment computation.
    normalize_coords : bool, default=True
        If True, normalize pixel coordinates to [-1, 1] range for numerical
        stability (recommended by literature).

    References
    ----------
    Mirzapour, A., & Ghassemian, H. (2016). Comparison of geometric,
        Zernike, and Legendre moments for hyperspectral images.
    Hu, M. K. (1962). Visual pattern recognition by moment invariants.
        IRE Transactions on Information Theory, 8(2), 179–187.
    """

    def __init__(
        self,
        n_components=3,
        max_order=6,
        window_sizes=[3, 9, 15],
        normalize_coords=True,
    ):
        """Initialize geometric moment extractor."""
        super().__init__()
        self.n_components = n_components
        self.max_order = max_order
        self.window_sizes = window_sizes
        self.normalize_coords = normalize_coords

    def _build_coordinate_grids(self, width, height):
        """Build coordinate grids, optionally normalized to [-1, 1]."""
        # Create coordinate grids
        x = np.arange(width, dtype=np.float64)
        y = np.arange(height, dtype=np.float64)

        # Normalize coordinates to [-1, 1] for numerical stability
        if self.normalize_coords:
            if width > 1:
                x = 2 * (x - x.mean()) / (width - 1)
            if height > 1:
                y = 2 * (y - y.mean()) / (height - 1)

        # Matrices with the coordinates of each pixel within the window
        return np.meshgrid(x, y)

    def _build_geometric_kernels(self, width, height):
        """Build monomial kernels X^p * Y^q for geometric moments."""
        X, Y = self._build_coordinate_grids(width, height)

        # Pre-compute powers of X and Y to avoid redundant calculations
        x_powers = [X**p for p in range(self.max_order + 1)]
        y_powers = [Y**q for q in range(self.max_order + 1)]

        # Kernels list with the monomials X^p * Y^q for p + q <= max_order
        # This is the spatial polynomial basis
        kernels = [
            x_powers[p] * y_powers[q]
            for p in range(self.max_order + 1)
            for q in range(self.max_order + 1 - p)
        ]

        # Stack the kernels in a single matrix
        return np.stack(kernels, axis=0)

    def _geometric_moments(self, patches):
        """Compute geometric moments for a set of patches."""
        N, height, width = patches.shape

        kernels = self._build_geometric_kernels(width, height)
        moments_count = kernels.shape[0]

        moments = np.zeros((N, moments_count), dtype=np.float64)
        block_size = 50000

        for i in range(0, N, block_size):
            # Extract the current batch
            batch = patches[i : i + block_size]

            # Compute moments: m_pq = Σ h_pq(x,y) * f(x,y)
            moments[i : i + block_size] = np.einsum(
                "bij, kij -> bk", batch, kernels
            )

        return moments

    def _extract_moments_multiscale(self, image):
        """Compute multiscale geometric moments for a single component."""
        height, width = image.shape
        all_scales = []

        for w in self.window_sizes:
            # Adds padding to the image
            pad = w // 2
            padded = np.pad(image, pad, mode="reflect")

            # Extract all windows using sliding window
            windows = view_as_windows(padded, (w, w))

            # Reshape windows to patches
            patches = windows.reshape(-1, w, w)

            # Compute geometric moments
            moments = self._geometric_moments(patches)

            # Reshape moments to original shape
            moments = moments.reshape(height, width, -1)
            all_scales.append(moments)

        # Concatenate all scales along the feature axis
        all_moments = np.concatenate(all_scales, axis=-1)

        return all_moments

    def _extract(self, data: HSI, **inputs):
        """
        Extract Geometric Moment features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:
                - "features": np.ndarray, shape (H, W, n_features)
                    Geometric moment features concatenated across scales
                    and components.
                - "explained_variance_ratio": array
                    Variance ratio explained by each PCA component.
                - "n_components": int, number of PCA components used.
                - "window_sizes": list of int, window sizes used for
                    multiscale computation.
                - "max_order": int, maximum order of geometric moments used.
                - "n_moments_per_scale": int
                    Number of moments computed per scale/component.
        """

        reflectance = data.reflectance
        height, width, bands = reflectance.shape
        reflectance_reshaped = reflectance.reshape(-1, bands)

        # Apply PCA for spectral reduction
        self.pca = PCA(n_components=self.n_components)
        pcs = self.pca.fit_transform(reflectance_reshaped)
        pcs = pcs.reshape(height, width, self.n_components)

        # Extract moments for each PC across all scales
        all_features = []
        for i in range(self.n_components):
            pc_img = pcs[..., i]
            # Calculate geometric moment for this PC at all scales
            feats = self._extract_moments_multiscale(pc_img)
            all_features.append(feats)

        # Concatenate features from all PCs
        features = np.concatenate(all_features, axis=-1)

        # Calculate number of moments per scale
        n_moments_per_scale = sum(
            1
            for p in range(self.max_order + 1)
            for q in range(self.max_order + 1 - p)
        )

        return {
            "features": features,
            "explained_variance_ratio": self.pca.explained_variance_ratio_,
            "n_components": self.n_components,
            "window_sizes": self.window_sizes,
            "max_order": self.max_order,
            "n_moments_per_scale": n_moments_per_scale,
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        validate_positive_int(self.n_components, "n_components")
        validate_non_negative_int(self.max_order, "max_order")
        validate_window_sizes(self.window_sizes)
        validate_sufficient_bands(data, self.n_components)
