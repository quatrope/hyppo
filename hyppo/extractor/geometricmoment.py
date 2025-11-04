"""Geometric moment feature extractor for hyperspectral images."""

import numpy as np
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA

from hyppo.core import HSI
from .base import Extractor


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
    window_sizes : list of int, default=[3, 9, 15]
        List of odd window sizes for multiscale moment computation.
    max_order : int, default=3
        Maximum order of geometric moments to compute.

    References
    ----------
    Kumar, A., & Dikshit, O. (2015a). Geometric moment features for
        hyperspectral image classification.
    Mirzapour, A., & Ghassemian, H. (2016). Comparison of geometric,
        Zernike, and Legendre moments for hyperspectral images.
    Hu, M. K. (1962). Visual pattern recognition by moment invariants.
        IRE Transactions on Information Theory, 8(2), 179–187.
    """

    def __init__(self, n_components=3, max_order=3, window_sizes=[3, 9, 15]):
        """Initialize geometric moment extractor."""
        super().__init__()
        self.n_components = n_components
        # Max order = 6 by the paper but it is slow # TODO! Check
        self.max_order = max_order
        self.window_sizes = window_sizes

    def _geometric_moments(self, patches):
        """Compute geometric moments for a set of patches."""
        N, height, width = patches.shape

        # Precompute X^p * Y^q
        x = np.arange(width)
        y = np.arange(height)
        # Matrices with the coordinates of each pixel within the window
        # Shape: (height, width)
        X, Y = np.meshgrid(x, y)

        # Kernels list with the monomials X^p * Y^q for p + q <= max_order
        # It is the spatial polynomial basis
        kernels = [
            (X**p) * (Y**q)
            for p in range(self.max_order + 1)
            for q in range(self.max_order + 1 - p)
        ]
        # Stack the kernels in a single matrix
        # Shape: (moments_count, height, width)
        kernels = np.stack(kernels, axis=0)

        moments_count = kernels.shape[0]

        moments = np.zeros((N, moments_count))
        block_size = 50000

        for i in range(0, N, block_size):
            # Extract the current batch
            batch = patches[i : i + block_size]

            # Multiply each patch by the kernel
            product = batch[:, None, :, :] * kernels[None, :, :, :]

            # Sum the product to get the scalar moment
            moments[i : i + block_size] = product.sum(axis=(-2, -1))

        return moments

    def _extract_moments_multiscale(self, image):
        """Compute multiscale geometric moments for a single component."""
        height, width = image.shape
        all_scales = []

        for w in self.window_sizes:
            # Adds padding to the image
            pad = w // 2
            padded = np.pad(image, pad, mode="reflect")
            # Extract all windows
            # (height, width, w, w)
            windows = view_as_windows(padded, (w, w))
            # Reshape windows to patches
            # (height * width, w, w)
            patches = windows.reshape(-1, w, w)

            # Compute geometric moments
            # (height * width, M)
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
        """
        reflectance = data.reflectance
        height, width, bands = reflectance.shape
        reflectance_reshaped = reflectance.reshape(-1, bands)

        # PCA para reducción espectral
        self.pca = PCA(n_components=self.n_components)
        pcs = self.pca.fit_transform(reflectance_reshaped)
        pcs = pcs.reshape(height, width, self.n_components)

        # Extraer momentos para cada PC y escala
        all_features = []
        for i in range(self.n_components):
            pc_img = pcs[..., i]
            # Calculates the geometric moments over that img
            feats = self._extract_moments_multiscale(pc_img)
            all_features.append(feats)

        # Shape: (height, width, features)
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
