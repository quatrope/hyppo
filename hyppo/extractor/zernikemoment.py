"""Zernike moment feature extractor for hyperspectral images."""

import numpy as np
from scipy.special import factorial
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA

from hyppo.core import HSI
from .base import Extractor


class ZernikeMomentExtractor(Extractor):
    """
    Zernike Moment feature extractor for hyperspectral images (HSI).

    Computes multiscale Zernike moments on principal components.

    For each principal component, the image is processed with sliding
    windows of specified sizes. Zernike polynomials up to a specified
    max_order are used to compute orthogonal moments within the unit disk.
    The magnitude of the moments is used as features, which are
    concatenated across scales and components.

    Parameters
    ----------
    n_components : int, default=3
        Number of PCA components to retain before computing Zernike moments.
    max_order : int, default=3
        Maximum order (degree) of Zernike polynomials to compute.
    window_sizes : list of int, default=[3, 9, 15]
        List of odd window sizes for multiscale moment computation.

    References
    ----------
    Mirzapour, A., & Ghassemian, H. (2016). Comparison of geometric,
        Zernike, and Legendre moments for hyperspectral images.
    Teague, M. R. (1980). Image analysis via the general theory
        of moments. Journal of the Optical Society of America, 70(8),
        920–930.
    Khotanzad, A., & Hong, Y. H. (1990). Invariant image
        recognition by Zernike moments. IEEE Transactions on Pattern
        Analysis and Machine Intelligence, 12(5), 489–497.
    """

    def __init__(self, n_components=3, max_order=6, window_sizes=[3, 9, 15]):
        """Initialize Zernike moment extractor.

        With PCA and polynomial parameters.
        """
        super().__init__()
        self.n_components = n_components
        self.max_order = max_order
        self.window_sizes = window_sizes

    def _zernike_radial_poly(self, p, q, r):
        """Compute the radial polynomial of Zernike moment."""
        R = np.zeros_like(r)
        q_abs = abs(q)

        for k in range((p - q_abs) // 2 + 1):
            # Compute coefficient for this term in the summation
            num = ((-1) ** k) * factorial(p - k)
            den = (
                factorial(k)
                * factorial((p + q_abs) // 2 - k)
                * factorial((p - q_abs) // 2 - k)
            )
            coef = num / den
            # Add this term to the polynomial: c_k * r^(p-2k)
            R += coef * r ** (p - 2 * k)
        return R

    def _zernike_moments(self, patches):
        """Compute Zernike moments for a set of patches."""
        N, height, width = patches.shape

        # Create normalized coordinates in [-1, 1] for the unit square
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)

        # Convert to polar coordinates for Zernike polynomials
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)

        # Mask for unit disk (Zernike polynomials defined on r ≤ 1)
        mask = R <= 1.0

        # Build list of valid (p, q) pairs, q ∈ {-p, -p+2, ..., p-2, p} with
        # (p-|q|) even. But we only use q ≥ 0 to avoid duplicate moments.
        pq_list = []
        for p in range(self.max_order + 1):
            # Only non-negative q values
            for q in range(p + 1):
                # Check parity condition: (p - q) must be even
                if (p - q) % 2 == 0:
                    pq_list.append((p, q))
        n_moments = len(pq_list)

        # Pre-compute Zernike polynomial basis functions (kernels)
        kernels = np.zeros((n_moments, height, width), dtype=complex)

        for i, (p, q) in enumerate(pq_list):
            # Compute radial polynomial R_pq(r)
            R_pq = self._zernike_radial_poly(p, q, R)

            # Compute complex Zernike polynomial V_pq(r,θ) = R_pq(r) * exp(j*q*θ)
            # Use the complex conjugate: V_pq* = R_pq(r) * exp(-j*q*θ)
            V_pq_star = R_pq * np.exp(-1j * q * Theta)

            # Apply normalization and mask:
            # h_pq(x,y) = (p+1)/π * V_pq*(x,y) for (x,y) in unit disk
            h_pq = ((p + 1) / np.pi) * V_pq_star * mask
            kernels[i] = h_pq

        # Compute moments in blocks
        moments = np.zeros((N, n_moments), dtype=np.float64)
        block_size = 50000

        for i in range(0, N, block_size):
            batch = patches[i : i + block_size]

            # Compute complex moments: m_pq = Σ h_pq(x,y) * f(x,y)
            complex_moments = np.einsum("bij,kij->bk", batch, kernels)

            # Take magnitude of complex moments as features
            moments[i : i + block_size] = np.abs(complex_moments)

        return moments

    def _extract_moments_multiscale(self, image):
        """Compute multiscale Zernike moments for a single component."""
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
            moments = self._zernike_moments(patches)
            moments = moments.reshape(height, width, -1)
            all_scales.append(moments)

        # Concatenate all scales
        return np.concatenate(all_scales, axis=-1)

    def _extract(self, data: HSI, **inputs):
        """
        Extract Zernike Moment features from a hyperspectral image.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:
                - "features": np.ndarray, shape (H, W, n_features)
                    Zernike moment features (magnitudes) concatenated 
                    across scales and components.
                - "explained_variance_ratio": array
                    Variance ratio explained by each PCA component.
                - "n_components": int
                    Number of PCA components used.
                - "window_sizes": list of int
                    Window sizes used for multiscale computation.
                - "max_order": int
                    Maximum order of Zernike polynomials used.
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

        # Extract moments for each principal component
        all_features = []
        for i in range(self.n_components):
            feats = self._extract_moments_multiscale(pcs[..., i])
            all_features.append(feats)

        # Concatenate features from all components
        features = np.concatenate(all_features, axis=-1)

        # Calculate number of moments per scale (only for q >= 0)
        n_moments_per_scale = sum(
            1
            for p in range(self.max_order + 1)
            for q in range(0, p + 1)
            if (p - q) % 2 == 0
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
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
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
        if not isinstance(self.max_order, int) or self.max_order < 0:
            raise ValueError("max_order must be a non-negative integer.")
        if data.reflectance.shape[-1] < self.n_components:
            raise ValueError(
                f"Number of spectral bands ({data.reflectance.shape[-1]}) "
                f"is less than n_components ({self.n_components})."
            )
