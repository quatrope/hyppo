import numpy as np
from hyppo.core import HSI
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA
from scipy.special import legendre
from .base import Extractor


class LegendreMomentExtractor(Extractor):
    """
    Legendre Moment feature extractor for hyperspectral images (HSI).

    Computes multiscale Legendre moments on the principal components of the HSI.
    For each principal component, the image is processed with sliding windows of
    specified sizes, and Legendre polynomials up to `max_order` are used to compute
    orthogonal moments. The features from all scales and components are concatenated
    into the final feature set.

    Parameters
    ----------
    n_components : int, default=3
        Number of PCA components to retain before computing Legendre moments.
    max_order : int, default=3
        Maximum order of Legendre polynomials used to compute moments.
    window_sizes : list of int, default=[3, 9, 15]
        List of odd window sizes for multiscale moment computation.

    References
    ----------
    Teague, M. R. (1980). Image analysis via the general theory of moments.
        Journal of the Optical Society of America, 70(8), 920–930.
    Lv, Z. Y., Zhang, P., Benediktsson, J. A., & Shi, W. Z. (2014).
        Morphological profiles based on differently shaped structuring elements
        for classification of images with very high spatial resolution.
        IEEE JSTARS, 7(12), 4644–4652.
    Zhou, Y., & Chellappa, R. (2004). Multiscale Legendre moments for image representation.
        Pattern Recognition, 37(7), 1387–1397.

    """

    def __init__(self, n_components=3, max_order=3, window_sizes=[3, 9, 15]):
        super().__init__()
        self.n_components = n_components
        self.max_order = max_order
        self.window_sizes = window_sizes

    def _legendre_moments(self, patches):
        """Compute Legendre moments for a set of patches."""
        N, h, w = patches.shape

        # Crear coordenadas normalizadas en [-1, 1]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)  # (h, w)

        # Construir base de polinomios de Legendre
        kernels = [
            np.outer(legendre(p)(x), legendre(q)(y)).T
            for p in range(self.max_order + 1)
            for q in range(self.max_order + 1 - p)
        ]
        kernels = np.stack(kernels, axis=0)  # (M, h, w)
        M = kernels.shape[0]

        # Calcular momentos por bloques
        moments = np.zeros((N, M))
        block_size = 50_000

        for i in range(0, N, block_size):
            batch = patches[i : i + block_size]
            product = batch[:, None, :, :] * kernels[None, :, :, :]
            moments[i : i + block_size] = product.sum(axis=(-2, -1))

        return moments

    def _extract_moments_multiscale(self, image):
        """Compute multiscale Legendre moments for a single component."""
        H, W = image.shape
        all_scales = []

        for w in self.window_sizes:
            pad = w // 2
            padded = np.pad(image, pad, mode="reflect")
            windows = view_as_windows(padded, (w, w))
            patches = windows.reshape(-1, w, w)

            moments = self._legendre_moments(patches)
            moments = moments.reshape(H, W, -1)
            all_scales.append(moments)

        return np.concatenate(all_scales, axis=-1)

    def extract(self, data: HSI):
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
                    Legendre moment features concatenated across scales and components.
                - "explained_variance_ratio": array
                    Variance ratio explained by each PCA component.
                - "n_components": int, number of PCA components used.
                - "window_sizes": list of int, window sizes used for multiscale computation.
                - "max_order": int, maximum Legendre polynomial order used.
        """
        X = data.reflectance()
        h, w, b = X.shape
        X_reshaped = X.reshape(-1, b)

        # Reducción espectral con PCA
        self.pca = PCA(n_components=self.n_components)
        pcs = self.pca.fit_transform(X_reshaped)
        pcs = pcs.reshape(h, w, self.n_components)

        # Extraer momentos para cada componente principal
        all_features = []
        for i in range(self.n_components):
            feats = self._extract_moments_multiscale(pcs[..., i])
            all_features.append(feats)

        features = np.concatenate(all_features, axis=-1)

        return {
            "features": features,
            "explained_variance_ratio": self.pca.explained_variance_ratio_,
            "n_components": self.n_components,
            "window_sizes": self.window_sizes,
            "max_order": self.max_order,
        }

    def validate(self):
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
