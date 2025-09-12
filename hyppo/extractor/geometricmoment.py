from .base import Extractor
from hyppo.core import HSI
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
import numpy as np


class GeometricMomentExtractor(Extractor):
    """
    Geometric Moment feature extractor for hyperspectral images (HSI).

    Computes multiscale geometric (raw) moments on the principal components of the HSI.
    For each principal component, the image is processed with sliding windows of
    specified sizes. Monomials X^p * Y^q up to a specified maximum order are used
    to compute the geometric moments within each window. Moments are concatenated
    across scales and components to form the final feature vector.

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
    Kumar, A., & Dikshit, O. (2015a). Geometric moment features for hyperspectral image classification.
    Mirzapour, A., & Ghassemian, H. (2016). Comparison of geometric, Zernike, and Legendre moments for hyperspectral images.
    Hu, M. K. (1962). Visual pattern recognition by moment invariants. IRE Transactions on Information Theory, 8(2), 179–187.
    """

    def __init__(self, n_components=3, max_order=3, window_sizes=[3, 9, 15]):
        super().__init__()
        self.n_components = n_components
        self.max_order = max_order  # max_order = 6 por el paper pero es lentisimo
        self.window_sizes = window_sizes

    def _geometric_moments(self, patches):
        """Compute geometric moments for a set of patches."""
        N, h, w = patches.shape

        # Precomputar X^p * Y^q
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(
            x, y
        )  # (h, w)  # matrices con las coord de cada pixel dentro de la ventana

        # lista de kernels que son los monomios X^p·Y^q para los (p,q) tq p+q <= max_order
        # o sea es la base polinomica espacial -> de aca saco la sumatoria desp
        kernels = [
            (X**p) * (Y**q)
            for p in range(self.max_order + 1)
            for q in range(self.max_order + 1 - p)
        ]
        kernels = np.stack(kernels, axis=0)  # (M, h, w)    # apila en una unica matriz
        M = kernels.shape[0]  # cant de momentos

        moments = np.zeros((N, M))
        block_size = 50000

        for i in range(0, N, block_size):
            batch = patches[i : i + block_size]  # extrae el bloque actual
            product = (
                batch[:, None, :, :] * kernels[None, :, :, :]
            )  # multiplica cada patch por el kernel => la sumatoria
            moments[i : i + block_size] = product.sum(
                axis=(-2, -1)
            )  # suma para obtener momento escalar

        return moments

    def _extract_moments_multiscale(self, image):
        """Compute multiscale geometric moments for a single component."""
        H, W = image.shape
        all_scales = []

        for w in self.window_sizes:
            pad = w // 2  # pixeles a agregar por lado
            padded = np.pad(
                image, pad, mode="reflect"
            )  # completa la imagen por reflexion
            windows = view_as_windows(
                padded, (w, w)
            )  # (H, W, w, w)   # extrae todas las ventanas

            # Aplicar kernels por broadcasting
            patches = windows.reshape(
                -1, w, w
            )  # (H*W, w, w)      # convierte las ventanas en patches

            moments = self._geometric_moments(patches)  # (H*W, M)

            moments = moments.reshape(H, W, -1)  # forma original
            all_scales.append(moments)  # agrega los momentos de esta escala a la lista
        all_moments = np.concatenate(
            all_scales, axis=-1
        )  # concatena todas las salidad a lo largo del eje de features

        return all_moments

    def extract(self, data: HSI, **inputs):
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
                    Geometric moment features concatenated across scales and components.
                - "explained_variance_ratio": array
                    Variance ratio explained by each PCA component.
                - "n_components": int, number of PCA components used.
                - "window_sizes": list of int, window sizes used for multiscale computation.
                - "max_order": int, maximum order of geometric moments used.
        """
        X = data.reflectance()
        h, w, b = X.shape
        X_reshaped = X.reshape(-1, b)

        # PCA para reducción espectral
        self.pca = PCA(n_components=self.n_components)
        pcs = self.pca.fit_transform(X_reshaped)
        pcs = pcs.reshape(h, w, self.n_components)

        # Extraer momentos para cada PC y escala
        all_features = []
        for i in range(self.n_components):
            pc_img = pcs[..., i]  # img 2d corresp al i-PC => (h, w)
            feats = self._extract_moments_multiscale(
                pc_img
            )  # calcula los momentos geom sobre esa img
            all_features.append(feats)

        features = np.concatenate(all_features, axis=-1)  # (h, w, features)

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
