import numpy as np
import math
from hyppo.core import HSI
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA
from .base import Extractor


class ZernikeMomentExtractor(Extractor):
    """
    Zernike Moment feature extractor for hyperspectral images (HSI).

    Computes multiscale Zernike moments on the principal components of the HSI.
    For each principal component, the image is processed with sliding windows of
    specified sizes. Zernike polynomials up to a specified degree are used to
    compute orthogonal moments within the unit disk. The magnitude of the moments
    is used as features, which are concatenated across scales and components.

    Parameters
    ----------
    n_components : int, default=3
        Number of PCA components to retain before computing Zernike moments.
    window_sizes : list of int, default=[3, 9, 15]
        List of odd window sizes for multiscale moment computation.
    degree : int, default=3
        Maximum degree of Zernike polynomials to compute.

    References
    ----------
    Teague, M. R. (1980). Image analysis via the general theory of moments.
        Journal of the Optical Society of America, 70(8), 920–930.
    Khotanzad, A., & Hong, Y. H. (1990). Invariant image recognition by Zernike moments.
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(5), 489–497.
    Mukundan, R., & Ramakrishnan, K. R. (1998). Moment functions in image analysis:
        Theory and applications. World Scientific.
    """

    def __init__(self, n_components=3, window_sizes=[3, 9, 15], degree=3):
        super().__init__()
        self.n_components = n_components
        self.window_sizes = window_sizes
        self.degree = degree

    def _zernike_radial_poly(self, n, m, r):
        """Compute the radial polynomial of Zernike moment."""
        R = np.zeros_like(r)
        m = abs(m)
        for s in range((n - m) // 2 + 1):  # sumatoria
            c = ((-1) ** s * math.factorial(n - s)) / (
                math.factorial(s)
                * math.factorial((n + m) // 2 - s)
                * math.factorial((n - m) // 2 - s)
            )
            R += c * r ** (n - 2 * s)  # polinomio radial evaluado en r
        return R

    def _zernike_moments(self, patches):
        """Compute Zernike moments for a set of patches."""
        N, h, w = patches.shape
        degree = self.degree

        # crea coordenadas normalizadas en [-1, 1] para cada pixel
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)

        R = np.sqrt(X**2 + Y**2)  # radio
        Theta = np.arctan2(Y, X)  # angulo en coord polares

        mask = (
            R <= 1.0
        )  # define la region del disco unitario (donde se def la base de zernike)
        area = np.sum(mask)  # cant de pixels dentro del disco para normalizar

        # lista con todos los valores (n,m) validos segun def de zernike
        nm_list = []
        for n in range(degree + 1):  # n \in [0, degree]
            for m in range(-n, n + 1, 2):  # m \in [-n, n] con paso 2
                if (n - abs(m)) % 2 == 0:  # solo si n-|m| es par
                    nm_list.append((n, m))
        M = len(nm_list)

        # precalcula la base de polinomios
        kernels = np.zeros((M, h, w), dtype=complex)
        for i, (n, m) in enumerate(nm_list):  # para (n,m) en el enrejado del parche
            Rnm = self._zernike_radial_poly(n, m, R)  # parte radial
            kernels[i] = Rnm * np.exp(
                -1j * m * Theta
            )  # multiplica por la parte angular
            kernels[i] *= mask  # multiplica por mask para forzar ceros fuera del disco

        moments = np.zeros((N, M))

        block_size = 50000
        for i in range(0, N, block_size):
            batch = patches[i : i + block_size]
            prod = (
                batch[:, None, :, :] * kernels[None, :, :, :]
            )  # multiplica cada batch por el polinomio
            complex_moments = (
                prod.sum(axis=(-2, -1)) / area
            )  # suma (que seria integrarcion discreta) y normalizacion
            moments[i : i + block_size] = np.abs(complex_moments)  # solo módulo real

        return moments

    def _extract_moments_multiscale(self, image):
        """Compute multiscale Zernike moments for a single component."""
        H, W = image.shape
        all_scales = []

        for w in self.window_sizes:
            r = w // 2
            padded = np.pad(image, r, mode="reflect")
            windows = view_as_windows(padded, (w, w))
            patches = windows.reshape(-1, w, w)

            moments = self._zernike_moments(patches)
            moments = moments.reshape(H, W, -1)
            all_scales.append(moments)

        return np.concatenate(all_scales, axis=-1)

    def extract(self, data: HSI):
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
                    Zernike moment features concatenated across scales and components.
                - "explained_variance_ratio": array
                    Variance ratio explained by each PCA component.
                - "n_components": int, number of PCA components used.
                - "window_sizes": list of int, window sizes used for multiscale computation.
                - "degree": int, maximum degree of Zernike polynomials used.
        """
        X = data.reflectance()
        h, w, b = X.shape
        X_reshaped = X.reshape(-1, b)

        self.pca = PCA(n_components=self.n_components)
        pcs = self.pca.fit_transform(X_reshaped).reshape(h, w, self.n_components)

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
            "degree": self.degree,
        }

    def validate(self):
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
        if not isinstance(self.degree, int) or self.degree < 0:
            raise ValueError("degree must be a non-negative integer.")
