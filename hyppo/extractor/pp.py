from .base import Extractor
from hyppo.hsi import HSI
import numpy as np
from sklearn.decomposition import PCA
import warnings


class PPExtractor(Extractor):
    """
    Projection Pursuit feature extractor for hyperspectral images (HSI).

    Implements the method of Ifarraguerri & Chang (2000) to find projections
    that maximize non-Gaussianity, using information divergence from a Gaussian
    distribution as the projection index.

    Parameters
    ----------
    n_projections : int, default=10
        Number of projections to keep.
    n_bins : int, default=50
        Number of histogram bins for divergence computation.
    pca_components : int or None, default=None
        Number of PCA components for sphering. If None, components are auto-selected
        to preserve 99% variance.
    sample_size : int, default=1000
        Number of pixels to sample when searching for projections.
    random_state : int, optional, default=42
        Random state for reproducibility.

    References
    ----------
    .. [1] Ifarraguerri, A., & Chang, C.-I. (2000). "Independent component analysis
           for hyperspectral image analysis." IEEE Trans. Geosci. Remote Sensing, 38(2), 677–697.
    """

    def __init__(
        self,
        n_projections=10,
        n_bins=50,
        pca_components=None,  # If None, auto-select based on 99% variance
        sample_size=1000,  # Number of pixels to sample for efficiency
        random_state=42,
    ):
        super().__init__()
        self.n_projections = n_projections
        self.n_bins = n_bins
        self.pca_components = pca_components
        self.sample_size = sample_size
        self.rng = np.random.default_rng(random_state)

    def _compute_information_divergence(self, projection_scores):
        """
        Compute information divergence between projection scores and Gaussian distribution.
        Following equations (4) and (5) from the paper.
        """
        # Standardize scores (mean=0, std=1)
        scores_std = (projection_scores - np.mean(projection_scores)) / (
            np.std(projection_scores) + 1e-10
        )

        # Create histogram with fixed range to cover Gaussian shape
        hist_range = (-4, 4)  # Covers ~99.99% of standard Gaussian
        hist, bin_edges = np.histogram(
            scores_std, bins=self.n_bins, range=hist_range, density=True
        )

        # Normalize histogram to get probability distribution p
        bin_width = bin_edges[1] - bin_edges[0]
        p = hist * bin_width
        p = p / np.sum(p)  # Ensure normalization

        # Create corresponding Gaussian distribution q
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        q = np.exp(-0.5 * bin_centers**2) / np.sqrt(2 * np.pi)
        q = q * bin_width
        q = q / np.sum(q)  # Normalize

        # Replace zeros with small values to avoid log(0)
        small_val = 1e-10
        p = np.where(p < small_val, small_val, p)
        q = np.where(q < small_val, small_val, q)

        # Compute symmetric information divergence (equation 5)
        div_pq = np.sum(p * np.log(p / q))
        div_qp = np.sum(q * np.log(q / p))
        divergence = div_pq + div_qp

        return divergence

    def _pca_preprocessing(self, X):
        """
        Apply PCA preprocessing ("sphering") as described in Section III.A
        """
        # Determine number of components if not specified
        if self.pca_components is None:
            pca_temp = PCA()
            pca_temp.fit(X)
            cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
            self.pca_components = np.argmax(cumvar >= 0.99) + 1

        # Apply PCA with sphering
        pca = PCA(n_components=self.pca_components, whiten=True)
        X_sphered = pca.fit_transform(X)

        return X_sphered, pca

    def _find_best_projection(self, X, previous_projections):
        """
        Find best projection using the simple search strategy from Section III.B:
        Use each pixel as candidate projection vector and select the one with highest index.
        """
        n_samples, n_features = X.shape

        # Sample pixels for efficiency (as mentioned in paper)
        if n_samples > self.sample_size:
            sample_indices = self.rng.choice(
                n_samples, size=self.sample_size, replace=False
            )
            X_sample = X[sample_indices]
        else:
            X_sample = X
            sample_indices = np.arange(n_samples)

        best_projection = None
        best_score = -np.inf
        best_pixel_idx = -1

        # Test each pixel as a candidate projection vector
        for i, candidate_vector in enumerate(X_sample):
            # Normalize candidate vector
            candidate_vector = candidate_vector / (
                np.linalg.norm(candidate_vector) + 1e-10
            )

            # Orthogonalize against previous projections
            for prev_proj in previous_projections:
                candidate_vector -= np.dot(candidate_vector, prev_proj) * prev_proj

            # Renormalize after orthogonalization
            norm = np.linalg.norm(candidate_vector)
            if norm < 1e-10:
                continue  # Skip if vector becomes zero after orthogonalization
            candidate_vector = candidate_vector / norm

            # Compute projection scores for all data
            projection_scores = X @ candidate_vector

            # Compute information divergence
            try:
                score = self._compute_information_divergence(projection_scores)
                if score > best_score:
                    best_score = score
                    best_projection = candidate_vector.copy()
                    best_pixel_idx = (
                        sample_indices[i] if n_samples > self.sample_size else i
                    )
            except Exception:
                continue  # Skip if divergence computation fails

        if best_projection is None:
            # Fallback to random orthogonal vector
            warnings.warn("No valid projection found, using random orthogonal vector")
            best_projection = self.rng.standard_normal(n_features)
            for prev_proj in previous_projections:
                best_projection -= np.dot(best_projection, prev_proj) * prev_proj
            best_projection = best_projection / (
                np.linalg.norm(best_projection) + 1e-10
            )
            best_score = 0.0
            best_pixel_idx = -1

        return best_projection, best_score, best_pixel_idx

    def extract(self, data: HSI):
        """
        Extract Projection Pursuit features from hyperspectral data.

        Parameters
        ----------
        data : HSI
            Hyperspectral image object containing reflectance data.

        Returns
        -------
        dict
            Dictionary containing:
            - features : ndarray of shape (H, W, n_projections)
                PP-transformed array.
            - n_features : int
                Number of projections used.
            - original_shape : tuple of int
                Shape of the original HSI cube (H, W, bands).
            - projection_vectors : ndarray of shape (n_projections, n_features)
                Array of projection vectors.
            - divergence_scores : list of float
                Information divergence score for each projection.
            - pca_components_used : int
                Number of PCA components used for sphering.
            - pca_model : sklearn.decomposition.PCA
                Fitted PCA object.
            - selected_pixel_indices : list of int
                Indices of pixels selected as projection vectors.
            - valid_pixel_mask : ndarray of bool, shape (H, W)
                Mask of valid pixels used in extraction.
        """
        X = data.reflectance()
        h, w, b = X.shape
        X_flat = X.reshape(-1, b)

        # Remove any invalid pixels (NaN, Inf)
        valid_mask = np.isfinite(X_flat).all(axis=1)
        X_clean = X_flat[valid_mask]

        if len(X_clean) == 0:
            raise ValueError("No valid pixels found in the data")

        # Step 1: PCA preprocessing (sphering)
        X_sphered, pca = self._pca_preprocessing(X_clean)

        # Step 2: Iterative projection pursuit
        projection_vectors = []
        projection_scores_list = []
        divergence_scores = []
        pixel_indices = []

        for i in range(self.n_projections):
            # Find best projection orthogonal to previous ones
            proj_vector, div_score, pixel_idx = self._find_best_projection(
                X_sphered, projection_vectors
            )

            # Store results
            projection_vectors.append(proj_vector)
            divergence_scores.append(div_score)
            pixel_indices.append(pixel_idx)

            # Compute projection scores for all sphered data
            proj_scores = X_sphered @ proj_vector
            projection_scores_list.append(proj_scores)

        # Reconstruct full projection scores including invalid pixels
        full_projections = np.zeros((len(X_flat), self.n_projections))
        for i, proj_scores in enumerate(projection_scores_list):
            full_projections[valid_mask, i] = proj_scores

        # Reshape to image format
        features = full_projections.reshape(h, w, self.n_projections)

        return {
            "features": features,
            "n_features": self.n_projections,
            "original_shape": (h, w, b),
            "projection_vectors": np.array(projection_vectors),
            "divergence_scores": divergence_scores,
            "pca_components_used": self.pca_components,
            "pca_model": pca,
            "selected_pixel_indices": pixel_indices,
            "valid_pixel_mask": valid_mask.reshape(h, w),
        }

    def validate(self):
        """Validate extractor parameters."""
        if self.n_projections <= 0:
            raise ValueError("n_projections must be positive")
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.pca_components is not None and self.pca_components <= 0:
            raise ValueError("pca_components must be positive if specified")
