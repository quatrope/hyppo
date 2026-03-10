"""Projection Pursuit feature extractor for hyperspectral images."""

import warnings

import numpy as np
from sklearn.decomposition import PCA

from hyppo.core import HSI
from .base import Extractor


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
    n_bins : int, default=128
        Number of histogram bins for divergence computation.
    pca_components : int or None, default=None
        Number of PCA components for sphering. If None, components
        are auto-selected to preserve 99% variance.
    sample_size : int, default=1000
        Number of pixels to sample when searching for projections.
    random_state : int, optional, default=42
        Random state for reproducibility.

    References
    ----------
    .. [1] Ifarraguerri, A., & Chang, C. I. (2000). Unsupervised hyperspectral
    image analysis with projection pursuit. IEEE Transactions on Geoscience
    and Remote Sensing, 38(6), 2529-2538.
    """

    def __init__(
        self,
        n_projections=10,
        n_bins=128,
        pca_components=None,  # If None, auto-select based on 99% variance
        sample_size=1000,  # Number of pixels to sample for efficiency
        random_state=42,
    ):
        """Initialize PP extractor with projection search parameters."""
        super().__init__()
        self.n_projections = n_projections
        self.n_bins = n_bins
        self.pca_components = pca_components
        self.sample_size = sample_size
        self.rng = np.random.default_rng(random_state)

    @classmethod
    def feature_name(cls) -> str:
        """Return the feature name."""
        return "pp"

    def _compute_information_divergence(self, projection_scores):
        """Symmetric KL divergence from standard Gaussian."""
        # Standardize scores (mean=0, std=1)
        scores_std = (projection_scores - np.mean(projection_scores)) / (
            np.std(projection_scores) + 1e-10
        )

        # Create histogram
        max_abs = np.max(np.abs(scores_std))
        limit = np.maximum(4.0, max_abs + 0.5)
        hist_range = (-limit, limit)
        hist, bin_edges = np.histogram(
            scores_std, bins=self.n_bins, range=hist_range, density=True
        )

        # Normalize histogram to get probability distribution p
        bin_width = bin_edges[1] - bin_edges[0]
        p = hist * bin_width
        p = p / (np.sum(p) + 1e-10)

        # Create corresponding Gaussian distribution q
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        q = np.exp(-0.5 * bin_centers**2) / np.sqrt(2 * np.pi)
        q = q * bin_width
        q = q / (np.sum(q) + 1e-10)

        # Compute divergence (Symmetric Kullback-Leibler)
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)

        divergence = np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))

        return divergence

    def _pca_preprocessing(self, X):
        """Apply PCA + whitening preprocessing ("sphering")."""
        pca_components = self.pca_components

        # Determine number of components if not specified
        if pca_components is None:
            pca_temp = PCA()
            pca_temp.fit(X)
            cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
            pca_components = np.argmax(cumvar >= 0.99) + 1

        # Apply PCA with sphering
        pca = PCA(n_components=pca_components, whiten=True)
        X_sphered = pca.fit_transform(X)

        return X_sphered, pca, pca_components

    def _find_best_projection(self, X):
        """
        Find projection vector maximizing information divergence.
        Candidate vectors are sampled directly from the data points.
        """
        n_samples, n_features = X.shape

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
            norm = np.linalg.norm(candidate_vector)
            if norm < 1e-10:
                continue

            candidate_vector = candidate_vector / norm
            projection_scores = X @ candidate_vector

            # Compute information divergence
            try:
                score = self._compute_information_divergence(projection_scores)
                if score > best_score:
                    best_score = score
                    best_projection = candidate_vector.copy()
                    best_pixel_idx = sample_indices[i]
            except (ValueError, FloatingPointError):
                continue  # Skip if divergence computation fails

        if best_projection is None:
            warnings.warn("PP: fallback to random projection vector")
            candidate_vector = self.rng.standard_normal(n_features)
            best_projection = candidate_vector / (
                np.linalg.norm(candidate_vector) + 1e-10
            )
            best_score = 0.0
            best_pixel_idx = -1

        return best_projection, best_score, best_pixel_idx

    def _extract(self, data: HSI, **inputs):
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
        X = data.reflectance
        h, w, b = X.shape
        X_flat = X.reshape(-1, b)

        # Remove any invalid pixels (NaN, Inf)
        valid_mask = np.isfinite(X_flat).all(axis=1)
        X_clean = X_flat[valid_mask]

        if len(X_clean) == 0:
            raise ValueError("No valid pixels found in the data")

        # Step 1: PCA preprocessing (sphering)
        X_sphered, pca, pca_components_used = self._pca_preprocessing(X_clean)

        # Working copy for deflation (paper eq. 9–10)
        X_work = X_sphered.copy()

        # Step 2: Iterative projection pursuit
        projection_vectors = []
        projection_scores_list = []
        divergence_scores = []
        pixel_indices = []

        for i in range(self.n_projections):
            # Find best projection in deflated space
            proj_vector, div_score, pixel_idx = self._find_best_projection(
                X_work
            )

            # Store results
            projection_vectors.append(proj_vector)
            divergence_scores.append(div_score)
            pixel_indices.append(pixel_idx)

            # Compute projection scores for all sphered data
            proj_scores = X_sphered @ proj_vector
            projection_scores_list.append(proj_scores)

            # Deflation: project X onto orthogonal subspace
            # Deflation: X <- (I - vvᵀ) X
            projection_component = np.outer(X_work @ proj_vector, proj_vector)
            X_work = X_work - projection_component

        # Reconstruct image
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
            "pca_components_used": pca_components_used,
            "pca_model": pca,
            "selected_pixel_indices": pixel_indices,
            "valid_pixel_mask": valid_mask.reshape(h, w),
        }

    def _validate(self, data: HSI, **inputs):
        """Validate extractor parameters."""
        if self.n_projections <= 0:
            raise ValueError("n_projections must be positive")
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.pca_components is not None and self.pca_components <= 0:
            raise ValueError("pca_components must be positive if specified")
