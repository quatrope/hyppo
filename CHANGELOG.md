# What's new?

<!-- BODY -->

## Version 0.1.0

First public release.

### Features

- **HSI container** (`hyppo.core.HSI`) with reflectance, wavelengths,
  mask, and metadata. Includes `describe`, `pseudo_rgb`, `crop`, and
  plot accessor.
- **17 feature extractors**:
    - **Dimensionality reduction:** PCA, ICA, MNF
    - **Spectral indices:** NDVI, NDWI, SAVI
    - **Texture:** GLCM, LBP, Gabor
    - **Moments:** GeometricMoment, LegendreMoment, ZernikeMoment
    - **Morphological / projection:** MP (Morphological Profile),
      PP (Projection Pursuit)
    - **Wavelet:** DWT1D, DWT2D, DWT3D
- **4 runners**: `SequentialRunner`, `DaskThreadsRunner`,
  `DaskProcessesRunner`, `LocalProcessRunner` (shared-memory).
- **`FeatureSpace` orchestrator** with NetworkX-based dependency
  graph and topological sort over extractor inputs.
- **I/O**:
    - HDF5 loader for HSI data with heuristic dataset discovery.
    - HDF5 saver for `FeatureCollection`.
    - YAML and JSON config support (`Config`).
- **Extractor registry** for plugin-style discovery.
- **Python 3.11–3.14 support**, including free-threaded builds.
