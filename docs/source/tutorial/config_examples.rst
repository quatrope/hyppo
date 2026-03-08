Configuration Examples
======================

This page provides ready-to-use configuration file examples for common
feature extraction pipelines.

.. note::

    All configurations use the ``pipeline`` top-level key with extractor
    class names as registered in the :doc:`../api/extractor/index`.


Spectral Features Only
-----------------------

Basic spectral statistics and indices:

.. code-block:: yaml

    # spectral_pipeline.yaml
    pipeline:
      mean:
        extractor: MeanExtractor

      std:
        extractor: StdExtractor

      ndvi:
        extractor: NDVIExtractor
        params:
          red_wavelength: 660
          nir_wavelength: 850

      ndwi:
        extractor: NDWIExtractor

      savi:
        extractor: SAVIExtractor
        params:
          L: 0.5


Dimensionality Reduction
-------------------------

PCA, ICA, and MNF for reducing spectral bands:

.. code-block:: yaml

    # reduction_pipeline.yaml
    pipeline:
      pca:
        extractor: PCAExtractor
        params:
          n_components: 20
          whiten: true
          random_state: 42

      ica:
        extractor: ICAExtractor
        params:
          n_components: 10
          random_state: 42

      mnf:
        extractor: MNFExtractor
        params:
          n_components: 15


Spatial Texture Features
------------------------

Texture and spatial pattern extractors:

.. code-block:: yaml

    # spatial_pipeline.yaml
    pipeline:
      glcm:
        extractor: GLCMExtractor
        params:
          distances:
            - 1
            - 2
          angles:
            - 0
            - 45
            - 90
            - 135

      lbp:
        extractor: LBPExtractor

      gabor:
        extractor: GaborExtractor


Wavelet Decomposition
---------------------

Multi-resolution analysis with wavelets:

.. code-block:: yaml

    # wavelet_pipeline.yaml
    pipeline:
      dwt1d:
        extractor: DWT1DExtractor

      dwt2d:
        extractor: DWT2DExtractor

      dwt3d:
        extractor: DWT3DExtractor


Morphological Profiles
-----------------------

Spatial features based on mathematical morphology:

.. code-block:: yaml

    # morphological_pipeline.yaml
    pipeline:
      pca:
        extractor: PCAExtractor
        params:
          n_components: 5

      mp:
        extractor: MPExtractor

      pp:
        extractor: PPExtractor


Full Pipeline
-------------

A comprehensive pipeline combining spectral, spatial, and reduction features:

.. code-block:: yaml

    # full_pipeline.yaml
    #
    # Complete feature extraction pipeline
    # Combines spectral indices, dimensionality reduction,
    # and spatial texture features.

    pipeline:
      # Spectral indices
      ndvi:
        extractor: NDVIExtractor
      ndwi:
        extractor: NDWIExtractor
      savi:
        extractor: SAVIExtractor

      # Dimensionality reduction
      pca:
        extractor: PCAExtractor
        params:
          n_components: 20
          whiten: true
          random_state: 42

      ica:
        extractor: ICAExtractor
        params:
          n_components: 10

      # Spatial texture
      glcm:
        extractor: GLCMExtractor
        params:
          distances:
            - 1
          angles:
            - 0
            - 90

      lbp:
        extractor: LBPExtractor

      # Morphological
      mp:
        extractor: MPExtractor

    # Runner configuration (optional, defaults to sequential)
    runner:
      type: dask-threads
      params:
        num_threads: 8


With Runner Configuration
--------------------------

Pipelines can include runner configuration for execution control:

.. code-block:: yaml

    # pipeline_with_runner.yaml
    pipeline:
      pca:
        extractor: PCAExtractor
        params:
          n_components: 10

      glcm:
        extractor: GLCMExtractor

    # Sequential (default)
    runner:
      type: sequential

.. code-block:: yaml

    # pipeline_parallel.yaml
    pipeline:
      pca:
        extractor: PCAExtractor
        params:
          n_components: 10

      glcm:
        extractor: GLCMExtractor

    # Process-based parallelism
    runner:
      type: dask-processes
      params:
        num_workers: 4
        threads_per_worker: 2

.. code-block:: yaml

    # pipeline_slurm.yaml
    pipeline:
      pca:
        extractor: PCAExtractor
        params:
          n_components: 10

    runner:
      type: dask-slurm
      params:
        cores: 4
        memory: "8GB"
        queue: normal
        walltime: "02:00:00"
        num_jobs: 10


Usage
-----

Load and execute any configuration:

.. code-block:: python

    import hyppo

    # Load configuration
    config = hyppo.io.load_config_yaml("full_pipeline.yaml")

    # Load data
    hsi = hyppo.io.load_h5_hsi("image.h5")

    # Extract features using the configured runner
    results = config.feature_space.extract(hsi, config.runner)

    # Save results
    hyppo.io.save_features_h5(results, "output_features.h5")
