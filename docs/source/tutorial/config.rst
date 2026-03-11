Configuration Files
===================

This tutorial shows how to use configuration files to define and manage feature extraction
pipelines, making them reusable and shareable.

Why Use Configuration Files?
-----------------------------

Configuration files provide several benefits:

- **Reproducibility**: Share exact extraction pipelines with collaborators
- **Version control**: Track changes to extraction parameters over time
- **Reusability**: Apply the same pipeline to multiple HSI datasets
- **Documentation**: Self-documenting feature extraction workflows
- **No code changes**: Modify extraction parameters without changing code


Loading Configuration Files
----------------------------

From YAML
~~~~~~~~~

YAML is the recommended format for human-readable configurations:

.. code-block:: python

    import hyppo

    # Load FeatureSpace from YAML configuration
    fs = hyppo.io.load_config_yaml("pipeline.yaml")

    # Use the feature space
    hsi = hyppo.io.load_h5_hsi("image.h5")
    results = fs.extract(hsi)


Example YAML Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file named ``pipeline.yaml``:

.. code-block:: yaml

    # Simple extraction pipeline
    extractors:
      - name: mean
        type: MeanExtractor

      - name: std
        type: StdExtractor

      - name: pca_10
        type: PCAExtractor
        params:
          n_components: 10
          whiten: true
          random_state: 42


From JSON
~~~~~~~~~

JSON format is also supported for programmatic generation:

.. code-block:: python

    # Load from JSON
    fs = hyppo.io.load_config_json("pipeline.json")


Example JSON Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file named ``pipeline.json``:

.. code-block:: json

    {
      "extractors": [
        {
          "name": "mean",
          "type": "MeanExtractor"
        },
        {
          "name": "std",
          "type": "StdExtractor"
        },
        {
          "name": "pca_10",
          "type": "PCAExtractor",
          "params": {
            "n_components": 10,
            "whiten": true,
            "random_state": 42
          }
        }
      ]
    }


Saving Configuration Files
---------------------------

Save Existing FeatureSpace
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export a programmatically created FeatureSpace to a configuration file:

.. code-block:: python

    from hyppo.core import FeatureSpace
    from hyppo.extractor import MeanExtractor, StdExtractor, PCAExtractor

    # Create feature space programmatically
    fs = FeatureSpace.from_list([
        MeanExtractor(),
        StdExtractor(),
        PCAExtractor(n_components=10, whiten=True)
    ])

    # Save to YAML
    fs.save_config("my_pipeline.yaml", format="yaml")

    # Or save to JSON
    fs.save_config("my_pipeline.json", format="json")


Reloading Saved Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load the saved configuration
    fs_reloaded = hyppo.io.load_config_yaml("my_pipeline.yaml")

    # Verify it matches the original
    print(fs_reloaded.get_extractors())


Configuration Format Details
-----------------------------

Extractor Parameters
~~~~~~~~~~~~~~~~~~~~

Specify extractor parameters in the configuration:

.. code-block:: yaml

    extractors:
      - name: pca_custom
        type: PCAExtractor
        params:
          n_components: 5
          whiten: false
          random_state: 123

      - name: glcm
        type: GLCMExtractor
        params:
          distances: [1, 2, 3]
          angles: [0, 45, 90, 135]
          properties: ["contrast", "energy", "homogeneity"]


Automatic Dependency Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HYPPO automatically resolves dependencies between extractors:

.. code-block:: yaml

    # Dependencies are resolved automatically
    extractors:
      - name: mean
        type: MeanExtractor

      - name: pca
        type: PCAExtractor
        params:
          n_components: 10

      # This extractor might depend on outputs from previous ones
      - name: composite
        type: CompositeExtractor


Complex Pipelines
-----------------

Multi-Stage Extraction
~~~~~~~~~~~~~~~~~~~~~~

Define multi-stage extraction pipelines:

.. code-block:: yaml

    # pipeline_complex.yaml
    extractors:
      # Stage 1: Basic spectral features
      - name: mean
        type: MeanExtractor

      - name: std
        type: StdExtractor

      # Stage 2: Dimensionality reduction
      - name: pca
        type: PCAExtractor
        params:
          n_components: 20

      # Stage 3: Spatial features
      - name: glcm
        type: GLCMExtractor
        params:
          distances: [1]
          angles: [0, 90]


Using Configuration in Workflows
---------------------------------

Batch Processing with Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply the same configuration to multiple files:

.. code-block:: python

    from pathlib import Path
    import hyppo

    # Load configuration once
    fs = hyppo.io.load_config_yaml("standard_pipeline.yaml")

    # Process multiple HSI files
    input_dir = Path("hsi_data/")
    output_dir = Path("features/")
    output_dir.mkdir(exist_ok=True)

    for h5_file in input_dir.glob("*.h5"):
        # Load HSI
        hsi = hyppo.io.load_h5_hsi(str(h5_file))

        # Extract using configuration
        results = fs.extract(hsi)

        # Save results
        output_file = output_dir / f"{h5_file.stem}_features.h5"
        results.save(str(output_file))


Configuration Versioning
~~~~~~~~~~~~~~~~~~~~~~~~~

Track pipeline versions in filenames:

.. code-block:: python

    # Save versioned configurations
    fs.save_config("pipeline_v1.0.yaml")

    # Later, modify and save new version
    fs_v2 = hyppo.io.load_config_yaml("pipeline_v1.0.yaml")
    # ... modify extractors ...
    fs_v2.save_config("pipeline_v2.0.yaml")


Environment-Specific Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create different configurations for different environments:

.. code-block:: yaml

    # pipeline_development.yaml - Fast, for testing
    extractors:
      - name: mean
        type: MeanExtractor

      - name: pca_small
        type: PCAExtractor
        params:
          n_components: 5

.. code-block:: yaml

    # pipeline_production.yaml - Complete pipeline
    extractors:
      - name: mean
        type: MeanExtractor

      - name: std
        type: StdExtractor

      - name: pca_full
        type: PCAExtractor
        params:
          n_components: 50


Validation and Testing
----------------------

Validate Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check if a configuration is valid before processing:

.. code-block:: python

    try:
        fs = hyppo.io.load_config_yaml("pipeline.yaml")
        print("Configuration is valid!")
        print(f"Extractors: {fs.get_extractors()}")
    except Exception as e:
        print(f"Configuration error: {e}")


Test Configuration on Sample Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load configuration
    fs = hyppo.io.load_config_yaml("new_pipeline.yaml")

    # Test on small HSI sample
    hsi = hyppo.io.load_h5_hsi("test_image_small.h5")

    # Extract and verify
    results = fs.extract(hsi)
    print("Test successful!")
    print(results.describe())


Configuration Examples
-----------------------

.. note::

    All configurations use the ``pipeline`` top-level key with extractor
    class names as registered in the :doc:`../api/extractor/index`.


Spectral Features Only
~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Loading and executing any configuration:

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


Best Practices
--------------

1. **Use descriptive names**: Name extractors clearly (e.g., ``pca_10`` instead of ``pca1``)
2. **Document parameters**: Add comments in YAML to explain non-obvious parameters
3. **Version configurations**: Include version numbers in filenames
4. **Keep configs simple**: Start with basic pipelines and add complexity as needed
5. **Test before deployment**: Always validate configurations on sample data
6. **Use YAML for readability**: Prefer YAML over JSON for human-edited files
7. **Store in version control**: Track configuration changes with git


Next Steps
----------

- See :doc:`basic_usage` for simple extraction workflows
- Learn about :doc:`advanced_usage` for parallel processing and custom extractors
