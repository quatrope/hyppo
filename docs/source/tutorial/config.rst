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


Best Practices
--------------

1. **Use descriptive names**: Name extractors clearly (e.g., ``pca_10`` instead of ``pca1``)
2. **Document parameters**: Add comments in YAML to explain non-obvious parameters
3. **Version configurations**: Include version numbers in filenames
4. **Keep configs simple**: Start with basic pipelines and add complexity as needed
5. **Test before deployment**: Always validate configurations on sample data
6. **Use YAML for readability**: Prefer YAML over JSON for human-edited files
7. **Store in version control**: Track configuration changes with git

Example Production Configuration
---------------------------------

A complete, documented pipeline configuration:

.. code-block:: yaml

    # production_pipeline_v1.0.yaml
    # 
    # Standard feature extraction pipeline for hyperspectral image analysis
    # 
    # Author: Your Name
    # Date: 2025-01-30
    # Description: Extracts spectral statistics, PCA features, and spatial texture
    
    extractors:
      # Spectral statistics
      - name: spectral_mean
        type: MeanExtractor
        # Mean reflectance across all bands
      
      - name: spectral_std
        type: StdExtractor
        # Standard deviation across bands
      
      # Dimensionality reduction
      - name: pca_components
        type: PCAExtractor
        params:
          n_components: 30
          whiten: true
          random_state: 42
        # PCA for dimensionality reduction while preserving 95% variance
      
      # Spatial texture features
      - name: texture_glcm
        type: GLCMExtractor
        params:
          distances: [1, 2]
          angles: [0, 45, 90, 135]
          properties: ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
        # Gray-Level Co-occurrence Matrix texture features


Next Steps
----------

- See :doc:`basic_usage` for simple extraction workflows
- Learn about :doc:`advanced_usage` for parallel processing with configs
- Check :doc:`hsi_io` for batch processing patterns
