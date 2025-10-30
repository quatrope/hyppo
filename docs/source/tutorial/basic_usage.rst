Basic Usage
===========

This tutorial covers the basic usage of HYPPO for extracting features from hyperspectral images.

Quick Start
-----------

The simplest way to use HYPPO is to load a hyperspectral image, define a feature space,
and extract features:

.. code-block:: python

    import hyppo
    from hyppo.core import FeatureSpace
    from hyppo.extractor import MeanExtractor, StdExtractor

    # Load hyperspectral image
    hsi = hyppo.io.load_h5_hsi("path/to/your/image.h5")

    # Create feature space with extractors
    fs = FeatureSpace.from_list([
        MeanExtractor(),
        StdExtractor()
    ])

    # Extract features
    results = fs.extract(hsi)

    # Access results
    print(results.get_all_features())
    print(results.describe())


Understanding the HSI Object
-----------------------------

The ``HSI`` object represents a hyperspectral image with its reflectance data and metadata:

.. code-block:: python

    # HSI properties
    print(f"Shape: {hsi.shape}")           # (height, width, bands)
    print(f"Height: {hsi.height}")         # Image height in pixels
    print(f"Width: {hsi.width}")           # Image width in pixels
    print(f"Bands: {hsi.n_bands}")         # Number of spectral bands

    # Access specific band
    band_data = hsi.get_band(10)           # Get 10th band

    # Get pixel spectrum
    spectrum = hsi.get_pixel_spectrum(100, 200)  # Pixel at (100, 200)


Available Extractors
--------------------

HYPPO provides several built-in extractors for different types of features:

Spectral Extractors
~~~~~~~~~~~~~~~~~~~

Extract features from the spectral dimension:

.. code-block:: python

    from hyppo.extractor import (
        MeanExtractor,      # Mean reflectance across bands
        StdExtractor,       # Standard deviation
        PCAExtractor,       # Principal Component Analysis
    )

    # PCA with custom parameters
    pca = PCAExtractor(n_components=10, whiten=True)


Creating a Feature Space
-------------------------

Method 1: Using from_list (Recommended for Simple Cases)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way when extractors have no dependencies:

.. code-block:: python

    fs = FeatureSpace.from_list([
        MeanExtractor(),
        StdExtractor(),
        PCAExtractor(n_components=5)
    ])


Method 2: Using Dictionary Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For explicit control over extractor names and inputs:

.. code-block:: python

    config = {
        "mean": (MeanExtractor(), {}),
        "std": (StdExtractor(), {}),
        "pca_5": (PCAExtractor(n_components=5), {})
    }
    
    fs = FeatureSpace(config)


Extracting Features
--------------------

Default Sequential Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, features are extracted sequentially:

.. code-block:: python

    results = fs.extract(hsi)
    
    # Results is a FeatureCollection object
    print(f"Extracted features: {results.get_extractor_names()}")
    
    # Access specific extractor results
    mean_features = results["mean"]
    print(mean_features)


Working with Results
--------------------

The ``FeatureCollection`` object provides several methods to access results:

.. code-block:: python

    # Get all features as a dictionary
    all_features = results.get_all_features()
    
    # Get list of extractor names
    extractors = results.get_extractor_names()
    
    # Access individual extractor results
    for name in extractors:
        features = results[name]
        print(f"{name}: {type(features)}")
    
    # Get summary description
    summary = results.describe()
    print(summary)
    
    # Convert to dictionary for custom processing
    results_dict = results.to_dict()


Next Steps
----------

- Learn about :doc:`hsi_io` for loading and saving
- Explore :doc:`config` for configuration files
- Check :doc:`advanced_usage` for complex workflows
