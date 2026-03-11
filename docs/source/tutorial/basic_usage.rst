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


Loading HSI Data
-----------------

From HDF5 Files
~~~~~~~~~~~~~~~~

HYPPO currently supports loading HSI data from HDF5 (.h5) files with automatic
detection of reflectance and radiance data:

.. code-block:: python

    import hyppo

    # Load HSI from H5 file
    hsi = hyppo.io.load_h5_hsi("path/to/image.h5")

    # The loader automatically detects:
    # - Reflectance data (3D arrays with scale factors)
    # - Radiance data (convertible to reflectance)
    # - Wavelength metadata

    # Inspect loaded data
    print(f"Image shape: {hsi.shape}")
    print(f"Reflectance range: [{hsi.reflectance.min()}, {hsi.reflectance.max()}]")
    print(f"Available wavelengths: {len(hsi.wavelength)} bands")


Understanding H5 File Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HYPPO uses heuristics to detect the required datasets in H5 files:

- **Reflectance datasets**: Look for keywords like 'reflectance', 'refl', 'Reflectance_Data'
- **Radiance datasets**: Look for keywords like 'radiance', 'rdn', 'Radiance_Data'
- **Wavelength datasets**: Look for 'wavelength', 'Wavelength', 'wvl'
- **Scale factors**: Automatically applied when present (e.g., 'Reflectance_Scale_Factor')

Example H5 file structure:

.. code-block:: text

    image.h5
    ├── Reflectance_Data (3D array: height × width × bands)
    ├── Wavelength (1D array: wavelength per band)
    ├── Reflectance_Scale_Factor (scalar)
    └── Null_Value (scalar for masking invalid pixels)


Working with Wavelength Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Access wavelength metadata from loaded HSI:

.. code-block:: python

    # Get all wavelengths
    wavelengths = hsi.wavelength
    print(f"Wavelength range: {wavelengths[0]} - {wavelengths[-1]} nm")

    # Find bands within a specific range
    red_bands = hsi.get_band_indices(620, 700)  # Red spectrum (620-700 nm)
    nir_bands = hsi.get_band_indices(700, 1000) # Near-infrared

    print(f"Red bands: {red_bands}")
    print(f"NIR bands: {nir_bands}")


Saving Extraction Results
--------------------------

To HDF5 Format
~~~~~~~~~~~~~~

Save feature extraction results to HDF5 for later analysis:

.. code-block:: python

    from hyppo.core import FeatureSpace
    from hyppo.extractor import MeanExtractor, StdExtractor, PCAExtractor

    # Load HSI and extract features
    hsi = hyppo.io.load_h5_hsi("input_image.h5")

    fs = FeatureSpace.from_list([
        MeanExtractor(),
        StdExtractor(),
        PCAExtractor(n_components=5)
    ])

    results = fs.extract(hsi)

    # Save results to H5 file
    results.save("output_features.h5")


Saved File Structure
~~~~~~~~~~~~~~~~~~~~

The saved H5 file contains all extracted features organized by extractor:

.. code-block:: text

    output_features.h5
    ├── mean/
    │   └── mean (array with extracted features)
    ├── std/
    │   └── std (array with extracted features)
    └── pca_5/
        ├── features (PCA-transformed array)
        ├── explained_variance_ratio (array)
        ├── explained_variance (array)
        ├── components (array)
        └── ... (other PCA outputs)


Loading Saved Results
~~~~~~~~~~~~~~~~~~~~~~

Load previously saved results for further processing:

.. code-block:: python

    from hyppo.core import FeatureCollection

    # Load features from H5 file
    loaded_results = hyppo.io.load_h5_features("output_features.h5")

    # Access loaded features
    print(loaded_results.get_extractor_names())
    print(loaded_results.describe())

    # Continue processing
    mean_features = loaded_results["mean"]


Batch Processing Multiple Files
---------------------------------

Process multiple HSI files efficiently:

.. code-block:: python

    import os
    from pathlib import Path

    # Define input and output directories
    input_dir = Path("hsi_data/")
    output_dir = Path("features/")
    output_dir.mkdir(exist_ok=True)

    # Configure feature space once
    fs = FeatureSpace.from_list([
        MeanExtractor(),
        StdExtractor()
    ])

    # Process all H5 files
    for h5_file in input_dir.glob("*.h5"):
        print(f"Processing {h5_file.name}...")

        # Load HSI
        hsi = hyppo.io.load_h5_hsi(str(h5_file))

        # Extract features
        results = fs.extract(hsi)

        # Save with same name in output directory
        output_file = output_dir / f"{h5_file.stem}_features.h5"
        results.save(str(output_file))

        print(f"  Saved to {output_file}")


Working with Masked Data
-------------------------

Handle invalid or masked pixels in HSI data:

.. code-block:: python

    # Load HSI (automatically handles null values)
    hsi = hyppo.io.load_h5_hsi("image_with_nulls.h5")

    # Get masked data (returns numpy masked array)
    masked_data = hsi.get_masked_data()

    # Check for masked pixels
    n_masked = masked_data.mask.sum()
    print(f"Number of masked pixels: {n_masked}")

    # Extract features (extractors handle masked data)
    results = fs.extract(hsi)


Next Steps
----------

- Explore :doc:`config` for configuration files
- Check :doc:`advanced_usage` for complex workflows
