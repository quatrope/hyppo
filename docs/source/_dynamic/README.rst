.. FILE AUTO GENERATED !!

hyppo
=====

Hyper Spectral Feature Extractor

Installation
^^^^^^^^^^^^

Install the package:

.. code-block:: bash

   pip install -e .

Usage
^^^^^

HYPPO is used as a Python library for extracting features from
hyperspectral imaging (HSI) data.

.. code-block:: python

    import hyppo
    from hyppo.core import FeatureSpace
    from hyppo.extractor import MeanExtractor, StdExtractor

    # Load hyperspectral image
    hsi = hyppo.io.load_h5_hsi("path_to_file.h5")

    # Configure feature space
    fs = FeatureSpace.from_list([MeanExtractor(), StdExtractor()])

    # Extract features
    results = fs.extract(hsi)

For more details, see the full documentation.
