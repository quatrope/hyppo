.. FILE AUTO GENERATED !! 

Overview
--------

HYPPO is a modular feature extraction library for hyperspectral images (HSI). It provides a uniform, configurable interface for computing spectral and spatial features commonly used in hyperspectral image classification.

Features
--------


* **17 feature extractors** covering spectral indices, dimensionality reduction, texture, morphological, wavelet, and moment-based methods
* **Automatic dependency resolution** between extractors via directed acyclic graph (DAG)
* **Multiple execution backends**\ : sequential, multi-threaded (Dask), multi-process (Dask), and local multiprocessing
* **HDF5 I/O** for loading hyperspectral images and saving extracted features
* **YAML/JSON configuration** for reproducible extraction pipelines

Installation
------------

Requires Python >= 3.11.

.. code-block:: bash

   git clone https://github.com/quatrope/hyppo.git
   cd hyppo
   pip install -e .

For development:

.. code-block:: bash

   pip install -e ".[dev]"

Quick Start
-----------

.. code-block:: python

   import hyppo

   # Load a hyperspectral image
   hsi = hyppo.io.load_h5_hsi("image.h5")

   # Define a feature space
   from hyppo.extractor import PCAExtractor, NDVIExtractor
   from hyppo.core import FeatureSpace

   fs = FeatureSpace.from_list([
       PCAExtractor(n_components=10),
       NDVIExtractor(),
   ])

   # Extract features
   results = fs.extract(hsi)

   # Save results
   results.save("output.h5")

Using a Configuration File
--------------------------

Define your pipeline in YAML:

.. code-block:: yaml

   pipeline:
     pca:
       extractor: PCAExtractor
       params:
         n_components: 10
     ndvi:
       extractor: NDVIExtractor

   runner:
     type: dask-threads
     params:
       num_threads: 4

Then load and run:

.. code-block:: python

   config = hyppo.io.load_config_yaml("pipeline.yaml")
   results = config.feature_space.extract(hsi, config.runner)

Available Extractors
--------------------

.. list-table::
   :header-rows: 1

   * - Category
     - Extractors
   * - Spectral indices
     - NDVI, NDWI, SAVI
   * - Dimensionality reduction
     - PCA, ICA, MNF, Projection Pursuit
   * - Texture
     - GLCM, LBP, Gabor
   * - Morphological
     - Morphological Profiles (MP)
   * - Wavelet
     - DWT1D, DWT2D, DWT3D
   * - Moment-based
     - Geometric, Legendre, Zernike


Execution Backends
------------------

.. list-table::
   :header-rows: 1

   * - Runner
     - Strategy
   * - ``SequentialRunner``
     - Single-threaded, topological order
   * - ``DaskThreadsRunner``
     - Dask with thread-based parallelism
   * - ``DaskProcessesRunner``
     - Dask with process-based parallelism
   * - ``LocalProcessRunner``
     - multiprocessing with shared memory


License
-------

HYPPO is distributed under the BSD 3-Clause License.
