Tutorials
=========

This section contains step-by-step tutorials demonstrating how to use HYPPO
for hyperspectral image feature extraction.

Getting Started
---------------

If you're new to HYPPO, start with the basic usage tutorial to learn the fundamentals.

.. toctree::
   :maxdepth: 1

   basic_usage


Working with Data
-----------------

Learn how to load hyperspectral images and save extraction results.

.. toctree::
   :maxdepth: 1

   hsi_io


Configuration Management
------------------------

Use configuration files to define reusable extraction pipelines.

.. toctree::
   :maxdepth: 1

   config


Advanced Topics
---------------

Explore complex extractor dependencies, parallel processing, and distributed computing.

.. toctree::
   :maxdepth: 1

   advanced_usage


Configuration Examples
----------------------

Ready-to-use YAML configurations for common pipelines.

.. toctree::
   :maxdepth: 1

   config_examples


Extending HYPPO
---------------

Learn how to create custom feature extractors.

.. toctree::
   :maxdepth: 1

   adding_extractors


Tutorial Overview
-----------------

:doc:`basic_usage`
    Learn the fundamentals of HYPPO including loading HSI data, creating feature spaces,
    and extracting features. Perfect for beginners.

:doc:`hsi_io`
    Master loading HSI from HDF5 files, working with wavelength metadata, saving results,
    and batch processing multiple files.

:doc:`config`
    Discover how to use YAML and JSON configuration files to define extraction pipelines,
    making them reproducible and shareable.

:doc:`advanced_usage`
    Explore advanced features including complex extractor dependencies, parallel execution
    with Dask, and distributed computing with SLURM clusters.

:doc:`config_examples`
    Ready-to-use YAML configuration files for spectral, spatial, wavelet,
    morphological, and full extraction pipelines.

:doc:`adding_extractors`
    Step-by-step guide to creating, registering, documenting, and testing
    a new feature extractor.


Prerequisites
-------------

Before starting these tutorials, you should have:

- Python 3.11 or higher installed
- HYPPO installed (see :doc:`../install`)
- Basic knowledge of Python programming
- Familiarity with NumPy arrays (helpful but not required)

For advanced tutorials, you may also need:

- Access to a SLURM cluster (for distributed computing)
- Understanding of parallel computing concepts
- Knowledge of HDF5 file format


Example Datasets
----------------

The tutorials reference example HSI data. You can use your own data or obtain
sample datasets from:

- Public hyperspectral image repositories
- Remote sensing data archives
- Your institution's data sharing platform

Make sure your HSI data is in HDF5 format with reflectance or radiance values.


Learning Path
-------------

We recommend following this learning path:

1. **Start here**: :doc:`basic_usage` - Get comfortable with HYPPO basics
2. **Next**: :doc:`hsi_io` - Learn data loading and saving
3. **Then**: :doc:`config` - Master configuration management
4. **Finally**: :doc:`advanced_usage` - Explore advanced features

Each tutorial builds on concepts from previous ones, so following this order
will provide the smoothest learning experience.


Getting Help
------------

If you encounter issues while following these tutorials:

- Check the :doc:`../api/index` for detailed API documentation
- Review error messages carefully - they often contain helpful information
- Verify your data format matches the expected structure
- Ensure all dependencies are properly installed


Next Steps
----------

Ready to get started? Head over to :doc:`basic_usage` to begin your journey
with HYPPO!
