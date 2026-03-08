Troubleshooting
===============

This page covers common issues and their solutions when working with HYPPO.


Data Loading Issues
-------------------

HDF5 file not recognized
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``ValueError: Could not detect reflectance dataset`` when loading
an H5 file.

**Cause:** HYPPO uses keyword-based heuristics to locate reflectance data
inside HDF5 files. If the dataset paths don't contain expected keywords
(e.g., ``reflectance``, ``radiance``), detection fails.

**Solution:** Inspect the file structure and specify the dataset path directly:

.. code-block:: python

    import h5py

    # Check the file structure
    with h5py.File("image.h5", "r") as f:
        f.visititems(lambda name, obj: print(name, type(obj)))

    # Load with explicit path if needed
    hsi = hyppo.io.load_h5_hsi("image.h5", reflectance_path="/custom/path")


Wavelength metadata missing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``ValueError: No wavelength information available`` during
extraction.

**Cause:** The H5 file does not contain a wavelength array, or it was not
detected by the heuristic loader.

**Solution:** Verify the file contains wavelength data. If wavelengths are
stored under a non-standard path, provide it manually when loading.


Memory errors with large images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``MemoryError`` or system slowdown when loading large HSI cubes.

**Solution:**

- Verify available RAM is sufficient for the image size
  (``rows × cols × bands × 8 bytes`` for float64)
- Use process-based runners to distribute memory across workers:

  .. code-block:: python

      from hyppo.runner import DaskProcessesRunner

      runner = DaskProcessesRunner(num_workers=4, memory_limit="4GB")
      results = fs.extract(hsi, runner)


Configuration Issues
--------------------

Unknown extractor type
~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``ValueError: Unknown extractor type: MyExtractor``

**Cause:** The extractor class name in the configuration does not match any
registered extractor.

**Solution:** Check the exact class name. Use the registry to list available
extractors:

.. code-block:: python

    from hyppo.extractor import registry

    print(registry.list_extractors())


Invalid extractor parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``ValueError: Failed to instantiate PCAExtractor with parameters``

**Cause:** The ``params`` dictionary in the configuration contains parameter
names or values that the extractor constructor does not accept.

**Solution:** Check the extractor's API documentation for valid parameters:

.. code-block:: python

    help(PCAExtractor.__init__)


Missing 'pipeline' field in config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``ValueError: Required field 'pipeline' missing from configuration``

**Cause:** The configuration file uses an incorrect top-level key.

**Solution:** Ensure the config uses ``pipeline`` as the top-level key:

.. code-block:: yaml

    # Correct
    pipeline:
      mean:
        extractor: MeanExtractor

    # Wrong
    extractors:
      - name: mean
        type: MeanExtractor


Extraction Issues
-----------------

Circular dependency detected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Error during FeatureSpace construction about circular dependencies.

**Cause:** Two or more extractors depend on each other, creating a cycle that
cannot be resolved.

**Solution:** Review your extractor dependencies and restructure them to form
a directed acyclic graph (DAG). See :doc:`architecture` for details on how
dependency resolution works.


Band wavelength warnings
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``UserWarning: Bands far from target wavelengths``

**Cause:** Spectral index extractors (NDVI, NDWI, SAVI) require specific
wavelength bands. If the closest available band is more than 50nm from the
target, a warning is issued.

**Solution:** This is a warning, not an error. The extractor will still run
using the closest available band. If accuracy is critical, verify your data
covers the required spectral range:

- NDVI: Red (~660nm) and NIR (~850nm)
- NDWI: Green (~560nm) and NIR (~850nm)
- SAVI: Red (~660nm) and NIR (~850nm)


Runner Issues
-------------

Dask workers failing silently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``None`` results in FeatureCollection, or missing extractors
in output.

**Solution:** Enable verbose logging to diagnose:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)

    runner = DaskProcessesRunner(num_workers=2)
    results = fs.extract(hsi, runner)


dask-jobqueue not installed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``ImportError: dask-jobqueue is required for SLURM execution``

**Solution:** Install the optional dependency:

.. code-block:: bash

    pip install dask-jobqueue


Port conflicts with Dask dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``OSError: Address already in use`` when creating a Dask runner.

**Cause:** Another Dask cluster or process is using the same port.

**Solution:** Ensure previous Dask clusters are properly closed, or specify
a different dashboard port in the cluster configuration.


Performance Tips
----------------

Choosing the right runner
~~~~~~~~~~~~~~~~~~~~~~~~~~

================= ====================== ==========================
Runner            Best for               Avoid when
================= ====================== ==========================
Sequential        Debugging, small data  Large datasets
DaskThreads       I/O-bound extractors   CPU-bound extractors
DaskProcesses     CPU-bound extractors   Very large shared data
DaskSLURM         HPC cluster workloads  Local development
================= ====================== ==========================


Reducing memory usage
~~~~~~~~~~~~~~~~~~~~~~

- Use fewer PCA/ICA components than spectral bands
- Process files sequentially rather than loading all into memory
- Set ``memory_limit`` on Dask runners to prevent OOM crashes
- Delete intermediate results when no longer needed:

  .. code-block:: python

      results = fs.extract(hsi, runner)
      results.save("output.h5")
      del hsi, results


Getting Help
------------

If your issue is not covered here:

1. Check the :doc:`api/index` for detailed class documentation
2. Review error messages carefully — they often indicate the exact problem
3. Enable debug logging to get more context
4. Open an issue on the project repository with a minimal reproducible example
