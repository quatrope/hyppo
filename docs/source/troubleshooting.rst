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
- Use ``LocalProcessRunner``, which shares HSI data across processes via
  shared memory instead of serializing copies to each worker:

  .. code-block:: python

      from hyppo.runner import LocalProcessRunner

      runner = LocalProcessRunner()
      results = fs.extract(hsi, runner)

- Avoid ``DaskProcessesRunner`` for large images, as it serializes the
  full HSI cube to each worker process
- Delete intermediate results when no longer needed


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
a directed acyclic graph (DAG).


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


Performance Tips
----------------

Choosing the right runner
~~~~~~~~~~~~~~~~~~~~~~~~~~

==================== ====================== ==========================
Runner               Best for               Avoid when
==================== ====================== ==========================
Sequential           Debugging, small data  Large datasets
LocalProcess         Large images (shared    Few extractors
                     memory), CPU-bound
DaskThreads          I/O-bound extractors   CPU-bound extractors
DaskProcesses        CPU-bound extractors   Very large shared data
==================== ====================== ==========================


Reducing memory usage
~~~~~~~~~~~~~~~~~~~~~~

- Process files sequentially rather than loading all into memory
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
