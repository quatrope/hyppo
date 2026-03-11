Advanced Usage
==============

This tutorial covers advanced features of HYPPO including complex extractor dependencies,
parallel execution with different runners, and creating custom extractors.


Complex Extractor Dependencies
-------------------------------

Understanding Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some extractors depend on outputs from other extractors. HYPPO automatically
manages these dependencies:

.. code-block:: python

    from hyppo.core import FeatureSpace
    from hyppo.extractor import PCAExtractor, ICAExtractor

    # Create extractors with dependencies
    config = {
        "pca": (PCAExtractor(n_components=10), {}),
        # ICA depends on PCA output
        "ica": (ICAExtractor(n_components=5), {"pca_features": "pca"})
    }

    fs = FeatureSpace(config)

    # HYPPO resolves execution order automatically
    results = fs.extract(hsi)


Explicit Input Mapping
~~~~~~~~~~~~~~~~~~~~~~~

Define explicit input mappings for complex pipelines:

.. code-block:: python

    from hyppo.extractor import (
        MeanExtractor, StdExtractor,
        PCAExtractor, CompositeExtractor
    )

    config = {
        # Base extractors (no dependencies)
        "mean": (MeanExtractor(), {}),
        "std": (StdExtractor(), {}),

        # PCA on original data
        "pca": (PCAExtractor(n_components=20), {}),

        # Composite extractor using outputs from mean, std, and pca
        "composite": (
            CompositeExtractor(),
            {
                "mean_input": "mean",
                "std_input": "std",
                "pca_input": "pca"
            }
        )
    }

    fs = FeatureSpace(config)


Dependency Graphs
~~~~~~~~~~~~~~~~~

Visualize the dependency graph to understand execution order:

.. code-block:: python

    # Get extractors in topologically sorted order
    extractors = fs.get_extractors()

    print("Execution order:")
    for idx, (name, extractor) in enumerate(extractors):
        deps = extractor.get_input_dependencies()
        if deps:
            print(f"{idx+1}. {name} (depends on: {list(deps.keys())})")
        else:
            print(f"{idx+1}. {name} (no dependencies)")


Parallel Execution
-------------------

Sequential Runner (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default runner executes extractors sequentially:

.. code-block:: python

    from hyppo.runner import SequentialRunner

    # Explicitly use sequential runner
    runner = SequentialRunner()
    results = fs.extract(hsi, runner)


Thread-Based Parallel Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Dask with threads for I/O-bound tasks:

.. code-block:: python

    from hyppo.runner import DaskRunner

    # Create thread-based runner
    runner = DaskRunner.threads(num_threads=8)

    # Extract features in parallel
    results = fs.extract(hsi, runner)

    # Runner automatically handles dependencies and parallelizes
    # independent extractors


Process-Based Parallel Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use process-based parallelism for CPU-intensive tasks:

.. code-block:: python

    # Create process-based runner
    runner = DaskRunner.processes(
        num_workers=4,
        threads_per_worker=2
    )

    # Extract with process-based parallelism
    results = fs.extract(hsi, runner)

    # Best for CPU-bound extractors (PCA, GLCM, etc.)


Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

Compare different runner strategies:

.. code-block:: python

    import time
    from hyppo.runner import SequentialRunner, DaskRunner

    runners = {
        "sequential": SequentialRunner(),
        "threads_4": DaskRunner.threads(num_threads=4),
        "threads_8": DaskRunner.threads(num_threads=8),
        "processes_4": DaskRunner.processes(num_workers=4),
    }

    for name, runner in runners.items():
        start = time.time()
        results = fs.extract(hsi, runner)
        elapsed = time.time() - start
        print(f"{name}: {elapsed:.2f} seconds")



Next Steps
----------

- See :doc:`adding_extractors` for creating custom extractors
- Review :doc:`basic_usage` for fundamentals
- Check :doc:`config` for pipeline management
