Runner Comparison
=================

HYPPO provides multiple runner implementations for different execution
strategies. This page helps you choose the right one.


Overview
--------

All runners implement the same ``BaseRunner`` interface and can be swapped
without changing the extraction pipeline.

======================== =========== ============ ============ ==========
Runner                   Parallelism GIL-free     Distributed  Setup
======================== =========== ============ ============ ==========
``SequentialRunner``     None        N/A          No           None
``LocalProcessRunner``   Process     Yes          No           None
``DaskThreadsRunner``    Threads     No           No           Minimal
``DaskProcessesRunner``  Processes   Yes          No           Minimal
``DaskSLURMRunner``      Cluster     Yes          Yes          SLURM
======================== =========== ============ ============ ==========


When to Use Each
----------------

SequentialRunner
~~~~~~~~~~~~~~~~

The default runner. Executes extractors one by one in the main process.

**Use when:**

- Debugging extraction pipelines
- Working with small datasets
- You need deterministic, easy-to-trace execution
- Profiling individual extractors

.. code-block:: python

    from hyppo.runner import SequentialRunner

    runner = SequentialRunner()
    results = fs.extract(hsi, runner)


LocalProcessRunner
~~~~~~~~~~~~~~~~~~

Runs each extractor in a separate local process using ``multiprocessing``.

**Use when:**

- You want process isolation without Dask overhead
- Simple parallelism for independent extractors

.. code-block:: python

    from hyppo.runner import LocalProcessRunner

    runner = LocalProcessRunner()
    results = fs.extract(hsi, runner)


DaskThreadsRunner
~~~~~~~~~~~~~~~~~

Uses a single Dask worker with multiple threads.

**Use when:**

- Extractors are I/O-bound (reading files, network calls)
- Extractors release the GIL (NumPy operations, C extensions)
- You need shared memory between extractors
- Low overhead parallel execution

**Avoid when:**

- Extractors are CPU-bound pure Python code (GIL contention)

.. code-block:: python

    from hyppo.runner import DaskThreadsRunner

    runner = DaskThreadsRunner(num_threads=8)
    results = fs.extract(hsi, runner)


DaskProcessesRunner
~~~~~~~~~~~~~~~~~~~

Uses multiple Dask worker processes with true parallelism.

**Use when:**

- Extractors are CPU-bound (PCA, GLCM, wavelet transforms)
- You have multiple CPU cores available
- Extractors don't need shared memory

**Avoid when:**

- HSI data is very large (data is serialized to each worker)
- You have limited RAM (each worker gets a copy)

.. code-block:: python

    from hyppo.runner import DaskProcessesRunner

    runner = DaskProcessesRunner(
        num_workers=4,
        threads_per_worker=2,
        memory_limit="4GB"
    )
    results = fs.extract(hsi, runner)


DaskSLURMRunner
~~~~~~~~~~~~~~~

Distributes extraction across SLURM cluster nodes.

**Use when:**

- Processing very large datasets that exceed local resources
- Running batch jobs on HPC clusters
- You have access to a SLURM-managed cluster

**Requires:** ``dask-jobqueue`` package and SLURM cluster access.

.. code-block:: python

    from hyppo.runner import DaskSLURMRunner

    runner = DaskSLURMRunner(
        cores=4,
        memory="8GB",
        queue="normal",
        walltime="02:00:00",
        num_jobs=10
    )
    results = fs.extract(hsi, runner)


Decision Flowchart
------------------

.. mermaid::

   flowchart TD
       Start["Choose a Runner"] --> Q1{"Debugging?"}
       Q1 -->|Yes| SEQ["SequentialRunner"]
       Q1 -->|No| Q2{"HPC cluster available?"}
       Q2 -->|Yes| SLURM["DaskSLURMRunner"]
       Q2 -->|No| Q3{"CPU-bound extractors?"}
       Q3 -->|Yes| PROC["DaskProcessesRunner"]
       Q3 -->|No| Q4{"Need parallelism?"}
       Q4 -->|Yes| THREAD["DaskThreadsRunner"]
       Q4 -->|No| SEQ

       style SEQ fill:#4a9eff,color:#fff
       style THREAD fill:#66bb6a,color:#fff
       style PROC fill:#ffa726,color:#fff
       style SLURM fill:#ef5350,color:#fff


Configuration File Usage
-------------------------

Runners can be specified in YAML/JSON configuration files:

.. code-block:: yaml

    pipeline:
      pca:
        extractor: PCAExtractor
        params:
          n_components: 10

    runner:
      type: sequential          # or dask-threads, dask-processes, dask-slurm
      params:
        # Runner-specific parameters here

Available runner type names in configuration files:

==================== ========================
Config name          Runner class
==================== ========================
``sequential``       ``SequentialRunner``
``local``            ``LocalProcessRunner``
``dask-threads``     ``DaskThreadsRunner``
``dask-processes``   ``DaskProcessesRunner``
``dask-slurm``       ``DaskSLURMRunner``
==================== ========================
