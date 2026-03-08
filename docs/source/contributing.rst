Contributing
============

This guide covers how to set up a development environment, run tests, and
contribute to HYPPO.


Development Setup
-----------------

Clone the repository and install in development mode:

.. code-block:: bash

    git clone <repository-url>
    cd hyppo
    pip install -e ".[dev]"


Running Tests
-------------

Run the full test suite with pytest:

.. code-block:: bash

    # Run all tests
    pytest

    # Run with coverage report
    pytest --cov=hyppo --cov-report=term-missing

    # Run a specific test file
    pytest tests/extractor/test_ndvi.py

    # Run tests matching a pattern
    pytest -k "test_pca"

Use tox to run tests across configurations:

.. code-block:: bash

    # Run tests in isolated environment
    tox

    # Build documentation
    tox -e mkdocs


Code Standards
--------------

Imports
~~~~~~~

All imports must be at the top of the file, never inside functions.
Organize them in three groups separated by blank lines:

.. code-block:: python

    # Standard library
    import json
    from pathlib import Path

    # Third-party
    import numpy as np

    # Local
    from hyppo.core import HSI
    from .base import Extractor


Docstrings
~~~~~~~~~~

Use NumPy-style docstrings for all public classes and methods:

.. code-block:: python

    class MyExtractor(Extractor):
        """
        Short description of the extractor.

        Longer description with context, algorithm details, and use cases.

        Parameters
        ----------
        param1 : int, default=10
            Description of parameter.

        References
        ----------
        .. [1] Author, A. (Year). Title. Journal.
        """

Use professional, technical language. Avoid informal terms in module
and class docstrings.


Testing
~~~~~~~

Follow the AAA pattern (Arrange, Act, Assert) with descriptive comments:

.. code-block:: python

    def test_extraction_returns_expected_shape(self, sample_hsi):
        # Arrange
        extractor = MyExtractor(n_components=5)

        # Act
        result = extractor.extract(sample_hsi)

        # Assert
        assert result["features"].shape == (10, 10, 5)

Coverage targets:

- **Core modules** (io, core, extractor base): 100%
- **Feature extractors**: 95%+
- **Utilities**: 90%+

See ``tests/CLAUDE.md`` for complete testing guidelines.


Project Structure
-----------------

.. code-block:: text

    hyppo/
    ├── core/                  # Core data structures
    │   ├── _hsi.py            # HSI class
    │   ├── _feature_space/    # FeatureSpace and dependency graph
    │   └── _feature.py        # Feature and FeatureCollection
    ├── extractor/             # Feature extractors
    │   ├── base.py            # Abstract Extractor base class
    │   ├── registry.py        # ExtractorRegistry singleton
    │   ├── ndvi.py            # Example: NDVI extractor
    │   └── ...                # Other extractors
    ├── runner/                # Execution engines
    │   ├── base.py            # Abstract BaseRunner
    │   ├── registry.py        # RunnerRegistry singleton
    │   ├── sequential.py      # SequentialRunner
    │   └── dask.py            # Dask-based runners
    ├── io/                    # Input/output
    │   ├── _config/           # Config loading/saving
    │   ├── _features/         # Feature I/O
    │   └── _hsi/              # HSI loading
    └── __init__.py
    tests/
    ├── core/                  # Core tests
    ├── extractor/             # Extractor tests
    ├── runner/                # Runner tests
    ├── io/                    # I/O tests
    ├── fixtures/              # Shared test fixtures
    └── CLAUDE.md              # Testing conventions
    docs/
    └── source/                # Sphinx documentation source


Adding New Features
-------------------

Extractors
~~~~~~~~~~

See :doc:`tutorial/adding_extractors` for a complete step-by-step guide.

Runners
~~~~~~~

1. Inherit from ``BaseRunner`` (or ``DaskRunner`` for Dask-based runners)
2. Implement ``resolve(data, feature_space) -> FeatureCollection``
3. Register in ``hyppo/runner/__init__.py``
4. Add tests in ``tests/runner/``
5. Add API documentation page in ``docs/source/api/runner/``


Commit Messages
---------------

Write concise commit messages that focus on the *why*, not the *what*:

.. code-block:: text

    # Good
    fix: align test suite with current extractor implementations
    add paper reference validation tests for NDVI, NDWI, SAVI

    # Avoid
    update files
    fix stuff
