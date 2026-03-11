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

Use tox to run tests and coverage:

.. code-block:: bash

    # Run tests with coverage
    tox -e py311

    # Coverage report (requires py311 first)
    tox -e coverage

    # Run all environments
    tox


Code Standards
--------------

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

Coverage target: 100% across all modules.



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
    └── conftest.py            # Shared test fixtures
    docs/
    └── source/                # Sphinx documentation source


Adding New Extractors
---------------------

See :doc:`tutorial/adding_extractors` for how to create the extractor class
itself. This section covers the steps to integrate it into the project.

Register the Extractor
~~~~~~~~~~~~~~~~~~~~~~

Add your extractor to ``hyppo/extractor/__init__.py``:

.. code-block:: python

    from .myfeature import MyFeatureExtractor

    # Add to __all__
    __all__ = [
        # ... existing entries ...
        "MyFeatureExtractor",
    ]

    # Register with the registry
    registry.register(MyFeatureExtractor)

After registration, the extractor is available by class name in configuration
files:

.. code-block:: yaml

    pipeline:
      my_feature:
        extractor: MyFeatureExtractor
        params:
          param1: 20


Add Documentation
~~~~~~~~~~~~~~~~~

Create an API documentation page at ``docs/source/api/extractor/myfeature.rst``.
Follow the same pattern as existing pages (e.g., ``api/extractor/pca.rst``):
use the ``automodule`` directive with ``:members:``, ``:undoc-members:``, and
``:show-inheritance:`` options. Docstrings in the source code are the single
source of truth.


Write Tests
~~~~~~~~~~~

Create tests in ``tests/extractor/test_myfeature.py``:

.. code-block:: python

    import numpy as np
    import pytest

    from hyppo.core import HSI
    from hyppo.extractor import MyFeatureExtractor


    class TestMyFeatureExtractor:

        def test_basic_extraction(self, sample_hsi):
            extractor = MyFeatureExtractor()
            result = extractor.extract(sample_hsi)

            assert "features" in result
            assert result["features"].ndim == 3

        def test_custom_params(self, sample_hsi):
            extractor = MyFeatureExtractor(param1=20)
            result = extractor.extract(sample_hsi)
            assert "features" in result

        def test_invalid_param(self, sample_hsi):
            extractor = MyFeatureExtractor(param1=-1)
            with pytest.raises(ValueError, match="param1 must be positive"):
                extractor.extract(sample_hsi)

See ``tests/extractor/test_base.py`` for additional test patterns.


Feature Naming
~~~~~~~~~~~~~~

The ``feature_name()`` class method auto-generates a name from the class name
by splitting on CamelCase boundaries:

- ``MyFeatureExtractor`` → ``my_feature``
- ``NDVIExtractor`` → ``ndvi``
- ``DWT1DExtractor`` → ``d_w_t1_d`` (acronyms may need override)

Override ``feature_name()`` if the auto-generated name is not suitable:

.. code-block:: python

    @classmethod
    def feature_name(cls) -> str:
        return "my_custom_name"


Extractor Checklist
~~~~~~~~~~~~~~~~~~~

Before submitting a new extractor, verify:

.. code-block:: text

    [ ] Inherits from Extractor
    [ ] Implements _extract() (not extract())
    [ ] Returns dict with "features" key
    [ ] Has NumPy-style docstring with Parameters, Returns, References
    [ ] Registered in __init__.py
    [ ] Has .rst documentation page
    [ ] Has tests with 100% coverage
    [ ] _validate() checks all user-facing parameters




