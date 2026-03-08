Adding a New Extractor
======================

This guide walks through the process of creating and registering a new
feature extractor in HYPPO.


Step 1: Create the Extractor Module
------------------------------------

Create a new file in ``hyppo/extractor/``. For example, to create a
``MyFeatureExtractor``:

.. code-block:: python

    # hyppo/extractor/myfeature.py
    """My custom feature extractor."""

    import numpy as np

    from hyppo.core import HSI
    from .base import Extractor


    class MyFeatureExtractor(Extractor):
        """
        Custom feature extractor for hyperspectral images.

        Parameters
        ----------
        param1 : int, default=10
            Description of parameter.
        """

        def __init__(self, param1=10):
            super().__init__()
            self.param1 = param1

        def _extract(self, data: HSI, **inputs) -> dict:
            reflectance = data.reflectance
            h, w, bands = reflectance.shape

            features = np.mean(reflectance, axis=2, keepdims=True)

            return {
                "features": features,
                "original_shape": (h, w, bands),
            }

.. important::

    - Implement ``_extract()``, **not** ``extract()``. The public ``extract()``
      method handles the validation flow automatically.
    - Always return a dictionary with at least a ``"features"`` key.


Step 2: Add Validation (Optional)
----------------------------------

Override ``_validate()`` to check preconditions before extraction:

.. code-block:: python

    def _validate(self, data: HSI, **inputs):
        if self.param1 <= 0:
            raise ValueError("param1 must be positive")

        if data.reflectance.ndim != 3:
            raise ValueError("Expected 3D reflectance array")


Step 3: Declare Dependencies (Optional)
-----------------------------------------

If your extractor depends on the output of another extractor, override
``get_input_dependencies()``:

.. code-block:: python

    from .pca import PCAExtractor

    class MyDependentExtractor(Extractor):

        @classmethod
        def get_input_dependencies(cls) -> dict:
            return {
                "pca_features": {
                    "extractor": PCAExtractor,
                    "required": True,
                }
            }

        def _extract(self, data: HSI, **inputs) -> dict:
            pca_result = inputs["pca_features"]
            pca_features = pca_result["features"]

            # Use PCA features for further processing
            return {"features": pca_features * 2}

For optional dependencies, set ``"required": False`` and provide a default
extractor:

.. code-block:: python

    @classmethod
    def get_input_dependencies(cls) -> dict:
        return {
            "optional_input": {
                "extractor": SomeExtractor,
                "required": False,
            }
        }

    @classmethod
    def get_input_default(cls, input_name: str):
        if input_name == "optional_input":
            return SomeExtractor(default_param=5)
        return None


Step 4: Register the Extractor
-------------------------------

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


Step 5: Add Documentation
--------------------------

Create an API documentation page at ``docs/source/api/extractor/myfeature.rst``:

.. code-block:: rst

    ``hyppo.extractor.myfeature`` module
    =====================================

    .. automodule:: hyppo.extractor.myfeature
       :members:
       :undoc-members:
       :show-inheritance:

The ``automodule`` directive will pull docstrings directly from the code.
No additional content is needed; keep docstrings in the source code.


Step 6: Write Tests
--------------------

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

See ``tests/extractor/test_base.py`` for additional test patterns
and ``tests/CLAUDE.md`` for testing conventions.


Feature Naming
--------------

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


Checklist
---------

Before submitting a new extractor, verify:

.. code-block:: text

    [ ] Inherits from Extractor
    [ ] Implements _extract() (not extract())
    [ ] Returns dict with "features" key
    [ ] Has NumPy-style docstring with Parameters, Returns, References
    [ ] Registered in __init__.py
    [ ] Has .rst documentation page
    [ ] Has tests with >95% coverage
    [ ] _validate() checks all user-facing parameters
