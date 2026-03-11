Adding a New Extractor
======================

This tutorial shows how to create a custom feature extractor for HYPPO.


Creating the Extractor
-----------------------

Create a new class that inherits from ``Extractor`` and implements the
``_extract()`` method:

.. code-block:: python

    """My custom feature extractor."""

    import numpy as np

    from hyppo.core import HSI
    from hyppo.extractor.base import Extractor


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


Adding Validation
------------------

Override ``_validate()`` to check preconditions before extraction:

.. code-block:: python

    def _validate(self, data: HSI, **inputs):
        if self.param1 <= 0:
            raise ValueError("param1 must be positive")

        if data.reflectance.ndim != 3:
            raise ValueError("Expected 3D reflectance array")


Declaring Dependencies
-----------------------

If your extractor depends on the output of another extractor, override
``get_input_dependencies()``:

.. code-block:: python

    from hyppo.extractor.pca import PCAExtractor

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


Using Your Extractor
---------------------

Once created, you can use it directly in a feature space:

.. code-block:: python

    from hyppo.core import FeatureSpace

    fs = FeatureSpace.from_list([
        MyFeatureExtractor(param1=20),
    ])

    hsi = hyppo.io.load_h5_hsi("image.h5")
    results = fs.extract(hsi)

To make it available by name in configuration files, see the
:doc:`../contributing` guide for registration and packaging steps.


Next Steps
----------

- See :doc:`advanced_usage` for parallel execution and complex dependencies
- Read :doc:`../contributing` for how to register, document, and test your extractor
