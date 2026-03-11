Installation
============

Requirements
------------

HYPPO requires Python 3.11 or higher.

Installing from source
----------------------

To install HYPPO from source, clone the repository and install using pip:

.. code-block:: bash

    git clone https://github.com/quatrope/hyppo.git
    cd hyppo
    pip install -e .

Development installation
------------------------

For development, install with the development dependencies:

.. code-block:: bash

    git clone https://github.com/quatrope/hyppo.git
    cd hyppo
    pip install -e ".[dev]"

Dependencies
------------

HYPPO depends on the following packages:

- dask[distributed]
- h5py
- numpy
- scipy
- networkx
- pyyaml
- attrs
- scikit-learn
- scikit-image
- PyWavelets
- pandas
