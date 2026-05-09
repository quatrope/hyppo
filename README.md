# hyppo-hsi

**Modular feature extractor for hyperspectral images**

[![QuatroPe](https://img.shields.io/badge/QuatroPe-Applications-1c5896)](https://quatrope.github.io/)
[![PyPI](https://img.shields.io/pypi/v/hyppo-hsi)](https://pypi.org/project/hyppo-hsi/)
[![License](https://img.shields.io/pypi/l/hyppo-hsi?color=blue)](https://www.tldrlegal.com/l/bsd3)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/hyppo-hsi/)

**HYPPO** is a modular feature extraction library for hyperspectral images (HSI). It provides a uniform, configurable interface for computing spectral and spatial features used in hyperspectral image classification.

## Features

- 17 feature extractors (spectral indices, dimensionality reduction, texture, morphological, wavelet, moment-based)
- Automatic dependency resolution between extractors via DAG
- Multiple execution backends: sequential, Dask threads/processes, multiprocessing with shared memory
- HDF5 I/O for HSI loading and feature saving
- YAML/JSON configuration for reproducible pipelines
- Python 3.11–3.14 support

## Installation

```bash
pip install hyppo-hsi
```

## Code Repository & Issues

<https://github.com/quatrope/hyppo>

## License

HYPPO is under the [BSD 3-Clause License](https://raw.githubusercontent.com/quatrope/hyppo/master/LICENSE).

Copyright (c) 2026, QuatroPe.
