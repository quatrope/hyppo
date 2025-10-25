# hyppo
Hyper Spectral Feature Extractor

## Command-Line Interface

Hyppo provides a command-line interface for extracting features from
hyperspectral imaging (HSI) data.

### Installation

After installing the package, the `hyppo` command becomes available:

```bash
pip install -e .
```

### Usage

All commands require a configuration file in YAML or JSON format:

```bash
hyppo -c config.yaml <command> [options]
```

### Available Commands

#### extract
Extract features from HSI data:

```bash
hyppo -c config.yaml extract input.h5
hyppo -c config.yaml extract input.h5 -o output.h5
```

#### info
Display configuration information:

```bash
hyppo -c config.yaml info
```

### Runner Options

Select different execution backends with the `-r/--runner` option:

- `sequential`: Single-threaded execution (default)
- `local`: Multi-process local execution
- `dask-thread`: Dask threaded execution
- `dask-process`: Dask process execution

Example with parallel execution:

```bash
hyppo -c config.yaml -r local -w 8 extract input.h5
```

### Getting Help

```bash
hyppo --help
hyppo extract --help
hyppo info --help
```
