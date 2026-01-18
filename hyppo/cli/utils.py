"""Utility functions for CLI operations.

This module provides helper functions for the CLI including configuration
loading and docstring parsing.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from hyppo.io import Config, load_config_json, load_config_yaml

if TYPE_CHECKING:
    pass


def load_config(config: Path) -> Config:
    """Load configuration from file.

    Loads a configuration file in YAML or JSON format and returns
    a Config object containing the FeatureSpace and Runner.

    Parameters
    ----------
    config : Path
        Path to configuration file (.yaml, .yml, or .json).

    Returns
    -------
    Config
        Configuration object with feature_space and runner.

    Raises
    ------
    typer.Exit
        If configuration format is not supported.
    """
    typer.echo(f"Loading configuration from {config}...")
    if config is None:
        return ""

    if config.suffix in [".yaml", ".yml"]:
        return load_config_yaml(config)
    elif config.suffix == ".json":
        return load_config_json(config)
    else:
        msg = "Error: Unsupported config format. "
        msg += "Use .yaml, .yml, or .json"
        typer.echo(msg, err=True)
        raise typer.Exit(code=1)


def make_help(obj) -> str:
    r"""Extract summary from a method's NumPy-style docstring.

    This function extracts the summary section from a objects's
    docstring, which includes all content before the first section
    separator (a line of dashes). This is useful for generating
    concise help text for CLI commands.

    Parameters
    ----------
    obj : object
        Object with a NumPy-style docstring to extract help from.

    Returns
    -------
    str
        Summary portion of the docstring, or empty string if no
        docstring exists.

    Notes
    -----
    The function stops extracting at the first line that contains
    only dashes (e.g., "----------"), which marks the beginning of
    a formal section in NumPy-style docstrings.

    Examples
    --------
    >>> def example():
    ...     '''Short description.
    ...
    ...     Longer description here.
    ...
    ...     Parameters
    ...     ----------
    ...     ...
    ...     '''
    >>> make_help(example)
    'Short description.\\n\\nLonger description here.'
    """
    lines = (obj.__doc__ or "").strip().splitlines()
    if lines:
        for lineno, line in enumerate(lines):
            line = line.strip()
            if line and not line.replace("-", ""):
                break
        last_line = lineno - 1
        lines = lines[:last_line]
    return "\n".join(lines)


