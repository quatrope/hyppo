#!/usr/bin/env python3

"""Command-line interface for hyppo feature extraction using Typer.

This module provides a CLI for extracting features from hyperspectral
imaging (HSI) data. It uses Typer for argument parsing and supports
multiple execution backends through a runner registry.

The CLI supports the following commands:
    - extract: Extract features from HSI data
    - info: Display configuration information

Examples
--------
Extract features using default sequential runner:
    $ hyppo -c config.yaml extract input.h5

Extract features with output file:
    $ hyppo -c config.yaml extract input.h5 -o output.h5

Display configuration information:
    $ hyppo -c config.yaml info
"""

import inspect
from pathlib import Path
import sys

import typer

from hyppo.utils.bunch import Bunch

from .manager import CLIManager
from .utils import load_config, make_help


# PROUD OF THIS HACK!
# This clever trick makes the config parameter conditionally required:
# - When "--help" is in sys.argv, the default is None (optional)
# - Otherwise, the default is ... (Ellipsis), which means required
#
# Why this works:
# 1. typer.Option(...) means "required parameter" (no default value)
# 2. typer.Option(None) means "optional parameter" (default is None)
# 3. sys.argv is checked at import/parse time, before Typer validates
# 4. This allows `hyppo --help` and `hyppo extract --help` to work
#    without requiring -c/--config
# 5. But when actually running a command (no --help), config becomes
#    required and Typer will error if not provided
#
# Benefits:
# - Users can see help without needing a config file
# - Config is still mandatory for actual command execution
# - No need to check ctx.obj or add validation in each command
def _global_config(
    ctx: typer.Context,
    config: Path = typer.Option(
        (None if "--help" in sys.argv else ...),
        "-c",
        "--config",
        help="Path to configuration file (.yaml or .json)",
        exists=True,
        dir_okay=False,
    ),
):
    """Global configuration callback for Typer application.

    This function is called before any command and sets up the global
    configuration context. It loads the configuration file which contains
    both the feature space and runner specifications.

    Parameters
    ----------
    ctx : typer.Context
        Typer context to store application configuration.
    config : Path
        Path to configuration file.
    """
    cfg = load_config(config)

    app_config = {
        "feature_space": cfg.feature_space,
        "runner": cfg.runner,
    }

    ctx.obj = Bunch("app_config", app_config)


def _create_app(cli_manager):
    """Create and configure the Typer application.

    This function sets up the main Typer application instance and
    automatically registers all public methods from the CLI class as
    subcommands using introspection. This approach allows for clean
    separation of command logic while maintaining a simple
    registration mechanism.

    The application is configured with:
        - name: "hyppo"
        - Global callback: _global_config (handles -c option)
        - Auto-completion: Disabled
        - Commands: Dynamically registered from CLI class methods

    Parameters
    ----------
    cli_manager : CLIManager
        Instance of CLIManager class containing command methods to
        register.

    Returns
    -------
    typer.Typer
        Configured Typer application instance with all commands
        registered and ready to use.

    Notes
    -----
    Only public methods (not starting with '_') from the CLIManager
    class are registered as commands. Each method should accept a
    typer.Context as its first parameter to access the global
    configuration.

    Examples
    --------
    The returned app can be invoked directly:
        >>> app = _create_app(CLIManager())
        >>> app()  # Processes sys.argv and runs appropriate command
    """
    app = typer.Typer(
        name="hyppo",
        help="Hyperspectral feature extraction toolkit",
        add_completion=False,
        callback=_global_config,
    )

    members = inspect.getmembers(cli_manager, predicate=inspect.ismethod)
    for name, method in members:
        if not name.startswith("_"):
            doc = make_help(method)
            cmd_wrapper = app.command(name=name, help=doc)
            cmd_wrapper(method)

    return app


def main():
    """Entry point for the hyppo CLI application."""
    cli_manager = CLIManager()
    app = _create_app(cli_manager)
    app()