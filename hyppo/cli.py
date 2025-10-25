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

Extract features using local parallel runner with 8 workers:
    $ hyppo -c config.yaml -r local -w 8 extract input.h5 -o output.h5

Display configuration information:
    $ hyppo -c config.yaml info
"""

# =============================================================================
# IMPORTS
# =============================================================================

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from .io import load_h5_hsi
from .io._config import load_config_json, load_config_yaml
from .utils.bunch import Bunch

if TYPE_CHECKING:
    from .core import FeatureSpace
    from .runner import BaseRunner


# =========================================================================
# RUNNER REGISTRY
# =========================================================================


def _create_sequential_runner(
    workers: int | None = None,
) -> "BaseRunner":
    """Create a sequential runner instance.

    Parameters
    ----------
    workers : int, optional
        Number of workers (unused for sequential runner).

    Returns
    -------
    BaseRunner
        Sequential runner instance.
    """
    from hyppo.runner import SequentialRunner

    return SequentialRunner()


def _create_local_runner(workers: int | None = None) -> "BaseRunner":
    """Create a local process pool runner instance.

    Parameters
    ----------
    workers : int, optional
        Number of worker processes. Defaults to 4.

    Returns
    -------
    BaseRunner
        Local process runner instance.
    """
    from hyppo.runner import LocalProcessRunner

    return LocalProcessRunner(num_workers=workers or 4)


def _create_dask_thread_runner(
    workers: int | None = None,
) -> "BaseRunner":
    """Create a Dask threaded runner instance.

    Parameters
    ----------
    workers : int, optional
        Number of workers (unused for Dask threaded runner).

    Returns
    -------
    BaseRunner
        Dask threaded runner instance.
    """
    from hyppo.runner import DaskThreadedRunner

    return DaskThreadedRunner()


def _create_dask_process_runner(
    workers: int | None = None,
) -> "BaseRunner":
    """Create a Dask process runner instance.

    Parameters
    ----------
    workers : int, optional
        Number of workers (unused for Dask process runner).

    Returns
    -------
    BaseRunner
        Dask process runner instance.
    """
    from hyppo.runner import DaskProcessRunner

    return DaskProcessRunner()


_RUNNERS_REGISTRY = {
    "sequential": _create_sequential_runner,
    "local": _create_local_runner,
    "dask-thread": _create_dask_thread_runner,
    "dask-process": _create_dask_process_runner,
}


# =========================================================================
# CLASSES
# =========================================================================


class CLIManager:
    """CLI manager that exposes methods as typer subcommands.

    This class contains methods that are automatically registered as
    Typer commands through introspection. Each public method becomes
    a CLI subcommand.

    Methods
    -------
    extract
        Extract features from HSI data.
    info
        Display feature space configuration information.
    """

    def extract(
        self,
        ctx: typer.Context,
        input_file: str = typer.Argument(
            ...,
            help="Path to input HSI file (.h5)",
        ),
        output: str | None = typer.Option(
            None,
            "-o",
            "--output",
            help="Path to output file (optional)",
        ),
    ) -> None:
        """Extract features from HSI data.

        This command loads an HSI file, applies the configured feature
        extractors from the feature space, and outputs the results.

        Parameters
        ----------
        ctx : typer.Context
            Typer context containing feature_space and runner.
        input_file : str
            Path to input HSI file in .h5 format.
        output : str, optional
            Path to save extraction results.

        Raises
        ------
        typer.Exit
            If input file not found or has invalid format.
        """
        feature_space, runner = (
            ctx.obj.feature_space,
            ctx.obj.runner,
        )

        input_path = Path(input_file)

        if not input_path.exists():
            msg = f"Error: Input file not found: {input_path}"
            typer.echo(msg, err=True)
            raise typer.Exit(code=1)

        if input_path.suffix != ".h5":
            msg = "Error: Input file must be .h5 format"
            typer.echo(msg, err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Loading HSI data from {input_path}...")
        hsi = load_h5_hsi(str(input_path))

        runner_name = type(runner).__name__
        typer.echo(f"Extracting features using {runner_name}...")
        result = feature_space.extract(hsi, runner=runner)

        typer.echo("Extraction complete!")
        typer.echo(result.describe())

        if output:
            output_path = Path(output)
            typer.echo(f"Saving results to {output_path}...")
            # TODO: Implement result saving

    def info(self, ctx: typer.Context) -> None:
        """Display information about feature space configuration.

        This command shows all configured feature extractors and the
        selected runner type.

        Parameters
        ----------
        ctx : typer.Context
            Typer context containing feature_space and runner.
        """
        feature_space, runner = (
            ctx.obj.feature_space,
            ctx.obj.runner,
        )

        typer.echo("Feature Space Configuration:")
        typer.echo("-" * 40)

        extractors = feature_space.get_extractors()
        for name, extractor in extractors.items():
            extractor_type = type(extractor).__name__
            typer.echo(f"  {name}: {extractor_type}")

        typer.echo(f"\nRunner: {type(runner).__name__}")


# =========================================================================
# FUNCTIONS
# =========================================================================


def _create_feature_space(config: Path) -> "FeatureSpace":
    """Create feature space from configuration file.

    Loads a configuration file in YAML or JSON format and creates
    a FeatureSpace instance with the specified extractors.

    Parameters
    ----------
    config : Path
        Path to configuration file (.yaml, .yml, or .json).

    Returns
    -------
    FeatureSpace
        Configured feature space instance.

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


def _create_runner(
    runner_type: str,
    workers: int | None = None,
) -> "BaseRunner":
    """Create runner from registry.

    Uses the runner registry to instantiate the appropriate runner
    type based on the provided name.

    Parameters
    ----------
    runner_type : str
        Type of runner to create. Valid options:
        'sequential', 'local', 'dask-thread', 'dask-process'.
    workers : int, optional
        Number of workers for parallel runners.

    Returns
    -------
    BaseRunner
        Configured runner instance.

    Raises
    ------
    typer.Exit
        If runner type is not found in registry.
    """
    factory = _RUNNERS_REGISTRY.get(runner_type)

    if factory is None:
        valid_runners = ", ".join(_RUNNERS_REGISTRY.keys())
        msg = f"Error: Unknown runner type: {runner_type}"
        typer.echo(msg, err=True)
        typer.echo(f"Valid options: {valid_runners}", err=True)
        raise typer.Exit(code=1)

    return factory(workers)


def _global_config(
    ctx: typer.Context,
    config: Path = typer.Option(
        ...,
        "-c",
        "--config",
        help="Path to configuration file (.yaml or .json)",
        exists=True,
        dir_okay=False,
    ),
    runner_type: str = typer.Option(
        "sequential",
        "-r",
        "--runner",
        help="Runner type",
    ),
    workers: int | None = typer.Option(
        None,
        "-w",
        "--workers",
        help="Number of workers for parallel runners",
    ),
):
    """Global configuration callback for Typer application.

    This function is called before any command and sets up the global
    configuration context. It loads the feature space from the config
    file and creates the appropriate runner.

    Parameters
    ----------
    ctx : typer.Context
        Typer context to store application configuration.
    config : Path
        Path to configuration file.
    runner_type : str
        Type of runner to use.
    workers : int, optional
        Number of workers for parallel runners.
    """
    # Load configuration and create components
    app_config = {
        "feature_space": _create_feature_space(config),
        "runner": _create_runner(runner_type, workers),
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
        - Global callback: _global_config (handles -c, -r, -w options)
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

    # Introspect CLI class and register methods as commands
    members = inspect.getmembers(cli_manager, predicate=inspect.ismethod)
    for name, method in members:
        if not name.startswith("_"):
            app.command(name=name)(method)

    return app


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    """Entry point for the hyppo CLI application."""
    cli_manager = CLIManager()
    app = _create_app(cli_manager)
    app()


if __name__ == "__main__":
    main()
