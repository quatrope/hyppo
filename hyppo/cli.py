"""Command-line interface for hyppo feature extraction using Typer."""

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from .io import load_h5_hsi
from .io._config import load_config_json, load_config_yaml

from .utils.bunch import Bunch

if TYPE_CHECKING:
    from hyppo.core import FeatureSpace
    from hyppo.runner import BaseRunner


# =============================================================================
# RUNNER REGISTRY
# =============================================================================


def _create_sequential_runner(workers: Optional[int] = None) -> "BaseRunner":
    from hyppo.runner import SequentialRunner

    return SequentialRunner()


def _create_local_runner(workers: Optional[int] = None) -> "BaseRunner":
    from hyppo.runner import LocalProcessRunner

    return LocalProcessRunner(num_workers=workers or 4)


def _create_dask_thread_runner(workers: Optional[int] = None) -> "BaseRunner":
    from hyppo.runner import DaskThreadedRunner

    return DaskThreadedRunner()


def _create_dask_process_runner(workers: Optional[int] = None) -> "BaseRunner":
    from hyppo.runner import DaskProcessRunner

    return DaskProcessRunner()


_RUNNERS_REGISTRY = {
    "sequential": _create_sequential_runner,
    "local": _create_local_runner,
    "dask-thread": _create_dask_thread_runner,
    "dask-process": _create_dask_process_runner,
}


# =============================================================================
# CLASSES
# =============================================================================


class CLI:
    """
    CLI class that exposes methods as typer subcommands.

    This class has one method per subcommand with the real typer interface.
    """

    def extract(
        self,
        ctx: typer.Context,
        input_file: str = typer.Argument(
            ...,
            help="Path to input HSI file (.h5)",
        ),
        output: Optional[str] = typer.Option(
            None,
            "-o",
            "--output",
            help="Path to output file (optional)",
        ),
    ) -> None:
        """Extract features from HSI data."""

        feature_space, runner = ctx.obj.feature_space, ctx.obj.runner

        input_path = Path(input_file)

        if not input_path.exists():
            msg = f"Error: Input file not found: {input_path}"
            typer.echo(msg, err=True)
            raise typer.Exit(code=1)

        if input_path.suffix != ".h5":
            typer.echo("Error: Input file must be .h5 format", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Loading HSI data from {input_path}...")
        hsi = load_h5_hsi(str(input_path))

        typer.echo(f"Extracting features using {type(runner).__name__}...")
        result = feature_space.extract(hsi, runner=runner)

        typer.echo("Extraction complete!")
        typer.echo(result.describe())

        if output:
            output_path = Path(output)
            typer.echo(f"Saving results to {output_path}...")
            # TODO: Implement result saving

    def info(self, ctx: typer.Context) -> None:
        """Display information about feature space configuration."""
        feature_space, runner = ctx.obj.feature_space, ctx.obj.runner

        typer.echo("Feature Space Configuration:")
        typer.echo("-" * 40)

        extractors = feature_space.get_extractors()
        for name, extractor in extractors.items():
            typer.echo(f"  {name}: {type(extractor).__name__}")

        typer.echo(f"\nRunner: {type(runner).__name__}")


# =============================================================================
# FUNCTIONS
# =============================================================================


def _create_feature_space(config: Path) -> "FeatureSpace":
    """
    Create feature space from configuration file.

    Args:
        config: Path to configuration file (.yaml, .yml, or .json)

    Returns:
        FeatureSpace instance
    """
    typer.echo(f"Loading configuration from {config}...")
    if config is None:
        return ""

    if config.suffix in [".yaml", ".yml"]:
        return load_config_yaml(config)
    elif config.suffix == ".json":
        return load_config_json(config)
    else:
        typer.echo(
            "Error: Unsupported config format. Use .yaml, .yml, or .json",
            err=True,
        )
        raise typer.Exit(code=1)


def _create_runner(runner_type: str, workers: Optional[int] = None) -> "BaseRunner":
    """
    Create runner from registry.

    Args:
        runner_type: Type of runner (sequential, local, dask-thread, dask-process)
        workers: Number of workers for parallel runners

    Returns:
        Runner instance
    """
    factory = _RUNNERS_REGISTRY.get(runner_type)

    if factory is None:
        valid_runners = ", ".join(_RUNNERS_REGISTRY.keys())
        typer.echo(f"Error: Unknown runner type: {runner_type}", err=True)
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
    workers: Optional[int] = typer.Option(
        None,
        "-w",
        "--workers",
        help="Number of workers for parallel runners",
    ),
):

    # Load configuration and create components
    app_config = {
        "feature_space": _create_feature_space(config),
        "runner": _create_runner(runner_type, workers),
    }

    ctx.obj = Bunch("app_config", app_config)


def _create_app():
    app = typer.Typer(
        name="hyppo",
        help="Hyperspectral feature extraction toolkit",
        add_completion=False,
        callback=_global_config,
    )

    cli = CLI()

    # Introspect CLI class and register methods as commands
    for name, method in inspect.getmembers(cli, predicate=inspect.ismethod):
        if not name.startswith("_"):
            app.command(name=name)(method)

    return app


def main():
    app = _create_app()
    app()


if __name__ == "__main__":
    main()
