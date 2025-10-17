"""Command-line interface for hyppo feature extraction using Typer."""

import sys
from pathlib import Path
from typing import Optional, Annotated

import typer
from typing_extensions import Literal

from hyppo.io import load_h5_hsi
from hyppo.io._config import load_config_yaml, load_config_json
from hyppo.core import FeatureSpace
from hyppo.runner import BaseRunner

app = typer.Typer(
    name="hyppo",
    help="Hyperspectral feature extraction toolkit",
    add_completion=False,
)


class CLI:
    """Command-line interface for hyppo feature extraction."""

    def __init__(self, feature_space: FeatureSpace, runner: BaseRunner):
        """
        Initialize CLI with feature space and runner.

        Args:
            feature_space: FeatureSpace instance with extractors
            runner: Runner instance for execution
        """
        self.feature_space = feature_space
        self.runner = runner
        self.app = typer.Typer(add_completion=False)
        self._setup_commands()

    def _setup_commands(self):
        """Setup CLI commands."""

        @self.app.command()
        def extract(
            input: Annotated[str, typer.Argument(help="Path to input HSI file (.h5)")],
            output: Annotated[
                Optional[str], typer.Option("-o", "--output", help="Path to output file")
            ] = None,
        ):
            """Extract features from HSI data."""
            self._handle_extract(input, output)

        @self.app.command()
        def info():
            """Display information about feature space."""
            self._handle_info()

    def run_from_command_line(self, argv=None):
        """
        Parse command-line arguments and execute extraction.

        Args:
            argv: Command-line arguments (defaults to sys.argv)
        """
        self.app(argv)

    def _handle_extract(self, input: str, output: Optional[str]):
        """
        Handle extract command.

        Args:
            input: Path to input HSI file
            output: Optional path to output file
        """
        input_path = Path(input)

        if not input_path.exists():
            typer.secho(
                f"Error: Input file not found: {input_path}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        if input_path.suffix != ".h5":
            typer.secho(
                "Error: Input file must be .h5 format", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(1)

        typer.echo(f"Loading HSI data from {input_path}...")
        hsi = load_h5_hsi(str(input_path))

        typer.echo(f"Extracting features using {type(self.runner).__name__}...")
        result = self.feature_space.extract(hsi, runner=self.runner)

        typer.secho("Extraction complete!", fg=typer.colors.GREEN)
        typer.echo(result.describe())

        if output:
            output_path = Path(output)
            typer.echo(f"Saving results to {output_path}...")
            # TODO: Implement result saving

    def _handle_info(self):
        """Handle info command."""
        typer.secho("Feature Space Configuration:", bold=True)
        typer.echo("-" * 40)

        extractors = self.feature_space.get_extractors()
        for name, extractor in extractors.items():
            typer.echo(f"  {name}: {type(extractor).__name__}")

        typer.echo(f"\nRunner: {type(self.runner).__name__}")


@app.command()
def extract(
    input: Annotated[str, typer.Argument(help="Path to input HSI file (.h5)")],
    config: Annotated[
        str, typer.Option("-c", "--config", help="Path to configuration file (.yaml or .json)")
    ],
    output: Annotated[
        Optional[str], typer.Option("-o", "--output", help="Path to output file")
    ] = None,
    runner: Annotated[
        Literal["sequential", "local", "dask-thread", "dask-process"],
        typer.Option("-r", "--runner", help="Runner type"),
    ] = "sequential",
    workers: Annotated[
        Optional[int], typer.Option("-w", "--workers", help="Number of workers for parallel runners")
    ] = None,
):
    """Extract features from HSI data."""
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        typer.secho(
            f"Error: Configuration file not found: {config_path}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Loading configuration from {config_path}...")

    if config_path.suffix in [".yaml", ".yml"]:
        feature_space = load_config_yaml(config_path)
    elif config_path.suffix == ".json":
        feature_space = load_config_json(config_path)
    else:
        typer.secho(
            "Error: Unsupported config format. Use .yaml, .yml, or .json",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Create runner
    runner_instance = _create_runner(runner, workers)

    # Create CLI and execute
    cli = CLI(feature_space=feature_space, runner=runner_instance)
    cli._handle_extract(input, output)


@app.command()
def info(
    config: Annotated[
        str, typer.Option("-c", "--config", help="Path to configuration file (.yaml or .json)")
    ],
):
    """Display information about configuration."""
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        typer.secho(
            f"Error: Configuration file not found: {config_path}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Loading configuration from {config_path}...")

    if config_path.suffix in [".yaml", ".yml"]:
        feature_space = load_config_yaml(config_path)
    elif config_path.suffix == ".json":
        feature_space = load_config_json(config_path)
    else:
        typer.secho(
            "Error: Unsupported config format. Use .yaml, .yml, or .json",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Create dummy runner for info
    from hyppo.runner import SequentialRunner

    runner = SequentialRunner()

    # Create CLI and execute
    cli = CLI(feature_space=feature_space, runner=runner)
    cli._handle_info()


def _create_runner(
    runner_type: Literal["sequential", "local", "dask-thread", "dask-process"],
    workers: Optional[int] = None,
) -> BaseRunner:
    """
    Create runner from command-line arguments.

    Args:
        runner_type: Type of runner to create
        workers: Number of workers for parallel runners

    Returns:
        Runner instance
    """
    if runner_type == "sequential":
        from hyppo.runner import SequentialRunner

        return SequentialRunner()
    elif runner_type == "local":
        from hyppo.runner import LocalProcessRunner

        num_workers = workers or 4
        return LocalProcessRunner(num_workers=num_workers)
    elif runner_type == "dask-thread":
        from hyppo.runner import DaskThreadedRunner

        return DaskThreadedRunner()
    elif runner_type == "dask-process":
        from hyppo.runner import DaskProcessRunner

        return DaskProcessRunner()


def main():
    """Entry point for standalone CLI usage with config files."""
    app()


if __name__ == "__main__":
    main()
