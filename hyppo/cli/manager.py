"""CLI command manager.

This module contains the CLIManager class that implements all CLI commands
as methods. Each public method is automatically registered as a Typer
subcommand.
"""

from pathlib import Path

import typer

from hyppo.io import load_h5_hsi


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
            help="Path to output file (.h5 format)",
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
            result.save(output_path)
            typer.echo("Results saved successfully!")

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
