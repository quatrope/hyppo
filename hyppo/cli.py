"""Command-line interface for hyppo feature extraction."""

import argparse
from hyppo.io import load_h5_hsi
from hyppo.io._config import load_config_json, load_config_yaml
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyppo.core import FeatureSpace
    from hyppo.runner import BaseRunner


class CLI:
    """Command-line interface for hyppo feature extraction."""

    def __init__(self, feature_space: "FeatureSpace", runner: "BaseRunner"):
        """
        Initialize CLI with feature space and runner.

        Args:
            feature_space: FeatureSpace instance with extractors
            runner: Runner instance for execution
        """
        self.feature_space = feature_space
        self.runner = runner

    def run_from_command_line(self, argv=None):
        """
        Parse command-line arguments and execute extraction.

        Args:
            argv: Command-line arguments (defaults to sys.argv)
        """
        parser = self._create_parser()
        args = parser.parse_args(argv)

        if args.command is None:
            parser.print_help()
            sys.exit(1)

        if args.command == "extract":
            self._handle_extract(args)
        elif args.command == "info":
            self._handle_info(args)

    def _create_parser(self):
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            prog="hyppo",
            description="Hyperspectral feature extraction toolkit",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Extract command
        extract_parser = subparsers.add_parser(
            "extract", help="Extract features from HSI data"
        )
        extract_parser.add_argument(
            "input",
            type=str,
            help="Path to input HSI file (.h5)",
        )
        extract_parser.add_argument(
            "-o",
            "--output",
            type=str,
            help="Path to output file (optional)",
        )

        # Info command
        subparsers.add_parser("info", help="Display information about feature space")

        return parser

    def _handle_extract(self, args):
        """
        Handle extract command.

        Args:
            args: Parsed command-line arguments
        """
        input_path = Path(args.input)

        if not input_path.exists():
            msg = f"Error: Input file not found: {input_path}"
            print(msg, file=sys.stderr)
            sys.exit(1)

        if input_path.suffix != ".h5":
            print("Error: Input file must be .h5 format", file=sys.stderr)
            sys.exit(1)

        print(f"Loading HSI data from {input_path}...")
        hsi = load_h5_hsi(str(input_path))

        print(f"Extracting features using {type(self.runner).__name__}...")
        result = self.feature_space.extract(hsi, runner=self.runner)

        print("Extraction complete!")
        print(result.describe())

        if args.output:
            output_path = Path(args.output)
            print(f"Saving results to {output_path}...")
            # TODO: Implement result saving

    def _handle_info(self, args):
        """
        Handle info command.

        Args:
            args: Parsed command-line arguments
        """
        print("Feature Space Configuration:")
        print("-" * 40)

        extractors = self.feature_space.get_extractors()
        for name, extractor in extractors.items():
            print(f"  {name}: {type(extractor).__name__}")

        print(f"\nRunner: {type(self.runner).__name__}")


def main():
    """Entry point for standalone CLI usage with config files."""
    parser = argparse.ArgumentParser(
        prog="hyppo",
        description="Hyperspectral feature extraction toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (.yaml or .json)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract features from HSI data"
    )
    extract_parser.add_argument(
        "input",
        type=str,
        help="Path to input HSI file (.h5)",
    )
    extract_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output file (optional)",
    )
    extract_parser.add_argument(
        "-r",
        "--runner",
        type=str,
        default="sequential",
        choices=["sequential", "local", "dask-thread", "dask-process"],
        help="Runner type (default: sequential)",
    )
    extract_parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Number of workers for parallel runners",
    )

    # Info command
    subparsers.add_parser("info", help="Display information about configuration")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        msg = f"Error: Configuration file not found: {config_path}"
        print(msg, file=sys.stderr)
        sys.exit(1)

    print(f"Loading configuration from {config_path}...")

    if config_path.suffix in [".yaml", ".yml"]:
        feature_space = load_config_yaml(config_path)
    elif config_path.suffix == ".json":
        feature_space = load_config_json(config_path)
    else:
        msg = "Error: Unsupported config format. Use .yaml, .yml, or .json"
        print(msg, file=sys.stderr)
        sys.exit(1)

    # Create runner
    runner = _create_runner(args)

    # Create CLI and execute
    cli = CLI(feature_space=feature_space, runner=runner)

    if args.command == "extract":
        cli._handle_extract(args)
    elif args.command == "info":
        cli._handle_info(args)


def _create_runner(args):
    """
    Create runner from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Runner instance
    """
    if args.command != "extract":
        from hyppo.runner import SequentialRunner

        return SequentialRunner()

    runner_type = args.runner

    if runner_type == "sequential":
        from hyppo.runner import SequentialRunner

        return SequentialRunner()
    elif runner_type == "local":
        from hyppo.runner import LocalProcessRunner

        num_workers = args.workers or 4
        return LocalProcessRunner(num_workers=num_workers)
    elif runner_type == "dask-thread":
        from hyppo.runner import DaskThreadedRunner

        return DaskThreadedRunner()
    elif runner_type == "dask-process":
        from hyppo.runner import DaskProcessRunner

        return DaskProcessRunner()


if __name__ == "__main__":
    main()
