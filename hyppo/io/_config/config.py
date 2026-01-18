"""Configuration container for HYPPO pipeline and runner."""

from pathlib import Path
from typing import TYPE_CHECKING, Union

import attr

if TYPE_CHECKING:
    from hyppo.core import FeatureSpace
    from hyppo.runner import BaseRunner


@attr.s(auto_attribs=True, frozen=True)
class Config:
    """Configuration container for feature extraction pipeline.

    This class encapsulates both the feature space (extractors) and
    the runner (execution backend) loaded from configuration files.

    Attributes
    ----------
    feature_space : FeatureSpace
        Configured feature extraction pipeline with all extractors.
    runner : BaseRunner
        Execution backend for running the feature extraction.
        If not provided, defaults to SequentialRunner.

    Examples
    --------
    >>> from hyppo.io import load_config_yaml
    >>> config = load_config_yaml("pipeline.yaml")
    >>> config.feature_space  # Access feature space
    >>> config.runner  # Access runner
    >>> results = config.feature_space.extract(hsi, runner=config.runner)

    >>> # Create Config without explicit runner (uses default)
    >>> from hyppo.core import FeatureSpace
    >>> fs = FeatureSpace.from_list([MeanExtractor()])
    >>> config = Config(feature_space=fs)
    >>> config.runner  # SequentialRunner instance

    >>> # Save config to file
    >>> config.save("pipeline.yaml")
    >>> config.save("pipeline.json")
    """

    feature_space: "FeatureSpace"
    runner: "BaseRunner" = attr.Factory(lambda: Config.get_default_runner())

    @staticmethod
    def get_default_runner() -> "BaseRunner":
        """
        Get the default runner for feature extraction.

        Returns
        -------
        BaseRunner
            Default runner instance (SequentialRunner)
        """
        from hyppo.runner import SequentialRunner

        return SequentialRunner()

    def save(self, path: Union[Path, str]) -> None:
        """
        Save this configuration to file.

        Determines format based on file extension and saves accordingly.

        Parameters
        ----------
        path : Path or str
            Output file path (.yaml, .yml, or .json extension)

        Raises
        ------
        ValueError
            If path doesn't have .yaml, .yml, or .json extension

        Examples
        --------
        >>> config = Config(feature_space=fs)
        >>> config.save("pipeline.yaml")
        >>> config.save("pipeline.json")
        """
        from . import save_config_json, save_config_yaml

        if not isinstance(path, Path):
            path = Path(path)

        if path.suffix in [".yaml", ".yml"]:
            save_config_yaml(self, path)
        elif path.suffix == ".json":
            save_config_json(self, path)
        else:
            msg = (
                f"Path must have .yaml, .yml, or .json extension, "
                f"got {path.suffix}"
            )
            raise ValueError(msg)
