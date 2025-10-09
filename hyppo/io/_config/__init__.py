# @classmethod
# def from_config(
#     cls, config_path: str | Path, validate: bool = True
# ) -> "FeatureSpace":
#     """
#     Create FeatureSpace from configuration file.

#     Args:
#         config_path: Path to configuration file (JSON or YAML)
#         validate: Whether to validate configuration before building

#     Returns:
#         Configured FeatureSpace ready for extraction

#     Example:
#         >>> fs = FeatureSpace.from_config("config.yaml")
#         >>> hsi = hyppo.io.load_h5_hsi("data.h5")
#         >>> results = fs.extract(hsi)
#     """
#     from hyppo.io import parse_config, ConfigExecutor

#     config = parse_config(config_path)
#     executor = ConfigExecutor(config, validate=validate)
#     return executor.build_feature_space()
