from hyppo.extractor.base import Extractor
from hyppo.utils import get_all_extractors


class FeatureSpace:
    def __init__(self, extractors: dict) -> None:
        self.extractors = self.validate_extractors(extractors)

    @classmethod
    def from_features(cls, features: list, **params):
        extractors = get_all_extractors()

        extractors_by_name = {
            extractor.feature_name(): extractor for extractor in extractors
        }

        for feature in features:
            extractor_cls = extractors_by_name.get(feature)
            if extractor_cls is None:
                raise ValueError(f"Feature desconocida: {feature}")

        tree = {}
        for feature in features:
            extractor_cls = extractors_by_name[feature]
            extractor_params = params.get(feature, {})
            extractor = extractor_cls(**extractor_params)

            tree[feature] = extractor

        return cls(tree)

    def extract(self, data, runner=None):
        if runner is None:
            # Create default ThreadsRunner (uses all available cores by default)
            from hyppo.runner.threads import ThreadsRunner

            runner = ThreadsRunner()

        result = runner.resolve(data, self)
        return result

    def validate_extractors(self, extractors: dict):
        if len(extractors) == 0:
            raise ValueError("No extractors supplied.")

        for alias, extractor in extractors.items():
            if not isinstance(extractor, Extractor):
                raise TypeError(f"Extractor for alias {alias} must be an Extractor")

            extractor.validate()

        return extractors

    def get_extractors(self) -> dict[str, Extractor]:
        return self.extractors
