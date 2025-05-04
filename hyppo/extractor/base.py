from abc import ABC, abstractmethod
import re
from typing import Any

from hyppo.hsi import HSI


class Extractor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def extract(self, data: HSI) -> dict[str, Any]: ...

    def validate(self):
        return

    @classmethod
    def feature_name(cls):
        name = cls.__name__
        name = name.removesuffix("FeatureExtractor").removesuffix("Extractor")

        name = re.split("(?<=.)(?=[A-Z])", name)

        return "_".join(name).lower()


