from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hyppo.hsi import HSI


if TYPE_CHECKING:
    from feature_space import FeatureSpace


class BaseRunner(ABC):
    @abstractmethod
    def resolve(self, data: HSI, feature_space: "FeatureSpace") -> dict: ...
