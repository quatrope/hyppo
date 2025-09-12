from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hyppo.core import HSI


if TYPE_CHECKING:
    from hyppo.core import FeatureSpace


class BaseRunner(ABC):
    @abstractmethod
    def resolve(self, data: HSI, feature_space: "FeatureSpace") -> dict: ...
