from hyppo.hsi import HSI
from .base import Extractor


class DummyExtractor(Extractor):
    def __init__(self, result: int) -> None:
        super().__init__()
        self.result = result

    def extract(self, data: HSI) -> dict:
        return {"result": self.result}
