"""Test fixture extractors for dependency testing."""

from hyppo.extractor.base import Extractor


class SimpleExtractor(Extractor):
    """Simple test extractor with no dependencies."""

    def _extract(self, data, **inputs):
        """Extract simple test value."""
        return {"simple_value": 1.0}


class MediumExtractor(Extractor):
    """Medium complexity test extractor with one dependency."""

    @classmethod
    def get_input_dependencies(cls) -> dict:
        """Return test dependencies."""
        return {
            "simple_input": {"extractor": SimpleExtractor, "required": True}
        }

    def _extract(self, data, **inputs):
        """Extract medium test value."""
        return {"medium_value": 2.0}


class AdvancedExtractor(Extractor):
    """Advanced test extractor with multiple dependencies."""

    @classmethod
    def get_input_dependencies(cls) -> dict:
        """Return test dependencies."""
        return {
            "medium_input": {"extractor": MediumExtractor, "required": True},
            "simple_input1": {"extractor": SimpleExtractor, "required": True},
            "simple_input2": {"extractor": SimpleExtractor, "required": False},
        }

    @classmethod
    def get_input_default(cls, input_name: str) -> "Extractor | None":
        """Return test default extractors."""
        if input_name == "simple_input2":
            return SimpleExtractor()
        return None

    def _extract(self, data, **inputs):
        """Extract advanced test value."""
        return {"advanced_value": 3.0}


class ComplexExtractor(Extractor):
    """Complex test extractor with deep dependency tree."""

    @classmethod
    def get_input_dependencies(cls) -> dict:
        """Return test dependencies."""
        return {
            "simple_input1": {"extractor": SimpleExtractor, "required": True},
            "simple_input2": {"extractor": SimpleExtractor, "required": False},
            "medium_input": {"extractor": MediumExtractor, "required": False},
            "advanced_input1": {
                "extractor": AdvancedExtractor,
                "required": True,
            },
            "advanced_input2": {
                "extractor": AdvancedExtractor,
                "required": True,
            },
        }

    @classmethod
    def get_input_default(cls, input_name: str) -> "Extractor | None":
        """Return test default extractors."""
        if input_name == "simple_input2":
            return SimpleExtractor()
        elif input_name == "medium_input":
            return MediumExtractor()
        return None

    def _extract(self, data, **inputs):
        """Extract complex test value."""
        return {"complex_value": 4.0}
