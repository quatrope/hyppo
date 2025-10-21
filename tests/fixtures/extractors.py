from hyppo.extractor.base import Extractor


class SimpleExtractor(Extractor):
    def _extract(self, data, **inputs):
        return {"simple_value": 1.0}


class MediumExtractor(Extractor):
    @classmethod
    def get_input_dependencies(cls) -> dict:
        return {"simple_input": {"extractor": SimpleExtractor, "required": True}}

    def _extract(self, data, **inputs):
        return {"medium_value": 2.0}


class AdvancedExtractor(Extractor):
    @classmethod
    def get_input_dependencies(cls) -> dict:
        return {
            "medium_input": {"extractor": MediumExtractor, "required": True},
            "simple_input1": {"extractor": SimpleExtractor, "required": True},
            "simple_input2": {"extractor": SimpleExtractor, "required": False},
        }

    @classmethod
    def get_input_default(cls, input_name: str) -> "Extractor | None":
        if input_name == "simple_input2":
            return SimpleExtractor()
        return None

    def _extract(self, data, **inputs):
        return {"advanced_value": 3.0}


class ComplexExtractor(Extractor):
    @classmethod
    def get_input_dependencies(cls) -> dict:
        return {
            "simple_input1": {"extractor": SimpleExtractor, "required": True},
            "simple_input2": {"extractor": SimpleExtractor, "required": False},
            "medium_input": {"extractor": MediumExtractor, "required": False},
            "advanced_input1": {"extractor": AdvancedExtractor, "required": True},
            "advanced_input2": {"extractor": AdvancedExtractor, "required": True},
        }

    @classmethod
    def get_input_default(cls, input_name: str) -> "Extractor | None":
        if input_name == "simple_input2":
            return SimpleExtractor()
        elif input_name == "medium_input":
            return MediumExtractor()
        return None

    def _extract(self, data, **inputs):
        return {"complex_value": 4.0}
