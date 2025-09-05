from hyppo.extractor.base import Extractor, InputDependency


class SimpleExtractor(Extractor):
    def extract(self, data):
        return {"simple_value": 1.0}


class MediumExtractor(Extractor):
    input_dependencies = {
        "simple_input": InputDependency(
            name="simple_input", extractor_type=SimpleExtractor, required=True
        )
    }

    def extract(self, data, **inputs):
        return {"medium_value": 2.0}


class AdvancedExtractor(Extractor):
    input_dependencies = {
        "medium_input": InputDependency(
            name="medium_input", extractor_type=MediumExtractor, required=True
        ),
        "simple_input1": InputDependency(
            name="simple_input1", extractor_type=SimpleExtractor, required=True
        ),
        "simple_input2": InputDependency(
            name="simple_input2", extractor_type=SimpleExtractor, required=True
        ),
        "simple_input3": InputDependency(
            name="simple_input3",
            extractor_type=SimpleExtractor,
            required=False,
            default_config={},
        ),
    }

    def extract(self, data, **inputs):
        return {"advanced_value": 3.0}


class ComplexExtractor(Extractor):
    input_dependencies = {
        "simple_input1": InputDependency(
            name="simple_input1", extractor_type=SimpleExtractor, required=True
        ),
        "simple_input2": InputDependency(
            name="simple_input2",
            extractor_type=SimpleExtractor,
            required=False,
            default_config={},
        ),
        "medium_input": InputDependency(
            name="medium_input",
            extractor_type=MediumExtractor,
            required=False,
            default_config={},
        ),
        "advanced_input1": InputDependency(
            name="advanced_input1", extractor_type=AdvancedExtractor, required=True
        ),
        "advanced_input2": InputDependency(
            name="advanced_input2", extractor_type=AdvancedExtractor, required=True
        ),
    }

    def extract(self, data, **inputs):
        return {"complex_value": 4.0}
