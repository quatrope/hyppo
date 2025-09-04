from hyppo.extractor import (
    DummyExtractor,
    MeanExtractor,
    StdExtractor,
    MinExtractor,
    MaxExtractor,
    MedianExtractor,
    GaborExtractor,
)


def get_all_extractors():
    """
    Returns all available feature extractors.
    TODO: Could be improved with automatic discovery using metaclasses or inspection.
    """

    extractors = [
        DummyExtractor,
        MeanExtractor,
        StdExtractor,
        MinExtractor,
        MaxExtractor,
        MedianExtractor,
        GaborExtractor,
    ]

    return extractors
