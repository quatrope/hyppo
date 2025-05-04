from hyppo.extractor import DummyExtractor


def get_all_extractors():
    """
    TODO! Probablemente habria que moverlo a un registry o una busqueda de subclases. Por ahora estan hardcodeados a modo de prueba.
    """

    extractors = [DummyExtractor]

    return extractors
