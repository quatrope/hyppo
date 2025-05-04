import dask.threaded as dsk
from .base import BaseRunner


class ThreadsRunner(BaseRunner):
    def __init__(self, num_workers: int = 32) -> None:
        super().__init__()
        if num_workers < 1:
            raise ValueError(f"Invalid number of workers: {num_workers}")

        self.num_workers = num_workers

    def resolve(self, data, feature_space) -> dict:
        # Build dependency tree
        extractors = feature_space.get_extractors()
        graph = {
            alias: (extractor.extract, data) for alias, extractor in extractors.items()
        }

        # Execute tree
        keys = [alias for alias in graph.keys()]
        results_list = dsk.get(graph, keys, num_workers=self.num_workers)
        data = {alias: data for alias, data in zip(keys, results_list, strict=True)}

        # Return results for each entry
        results = {
            name: {"data": result, "extractor": extractors[name]}
            for name, result in data.items()
        }

        return results
