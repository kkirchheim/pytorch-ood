from abc import ABC, abstractmethod
from typing import List, Dict

from torch.utils.data import Dataset
from pytorch_ood.api import Detector


class Benchmark(ABC):
    """
    Base class for Benchmarks
    """

    @abstractmethod
    def train_set(self) -> Dataset:
        """
        Training dataset
        """

    @abstractmethod
    def test_sets(self, known=True, unknown=True) -> List[Dataset]:
        """
        List of the different test datasets.
        If known and unknown are true, each dataset contains IN and OOD data.

        :param known: include IN
        :param unknown: include OOD
        """
        pass

    @abstractmethod
    def evaluate(
            self, detector: Detector, *args, **kwargs
    ) -> List[Dict]:
        """
        Evaluates the given detector on all datasets and returns a list with the results
        """
        pass
