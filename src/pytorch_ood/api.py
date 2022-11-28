from abc import ABC, abstractmethod
from typing import TypeVar

import torch
from torch.utils.data import DataLoader

Self = TypeVar("Self")


class RequiresFittingException(Exception):
    """
    Raised when predict is called on a detector that has not been fitted.
    """

    pass


class Detector(ABC):
    """
    Abstract Base Class for an Out-of-Distribution Detector
    """

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Forwards to predict
        """
        return self.predict(*args, **kwargs)

    @abstractmethod
    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Fit the model to a dataset. Some methods require this.

        :param data_loader: dataset to fit on. This is usually the training dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates outlier scores.

        :param x: batch of data
        :return: outlier scores for points

        :raise RequiresFitException: if detector has to be fitted to some data
        """
        raise NotImplementedError
