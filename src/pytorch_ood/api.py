from abc import ABC, abstractmethod
from typing import TypeVar

from torch import Tensor
from torch.utils.data import DataLoader

Self = TypeVar("Self")


class RequiresFittingException(Exception):
    """
    Raised when predict is called on a detector that has not been fitted.
    """

    def __init__(self, msg="You have to call fit() before predict()"):
        super(RequiresFittingException, self).__init__(msg)


class ModelNotSetException(ValueError):
    """
    Raised when predict() is called but no model was given.
    """

    def __init__(self, msg="When using predict(), model must not be None"):
        super(ModelNotSetException, self).__init__(msg)


class Detector(ABC):
    """
    Abstract Base Class for an Out-of-Distribution Detector
    """

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Forwards to predict
        """
        return self.predict(*args, **kwargs)

    @abstractmethod
    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Fit the detector to a dataset. Some methods require this.

        :param data_loader: dataset to fit on. This is usually the training dataset.

        :raise ModelNotSetException: if model was not set
        """
        raise NotImplementedError

    @abstractmethod
    def fit_features(self: Self, x: Tensor, y: Tensor) -> Self:
        """
        Fit the detector directly on features. Some methods require this.

        :param x: training features to use for fitting.
        :param y: corresponding class labels.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """
        Calculates outlier scores. Inputs will be passed through the model.

        :param x: batch of data
        :return: outlier scores for points

        :raise RequiresFitException: if detector has to be fitted to some data
        :raise ModelNotSetException: if model was not set
        """
        raise NotImplementedError

    @abstractmethod
    def predict_features(self, x: Tensor) -> Tensor:
        """
        Calculates outlier scores based on features.

        :param x: batch of data
        :return: outlier scores for points

        :raise RequiresFitException: if detector has to be fitted to some data
        """
        raise NotImplementedError
