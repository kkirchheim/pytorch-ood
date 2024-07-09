from abc import ABC, abstractmethod
from typing import TypeVar, Dict, Any

from torch import Tensor
from torch.utils.data import DataLoader
import torch
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

class AlreadyFittedException(Exception):
    """
    Raised when fit is called on a detector that has already been fitted.
    """

    def __init__(self, msg="Detector has already been fitted."):
        super(AlreadyFittedException, self).__init__(msg)
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
    
    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare the object's state for pickling.

        Returns:
            Dict[str, Any]: A copy of the object's __dict__ with the PyTorch model handled separately.
        """
        state = self.__dict__.copy()
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            state['model_state_dict'] = self.model.state_dict()
            state['model_class'] = self.model.__class__
            del state['model']
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore the object's state after unpickling.

        Args:
            state (Dict[str, Any]): The unpickled state dictionary.
        """
        model_state_dict = state.pop('model_state_dict', None)
        model_class = state.pop('model_class', None)
        self.__dict__.update(state)
        if model_state_dict is not None and model_class is not None:
            self.model = model_class()  # This assumes model_class can be instantiated without arguments
            self.model.load_state_dict(model_state_dict)
            
    @classmethod    
    def load(cls, path: str) -> Self:
        """
        Load a detector from a file.

        Args:
            path (str): The path to the file.

        Returns:
            Self: The loaded detector.
        """
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
        
    def save(self, path: str) -> None:
        """
        Save the detector to a file.

        Args:
            path (str): The path to the file.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)