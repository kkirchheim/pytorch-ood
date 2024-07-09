import logging
from typing import Callable, TypeVar

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_ood.utils import extract_features, is_known, contains_unknown
from pytorch_ood.api import ModelNotSetException, Detector, RequiresFittingException

log = logging.getLogger(__name__)
Self = TypeVar("Self")

def _circfuncs_common(samples, high, low):
    sin_samp = torch.sin((samples - low) * 2. * torch.pi / (high - low))
    cos_samp = torch.cos((samples - low) * 2. * torch.pi / (high - low))
    return samples, sin_samp, cos_samp

def circmean(samples, high=2*torch.pi, low=0, axis=None):
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_sum = torch.sum(sin_samp, axis)
    cos_sum = torch.sum(cos_samp, axis)
    res = torch.atan2(sin_sum, cos_sum) % (2*torch.pi)
    return res * (high - low) / (2. * torch.pi) + low
import logging
from typing import Callable, TypeVar, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_ood.utils import extract_features
from pytorch_ood.api import Detector, RequiresFittingException

log = logging.getLogger(__name__)
Self = TypeVar("Self")

class Distance2Clusters(Detector):
    """
    Implements Distance to Clusters OOD detection.

    This method calculates the distance to the nearest cluster center for each sample.
    The distance is measured as the angle between the input features and the cluster centers.
    It supports multiple subclusters per class for more fine-grained modeling.
    Ideally, the cluster centers should be given to the model as they are the weights of the final layer of the model.
    One can use pytorch_metric_learning library to train with ArcFace Loss to get the cluster centers.

    Attributes:
        model (Callable[[Tensor], Tensor]): The feature extraction model.
        subclusters (int): The number of subclusters per class.
        return_cluster_index (bool): Whether to return the index of the nearest cluster.
        clusters (Optional[Tensor]): The cluster centers after fitting.
        n_classes (Optional[int]): The number of classes in the dataset.
    """

    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
        subclusters: int = 1,
        return_cluster_index: bool = False
    ):
        """
        Initialize the Distance2Clusters detector.

        Args:
            model (Callable[[Tensor], Tensor]): The feature extraction model.
            subclusters (int, optional): The number of subclusters per class. Defaults to 1.
            return_cluster_index (bool, optional): Whether to return the index of the nearest cluster. Defaults to False.
        """
        super().__init__()
        self.model = model
        self.subclusters = subclusters
        self.return_cluster_index = return_cluster_index
        self.clusters: Optional[Tensor] = None
        self.n_classes: Optional[int] = None

    def fit(self, data_loader: DataLoader, device: Optional[str] = None) -> Self:
        """
        Fit the detector to the data.

        This method extracts features from the data loader and calls fit_features.

        Args:
            data_loader (DataLoader): The data loader containing the training data.
            device (Optional[str], optional): The device to use for computations. Defaults to None.

        Returns:
            Self: The fitted detector.
        """
        raise NotImplemented # not tested yet

        device = device or next(self.model.parameters()).device
        z, y = extract_features(data_loader, self.model, device)
        return self.fit_features(z, y, device)

    def fit_features(self, z: Tensor, y: Tensor, device: Optional[str] = None) -> Self:
        """
        Fit the detector to pre-extracted features.

        This method calculates cluster centers for each class and subclass.

        Args:
            z (Tensor): The extracted features.
            y (Tensor): The corresponding labels.
            device (Optional[str], optional): The device to use for computations. Defaults to None.

        Returns:
            Self: The fitted detector.
        """
        raise NotImplemented # not tested yet
        device = device or next(self.model.parameters()).device
        z, y = z.to(device), y.to(device)
        
        classes = y.unique()
        self.n_classes = len(classes)
        
        clusters = []
        for clazz in range(self.n_classes):
            zs = z[y == clazz]
            if self.subclusters == 1:
                # If only one subcluster, simply take the mean
                clusters.append(zs.mean(dim=0, keepdim=True))
            else:
                # Use K-means for multiple subclusters
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.subclusters).fit(zs.cpu().numpy())
                clusters.append(torch.tensor(kmeans.cluster_centers_, device=device))
        
        # Normalize all cluster centers
        self.clusters = torch.nn.functional.normalize(torch.cat(clusters), p=2, dim=-1)
        return self

    def predict_features(self, x: Tensor, y: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Predict the OOD score for pre-extracted features.

        This method calculates the angular distance between input features and cluster centers.

        Args:
            x (Tensor): The input features.
            y (Optional[Tensor], optional): The corresponding labels. If provided, only considers
                                            subclusters of the given class. Defaults to None.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: The OOD scores (distances) and optionally the indices
                                             of the nearest clusters.

        Raises:
            RequiresFittingException: If the detector hasn't been fitted yet.
        """
        if self.clusters is None:
            raise RequiresFittingException("Model needs to be fitted before prediction.")

        # Normalize input features
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        
        # Calculate cosine similarity
        cos_sim = torch.matmul(x_norm, self.clusters.T)
        
        # Convert to angles (in degrees)
        angles = torch.arccos(cos_sim.clamp(-1, 1)) * (180 / torch.pi)

        if y is None:
            # If no labels provided, find the minimum distance to any cluster
            distances, indices = torch.min(angles, dim=1)
        else:
            # If labels provided, only consider subclusters of the given class
            angles = angles.view(x.shape[0], self.n_classes, self.subclusters)
            relevant_angles = angles[torch.arange(x.shape[0]), y]
            distances, indices = torch.min(relevant_angles, dim=1)

        return (distances, indices) if self.return_cluster_index else distances

    def predict(self, x: Tensor, y: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Predict the OOD score for input samples.

        This method first extracts features using the model, then calls predict_features.

        Args:
            x (Tensor): The input samples.
            y (Optional[Tensor], optional): The corresponding labels. If provided, only considers
                                            subclusters of the given class. Defaults to None.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: The OOD scores (distances) and optionally the indices
                                             of the nearest clusters.
        """
        z = self.model(x)
        return self.predict_features(z, y)