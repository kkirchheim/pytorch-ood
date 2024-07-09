import logging
from typing import Callable, TypeVar, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_ood.utils import extract_features, is_known, contains_unknown
from pytorch_ood.api import  Detector, RequiresFittingException

log = logging.getLogger(__name__)
Self = TypeVar("Self")

def _circfuncs_common(samples, high, low):
    sin_samp = torch.sin((samples - low) * 2. * torch.pi / (high - low))
    cos_samp = torch.cos((samples - low) * 2. * torch.pi / (high - low))
    return samples, sin_samp, cos_samp

def circmean(samples, high=2*torch.pi, low=0, axis=None):
    "Compute the circular mean of a set of samples. Assumes samples are in radians."
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_sum = torch.sum(sin_samp, axis)
    cos_sum = torch.sum(cos_samp, axis)
    res = torch.atan2(sin_sum, cos_sum) % (2*torch.pi)
    return res * (high - low) / (2. * torch.pi) + low
import logging


class Distance2Clusters(Detector):
    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
        cluster_centers: Optional[Tensor] = None,
        subclusters: int = 1,
        n_classes: Optional[int] = None,
        return_cluster_index: bool = False,
    ):
        super().__init__()
        self.model = model
        self.cluster_centers = cluster_centers
        self.subclusters = subclusters
        self.n_classes = n_classes
        self.return_cluster_index = return_cluster_index
        if self.cluster_centers is not None:
            self.sanity_check()

    def sanity_check(self):
        if self.cluster_centers is None:
            return
        n_clusters = self.n_classes * self.subclusters
        self.cluster_centers = torch.nn.functional.normalize(self.cluster_centers, p=2, dim=1)
        assert n_clusters == self.cluster_centers.shape[1], "Number of clusters should be equal to n_classes * subclusters"

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
        if self.cluster_centers is None:
            raise RequiresFittingException("Model needs to be fitted before prediction.")

        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        cos_sim = torch.matmul(x_norm, self.cluster_centers)
        angles = torch.arccos(cos_sim.clamp(-1, 1)) * (180 / torch.pi)

        if y is None:
            distances, indices = torch.min(angles, dim=1)
        else:
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