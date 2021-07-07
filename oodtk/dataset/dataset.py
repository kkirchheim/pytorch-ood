"""
Base Class for Datasets
"""
import abc

import numpy as np
from torch.utils.data import Dataset


class OSRDataset(Dataset):
    """
    A Basic Dataset
    """
    def __init__(self):
        super(OSRDataset, self).__init__()
        self.target_transform = None
        self.transforms = None

    @property
    @abc.abstractmethod
    def unique_targets(self) -> np.ndarray:
        """
        List of possible target classes in this dataset. Required for OSR.
        """
        raise NotImplementedError


class OSRVisionDataset(OSRDataset):
    """
    A Dataset for vision tasks
    """
    
    def __init__(self, root):
        super(OSRVisionDataset, self).__init__(root)

