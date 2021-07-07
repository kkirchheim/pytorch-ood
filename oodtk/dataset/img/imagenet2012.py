"""

"""
from torchvision.datasets import ImageFolder
import os

from oodtk.dataset import OSRVisionDataset


class Imagenet2012(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, train=True, **kwargs):
        if train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "val")
        super(Imagenet2012, self).__init__(root, transform=transform, target_transform=target_transform)


class Imagenet2012_64x64(OSRVisionDataset):
    """
    Downscaled version of the imagenet dataset
    """
    def __init__(self, root, **kwargs):
        super(Imagenet2012_64x64, self).__init__(root)

        self.dataset = ImageFolder(root=os.path.join(root))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)
