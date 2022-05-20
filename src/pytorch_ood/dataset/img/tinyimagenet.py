import os

from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder, VisionDataset


class TinyImagenet(VisionDataset):
    """ """

    def __init__(self, root, **kwargs):
        """

        :param root: root folder
        :param kwargs:
        """
        super(TinyImagenet, self).__init__(root)
        dataset1 = ImageFolder(root=os.path.join(root, "train"))
        dataset2 = ImageFolder(root=os.path.join(root, "val"))
        dataset3 = ImageFolder(root=os.path.join(root, "test"))
        self.dataset = ConcatDataset([dataset1, dataset2, dataset3])
        self.targets = []
        self.targets.extend(dataset1.targets)
        self.targets.extend(dataset2.targets)
        self.targets.extend(dataset3.targets)

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
