import os

from torchvision.datasets import ImageFolder, VisionDataset


class TinyImagenet(VisionDataset):
    """
    Small Version of the ImageNet with images of size :math:`64 \times 64` from 200 classes.

    Automatic downloading is not supported.
    """

    subsets = ["train", "val", "test"]

    def __init__(self, root, subset, transform=None, target_transform=None):
        """
        :param root: root folder
        """
        super(TinyImagenet, self).__init__(
            root, target_transform=target_transform, transform=transform
        )

        if subset not in self.subsets:
            raise ValueError()

        self.root = os.path.join(root, subset)
        self.data = ImageFolder(root=root)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)
