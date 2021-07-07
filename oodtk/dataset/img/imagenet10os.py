from torchvision.datasets import ImageFolder


class ImageNet2010OS(ImageFolder):
    """
    Imagenet 10 Open Set

    Classes from the imagenet that have been removed in later versions of the imagenet
    """
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        super(ImageNet2010OS, self).__init__(root, transform=transform, target_transform=target_transform)

