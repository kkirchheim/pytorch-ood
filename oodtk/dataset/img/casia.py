from torchvision.datasets import ImageFolder
import logging

log = logging.getLogger(__name__)


class CASIA(ImageFolder):
    """
    Casia Dataset, presented in *Learning Face Representation from Scratch*

    494414 Images, 10576 Classes

    :see Paper: https://arxiv.org/pdf/1411.7923.pdf
    """

    train_annot_file = "train.p"
    test_annot_file = "test.p"

    def __init__(self, root, transform=None, target_transform=None, **kwargs):

        super(CASIA, self).__init__(root, transform=transform, target_transform=target_transform)

