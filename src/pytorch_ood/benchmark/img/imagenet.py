from typing import List, Dict

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torchvision.datasets import ImageNet, MNIST, SVHN

from pytorch_ood.api import Detector
from pytorch_ood.benchmark import Benchmark
from pytorch_ood.dataset.img import Textures, OpenImagesO, ImageNetO
from pytorch_ood.utils import ToUnknown, ToRGB, OODMetrics


class ImageNet_OpenOOD(Benchmark):
    """
    Aims to replicate the ImageNet benchmark proposed in
    *OpenOOD: Benchmarking Generalized Out-of-Distribution Detection*.

    :see Paper: `OpenOOD <https://openreview.net/pdf?id=gT6j4_tskUt>`__

    Outlier datasets are

     * ImageNet-O
     * OpenImage-O
     * Textures
     * MNIST
     * SVHN
     * Texture

    .. warning :: This currently does not reproduce the benchmark accurately, as it does not exclude images with
        overlap with ImageNet and is missing the Species dataset.
    """

    def __init__(self, root, image_net_root, transform):
        """
        :param root: where to store datasets
        :param image_net_root: root for the ImageNet dataset
        :param transform: transform to apply to images
        """
        self.transform = Compose([ToRGB(), transform])
        self._train_in = None
        self.image_net_root = image_net_root
        self.test_in = ImageNet(image_net_root, transform=self.transform, split="val")

        self.test_oods = [
            ImageNetO(root, download=True, transform=self.transform, target_transform=ToUnknown()),
            OpenImagesO(root, download=True, transform=self.transform, target_transform=ToUnknown()),
            Textures(
                root, download=True, transform=self.transform, target_transform=ToUnknown()
            ),
            SVHN(root, split="test", download=True, transform=self.transform, target_transform=ToUnknown()),
            MNIST(root, root, download=True, transform=self.transform, target_transform=ToUnknown())
        ]

        self.ood_names: List[str] = []  #: OOD Dataset names
        self.ood_names = [type(d).__name__ for d in self.test_oods]

    @property
    def train_in(self):
        # lazy loading only if needed
        if not self._train_in:
            self._train_in = ImageNet(self.image_net_root, split="train", transform=self.transform)

        return self._train_in

    def train_set(self) -> Dataset:
        """
        Training dataset
        """
        return self.train_in

    def test_sets(self, known=True, unknown=True) -> List[Dataset]:
        """
        List of the different test datasets.
        If known and unknown are true, each dataset contains IN and OOD data.

        :param known: include IN
        :param unknown: include OOD
        """

        if known and unknown:
            return [self.test_in + other for other in self.test_oods]

        if known and not unknown:
            return [self.train_in]

        if not known and unknown:
            return self.test_oods

        raise ValueError()

    def evaluate(
            self, detector: Detector, loader_kwargs: Dict = None, device: str = "cpu"
    ) -> List[Dict]:
        """
        Evaluates the given detector on all datasets and returns a list with the results

        :param detector: the detector to evaluate
        :param loader_kwargs: keyword arguments to give to the data loader
        :param device: the device to move batches to
        """
        if loader_kwargs is None:
            loader_kwargs = {}

        metrics = []

        for name, dataset in zip(self.ood_names, self.test_sets()):
            print(name)
            loader = DataLoader(dataset=dataset, **loader_kwargs)

            m = OODMetrics()

            for x, y in loader:
                m.update(detector(x.to(device)), y)

            r = m.compute()
            r.update({"Dataset": name})

            metrics.append(r)

        return metrics
