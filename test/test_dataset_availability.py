import unittest
from urllib.request import urlopen

from oodtk.dataset.img.cifar import CIFAR10C, CIFAR100C
from oodtk.dataset.img.imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetR
from oodtk.dataset.img.mnistc import MNISTC


class TestDatasetAvailability(unittest.TestCase):
    def test_download_ImageNetA(self):
        status = urlopen(ImageNetA.url).getcode()
        self.assertEqual(status,200)

    def test_download_ImageNetO(self):
        status = urlopen(ImageNetO.url).getcode()
        self.assertEqual(status,200)

    def test_download_ImageNetR(self):
        status = urlopen(ImageNetR.url).getcode()
        self.assertEqual(status,200)

    def test_download_ImageNetC(self):
        status = urlopen(ImageNetC.url_list[0]).getcode()
        self.assertEqual(status,200)

    def test_download_CIFAR10C(self):
        status = urlopen(CIFAR10C.url).getcode()
        self.assertEqual(status,200)

    def test_download_CIFAR100C(self):
        status = urlopen(CIFAR100C.url).getcode()
        self.assertEqual(status,200)

    def test_download_MNISTC(self):
        status = urlopen(MNISTC.urls[0]).getcode()
        self.assertEqual(status,200)


if __name__ == '__main__':
    unittest.main()