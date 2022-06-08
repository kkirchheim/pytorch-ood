import unittest
from urllib.request import urlopen

from src.pytorch_ood.dataset.img.cifar import CIFAR10C, CIFAR100C
from src.pytorch_ood.dataset.img.imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetR
from src.pytorch_ood.dataset.img.mnistc import MNISTC
from src.pytorch_ood.dataset.img.streethazards import StreetHazards
from src.pytorch_ood.dataset.txt import Multi30k, NewsGroup20, Reuters52, WikiText2, WikiText103, Reuters8


class TestDatasetAvailability(unittest.TestCase):
    def test_download_ImageNetA(self):
        status = urlopen(ImageNetA.url).getcode()
        self.assertEqual(status, 200)

    def test_download_ImageNetO(self):
        status = urlopen(ImageNetO.url).getcode()
        self.assertEqual(status, 200)

    def test_download_ImageNetR(self):
        status = urlopen(ImageNetR.url).getcode()
        self.assertEqual(status, 200)

    # @unittest.skip("Unavailable")
    def test_download_ImageNetC(self):
        status = urlopen(ImageNetC.url_list[0]).getcode()
        self.assertEqual(status, 200)

    # @unittest.skip("Unavailable")
    def test_download_CIFAR10C(self):
        status = urlopen(CIFAR10C.url).getcode()
        self.assertEqual(status, 200)

    # @unittest.skip("Unavailable")
    def test_download_CIFAR100C(self):
        status = urlopen(CIFAR100C.url).getcode()
        self.assertEqual(status, 200)

    # @unittest.skip("Unavailable")
    def test_download_MNISTC(self):
        status = urlopen(MNISTC.urls[0]).getcode()
        self.assertEqual(status, 200)
    
    def test_download_StreetHazards(self):
        status = urlopen(StreetHazards.url_list[0]).getcode()
        self.assertEqual(status, 200)
        status = urlopen(StreetHazards.url_list[1]).getcode()
        self.assertEqual(status, 200)
        status = urlopen(StreetHazards.url_list[2]).getcode()
        self.assertEqual(status, 200)

    def test_download_Wiki2(self):
        status = urlopen(WikiText2.url).getcode()
        self.assertEqual(status, 200)

    def test_download_Wiki103(self):
        status = urlopen(WikiText103.url).getcode()
        self.assertEqual(status, 200)

    def test_download_Reuters52(self):
        status = urlopen(Reuters52.train_url).getcode()
        self.assertEqual(status, 200)
        status = urlopen(Reuters52.test_url).getcode()
        self.assertEqual(status, 200)
    
    def test_download_Reuters8(self):
        status = urlopen(Reuters8.train_url).getcode()
        self.assertEqual(status, 200)
        status = urlopen(Reuters8.test_url).getcode()
        self.assertEqual(status, 200)

    def test_download_Multi30k(self):
        status = urlopen(Multi30k.test_url).getcode()
        self.assertEqual(status, 200)

        status = urlopen(Multi30k.train_url).getcode()
        self.assertEqual(status, 200)

    def test_download_NewsGroup20(self):
        status = urlopen(NewsGroup20.test_url).getcode()
        self.assertEqual(status, 200)

        status = urlopen(NewsGroup20.train_url).getcode()
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
