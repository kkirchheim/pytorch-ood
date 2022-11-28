import unittest

from src.pytorch_ood.dataset.img import GaussianNoise, PixMixDataset, UniformNoise


class TestPixMix(unittest.TestCase):
    def test_download_ImageNetA(self):
        a = UniformNoise(length=100)
        b = GaussianNoise(length=200)

        data = PixMixDataset(dataset=a, mixing_set=b)

        for i in range(100):
            self.assertEqual(data[i][0].shape, (3, 224, 224))
