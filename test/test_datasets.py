import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path

from oodtk.dataset.img.cifar import CIFAR10C, CIFAR10P, CIFAR100C
from oodtk.dataset.img.imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetR
from oodtk.dataset.img.mnistc import MNISTC


class TestDownloadDatasets(unittest.TestCase):
    """
    Test downloading of various datasets
    """

    def _try_download(self, fun, *args, **kwargs):
        p = multiprocessing.Process(target=fun, args=args, kwargs=kwargs)
        p.start()
        time.sleep(5)
        p.terminate()

    def test_download_ImageNetA(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._try_download(ImageNetA, download=True, root=tmp_dir)
            self.assertTrue(self._check_download(tmp_dir))

    def test_download_ImageNetO(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._try_download(ImageNetO, download=True, root=tmp_dir)
            self.assertTrue(self._check_download(tmp_dir))

    def test_download_ImageNetR(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._try_download(ImageNetR, download=True, root=tmp_dir)
            self.assertTrue(self._check_download(tmp_dir))

    def test_download_ImageNetC(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._try_download(ImageNetC, subset="blur", download=True, root=tmp_dir)
            self.assertTrue(self._check_download(tmp_dir))

    # def test_download_ImageNetP(self):
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         self._try_download(ImageNetP, subset="digital", download=True, root=tmp_dir)
    #         self.assertTrue(self._check_download(tmp_dir))

    def test_download_CIFAR10P(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._try_download(CIFAR10P, download=True, root=tmp_dir)
            self.assertTrue(self._check_download(tmp_dir))

    def test_download_CIFAR10C(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._try_download(CIFAR10C, subset="all", download=True, root=tmp_dir)
            self.assertTrue(self._check_download(tmp_dir))

    def test_download_CIFAR100C(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._try_download(CIFAR100C, download=True, root=tmp_dir)
            self.assertTrue(self._check_download(tmp_dir))

    def test_download_MNISTC(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._try_download(MNISTC, subset="all", split="train", download=True, root=tmp_dir)
            self.assertTrue(self._check_download(tmp_dir))

    def _check_download(self, directory):
        files = Path(directory).glob("*")
        for file in files:
            print(file)
            file_size = os.path.getsize(file)
            print(file_size)
            if file_size > 0:
                return True
            else:
                return False

        return False
