"""
Tests for the download and preparation of the OpenMIBOOD datasets. They use synthetic data and
mock all network access.
"""

import hashlib
import io
import os
import ssl
import sys
import tarfile
import tempfile
import unittest
import urllib.error
from os.path import exists, join
from unittest import mock

from PIL import Image

from pytorch_ood.benchmark.img import _openmibood_download as data

URL = "https://example.org/data.zip"
CONTENT = b"some data"
MD5 = hashlib.md5(CONTENT).hexdigest()


def _certificate_error():
    return urllib.error.URLError(ssl.SSLCertVerificationError("unable to get local issuer"))


def _write_to(content):
    """
    Side effect for a mocked download_url that writes ``content`` to the requested file.
    """

    def download_url(url, root, filename, md5=None):
        with open(join(root, filename), "wb") as f:
            f.write(content)

    return download_url


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.fpath = join(self.root, "data.zip")

    def tearDown(self):
        self.tmp.cleanup()

    def _download(self, error, md5, content=CONTENT):
        download_url = mock.patch.object(data, "download_url", side_effect=error)
        urlopen_patch = mock.patch.object(
            data.urllib.request, "urlopen", return_value=io.BytesIO(content)
        )
        with download_url, urlopen_patch as urlopen:
            try:
                data._download(URL, self.root, "data.zip", md5=md5)
            finally:
                self.urlopen = urlopen

    def test_download_writes_part_file_and_renames(self):
        with mock.patch.object(data, "download_url", side_effect=_write_to(CONTENT)) as dl:
            path = data._download(URL, self.root, "data.zip", md5=MD5)
        self.assertEqual(dl.call_args.kwargs["filename"], "data.zip.part")
        self.assertEqual(path, self.fpath)
        self.assertEqual(os.listdir(self.root), ["data.zip"])

    def test_interrupted_download_leaves_no_file(self):
        def interrupted(url, root, filename, md5=None):
            with open(join(root, filename), "wb") as f:
                f.write(b"partial")
            raise ConnectionResetError()

        with mock.patch.object(data, "download_url", side_effect=interrupted):
            with self.assertRaises(ConnectionResetError):
                data._download(URL, self.root, "data.zip")
        self.assertFalse(exists(self.fpath))

        # the next download replaces the partial file
        with mock.patch.object(data, "download_url", side_effect=_write_to(CONTENT)):
            data._download(URL, self.root, "data.zip")
        with open(self.fpath, "rb") as f:
            self.assertEqual(f.read(), CONTENT)

    def test_existing_file_is_not_downloaded_again(self):
        with open(self.fpath, "wb") as f:
            f.write(CONTENT)
        with mock.patch.object(data, "download_url") as dl:
            data._download(URL, self.root, "data.zip", md5=MD5)
        dl.assert_not_called()

    def test_existing_file_with_wrong_md5_is_downloaded_again(self):
        with open(self.fpath, "wb") as f:
            f.write(b"corrupted")
        with mock.patch.object(data, "download_url", side_effect=_write_to(CONTENT)) as dl:
            data._download(URL, self.root, "data.zip", md5=MD5)
        dl.assert_called_once()

    def test_certificate_error_with_md5_falls_back(self):
        self._download(_certificate_error(), MD5)
        self.urlopen.assert_called_once()
        with open(self.fpath, "rb") as f:
            self.assertEqual(f.read(), CONTENT)

    def test_certificate_error_without_md5_raises(self):
        with self.assertRaises(urllib.error.URLError):
            self._download(_certificate_error(), None)
        self.urlopen.assert_not_called()

    def test_other_error_raises(self):
        with self.assertRaises(urllib.error.URLError):
            self._download(urllib.error.URLError("connection refused"), MD5)
        self.urlopen.assert_not_called()

    def test_wrong_md5_after_fallback_raises(self):
        with self.assertRaises(RuntimeError):
            self._download(_certificate_error(), MD5, content=b"tampered data")
        self.assertEqual(os.listdir(self.root), [])

    def test_detects_chained_certificate_error(self):
        try:
            try:
                raise ssl.SSLCertVerificationError("unable to get local issuer")
            except ssl.SSLError as e:
                raise urllib.error.URLError("download failed") from e
        except urllib.error.URLError as e:
            self.assertTrue(data._is_certificate_error(e))
        self.assertFalse(data._is_certificate_error(ValueError("other")))


class TestExtractTar(unittest.TestCase):
    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = join(tmp, "evil.tar")
            with tarfile.open(archive, "w") as tf:
                info = tarfile.TarInfo("../evil.txt")
                info.size = 4
                tf.addfile(info, io.BytesIO(b"evil"))
            with self.assertRaises(Exception):
                data._extract_tar(archive, join(tmp, "out"))
            self.assertFalse(exists(join(tmp, "evil.txt")))


class TestMIDOGPatches(unittest.TestCase):
    def test_patches(self):
        # left half black, right half white, so that the content of the patches can be checked
        image = Image.new("RGB", (300, 200), color=(0, 0, 0))
        image.paste((255, 255, 255), (150, 0, 300, 200))
        img_data = {"id": 7, "width": 300, "height": 200}
        annotations = [
            # mitotic figure, the background patch [110, 10, 160, 60] overlaps nothing
            {"id": 1, "category_id": 1, "bbox": [10, 10, 60, 60]},
            # imposter, its background patch would leave the image
            {"id": 2, "category_id": 2, "bbox": [200, 100, 250, 150]},
            # not 50x50, ignored
            {"id": 3, "category_id": 1, "bbox": [0, 150, 40, 190]},
            # background patch of this one overlaps annotation 2
            {"id": 4, "category_id": 1, "bbox": [110, 110, 160, 160]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            data._extract_midog_patches(image, img_data, annotations, tmp)
            self.assertEqual(
                sorted(os.listdir(tmp)),
                ["007_1_0.tiff", "007_1_1.tiff", "007_2_2.tiff", "007_4_1.tiff"],
            )
            mitosis = Image.open(join(tmp, "007_1_1.tiff"))
            self.assertEqual(mitosis.size, (50, 50))
            self.assertEqual(mitosis.getpixel((0, 0)), (0, 0, 0))
            # the background patch is shifted by 100 px and covers [110, 160), across the border
            background = Image.open(join(tmp, "007_1_0.tiff"))
            self.assertEqual(background.getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(background.getpixel((49, 0)), (255, 255, 255))


class TestCCAgTPatches(unittest.TestCase):
    def test_patches(self):
        coco = {
            "images": [{"id": 1, "file_name": "A_1"}, {"id": 2, "file_name": "B_2"}],
            "annotations": [
                # nucleus with center (50, 50)
                {"id": 10, "image_id": 1, "category_id": 1, "bbox": [40, 40, 20, 20]},
                # cluster, skipped
                {"id": 11, "image_id": 1, "category_id": 2, "bbox": [40, 40, 20, 20]},
                # patch would leave the image
                {"id": 12, "image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10]},
                # image file does not exist
                {"id": 13, "image_id": 2, "category_id": 1, "bbox": [40, 40, 20, 20]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(join(tmp, "images", "A"))
            Image.new("RGB", (100, 80), color=(10, 20, 30)).save(
                join(tmp, "images", "A", "A_1.jpg")
            )
            out = join(tmp, "out")
            data._extract_ccagt_patches(coco, join(tmp, "images"), out)
            self.assertEqual(os.listdir(out), ["A_1"])
            self.assertEqual(os.listdir(join(out, "A_1")), ["10_1.jpg"])
            self.assertEqual(Image.open(join(out, "A_1", "10_1.jpg")).size, (50, 50))


class TestCHAOS(unittest.TestCase):
    def test_download_without_simpleitk_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, {"SimpleITK": None}):
                with mock.patch.object(data, "_download") as download:
                    with self.assertRaises(RuntimeError) as ctx:
                        data._download_chaos(tmp)
            self.assertIn("SimpleITK", str(ctx.exception))
            download.assert_not_called()


class TestSource(unittest.TestCase):
    def test_midog_requires_all_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            for target in data.MIDOG.targets[:-1]:
                os.makedirs(join(tmp, target, "001"))
            self.assertFalse(data.MIDOG.is_present(tmp))
            os.makedirs(join(tmp, data.MIDOG.targets[-1], "001"))
            self.assertTrue(data.MIDOG.is_present(tmp))

    def test_midog_table_covers_all_slides(self):
        names = set(data._MIDOG_FIGSHARE_FILES.values())
        self.assertEqual(len(names), 503)


if __name__ == "__main__":
    unittest.main()
