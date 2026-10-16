"""
The download of the OpenMIBOOD datasets may only skip certificate verification if the
certificate can not be verified and the file can be checked with its MD5 checksum afterwards.
"""

import hashlib
import io
import os
import ssl
import tempfile
import unittest
import urllib.error
from unittest import mock

from src.pytorch_ood.dataset.img import openmibood

URL = "https://example.org/data.zip"
CONTENT = b"some data"
MD5 = hashlib.md5(CONTENT).hexdigest()


def _certificate_error():
    return urllib.error.URLError(ssl.SSLCertVerificationError("unable to get local issuer"))


class TestDownloadFallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.fpath = os.path.join(self.root, "data.zip")

    def tearDown(self):
        self.tmp.cleanup()

    def _download(self, error, md5, content=CONTENT):
        with (
            mock.patch.object(openmibood, "download_url", side_effect=error),
            mock.patch.object(
                openmibood.urllib.request, "urlopen", return_value=io.BytesIO(content)
            ) as urlopen,
        ):
            try:
                openmibood._download_url_with_ssl_fallback(URL, self.root, "data.zip", md5=md5)
            finally:
                self.urlopen = urlopen

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
        self.assertFalse(os.path.exists(self.fpath))

    def test_detects_chained_certificate_error(self):
        try:
            try:
                raise ssl.SSLCertVerificationError("unable to get local issuer")
            except ssl.SSLError as e:
                raise urllib.error.URLError("download failed") from e
        except urllib.error.URLError as e:
            self.assertTrue(openmibood._is_certificate_error(e))
        self.assertFalse(openmibood._is_certificate_error(ValueError("other")))


if __name__ == "__main__":
    unittest.main()
