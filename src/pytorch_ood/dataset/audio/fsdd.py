import glob
import logging
import os
from os.path import join
from pathlib import Path

from scipy.io import wavfile
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

log = logging.getLogger(__name__)


class FSDD(Dataset):
    """
    Free Spoken Digit Dataset, a simple audio/speech dataset consisting of recordings of spoken
    digits in `wav` format at 8kHz.

    :see Website: `GitHub <https://github.com/Jakobovski/free-spoken-digit-dataset>`__
    """

    metadata = {
        "jackson": {"gender": "male", "accent": "USA/neutral", "language": "english"},
        "nicolas": {"gender": "male", "accent": "BE/French", "language": "english"},
        "theo": {"gender": "male", "accent": "USA/neutral", "language": "english"},
    }

    url = "https://zenodo.org/record/1342401/files/Jakobovski/free-spoken-digit-dataset-v1.0.8.zip"
    md5 = "54f48186ecb1d5ac4e971143086d529b"
    filename = "free-spoken-digit-dataset-v1.0.8.zip"
    base_folder = "Jakobovski-free-spoken-digit-dataset-e9e1155/recordings"

    def __init__(self, root, transform=None, target_transform=None, download=True):
        """
        :param root: root folder for the dataset
        :param transform: transform that will be applied to the instance
        :param target_transform: transform that will be applied to the label
        :param download: set true if you want to download dataset automatically
        """
        super(Dataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.transforms = transform
        self.target_transform = target_transform

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self._data = self._load_data()

    def _download(self):
        if self._check_integrity():
            log.info("Files already downloaded and verified")
            return

        download_and_extract_archive(
            url=self.url, download_root=self.root, extract_root=self.root, md5=self.md5
        )

    def _load_data(self):
        return list(glob.glob(join(self.root, self.base_folder, "*.wav")))

    def _check_integrity(self) -> bool:
        try:
            if len(self._load_data()) > 0:
                return True
            return False
        except Exception as e:
            return False

    def __getitem__(self, index):
        """
        Returns a tuple with the instance and the corresponding label
        """
        file_path = self._data[index]
        label, speaker, _ = (Path(file_path).name).split("_")
        sample_rate, waveform = wavfile.read(file_path)

        label = int(label)

        if self.transforms:
            waveform = self.transforms(waveform)

        if self.target_transform:
            label = self.target_transform(label)

        return waveform, int(label)

    def __len__(self):
        return len(self._data)
