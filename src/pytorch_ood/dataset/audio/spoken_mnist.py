import logging
import os
import glob
from typing import Tuple
from pathlib import Path

from scipy.io import wavfile

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

log = logging.getLogger(__name__)


class SpokenMNIST(Dataset):
    """
    A free audio dataset of spoken digits. Think MNIST for audio.

    :see Website: https://github.com/Jakobovski/free-spoken-digit-dataset
    """

    metadata = {
    'jackson': {
        'gender': 'male',
        'accent': 'USA/neutral',
        'language': 'english'
    },
    'nicolas': {
    	'gender': 'male',
    	'accent': 'BE/French',
    	'language': 'english'
    },
    'theo': {
    	'gender': 'male',
    	'accent': 'USA/neutral',
    	'language': 'english'
    }
    }

    url = "https://zenodo.org/record/1342401/files/Jakobovski/free-spoken-digit-dataset-v1.0.8.zip"
    md5 = "54f48186ecb1d5ac4e971143086d529b"
    filename = "free-spoken-digit-dataset-v1.0.8.zip"
    base_folder = "Jakobovski-free-spoken-digit-dataset-e9e1155/recordings"


    def __init__(self, root, transform=None, download=True):
        super(Dataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.transforms = transform

        if download:
            self._download()

        self._data = self._load_data()

    def _download(self):
        if self._check_integrity():
            log.info("Files already downloaded and verified")
            return

        download_and_extract_archive(
            url=self.url, download_root=self.root, extract_root=self.root, md5=self.md5
        )

    def _load_data(self) -> Tuple:
        return [f for f in glob.glob(os.path.join(self.root, self.base_folder, "*.wav"))]

    def _check_integrity(self):
        try:
            self._load_data()
        except Exception as e:
            # log.exception(e)
            return False

        return True

    def __getitem__(self, index):
        file_path = self._data[index]
        rating, fileType, _ = (Path(file_path).name).split("_")
        sample_rate, waveform = wavfile.read(file_path) 

        if self.transforms:
            waveform = self.target_transform(waveform)

        return waveform, sample_rate, int(rating) , self.metadata[fileType]

    def __len__(self):
        return len(self._data)


