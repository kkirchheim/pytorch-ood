"""
Imglist-driven OpenOOD v1.5 benchmarks.

These classes reproduce the OpenOOD v1.5 splits *exactly* by reading the upstream
``benchmark_imglist`` files (downloaded once on demand from OpenOOD's own distribution and
cached under ``torch.hub``'s cache dir -- never shipped with the package) and loading the
referenced images with
:class:`ImageListDataset <pytorch_ood.dataset.img.ImageListDataset>`.

Every split -- the 9000-image ID test set, the held-out ID/OOD *validation* splits used
for hyperparameter tuning (APS), and each near-/far-OOD test set -- matches OpenOOD's
curated subsets rather than the full source datasets, so results are directly comparable
to the OpenOOD leaderboard.

With ``download=True`` (the default) each referenced dataset is fetched from OpenOOD's
own Google-Drive distribution and extracted into ``root`` in the expected layout
(``<root>/<dataset>/...``), mirroring OpenOOD's ``scripts/download/download.py``. Requires
the optional ``gdown`` dependency. ImageNet-1K itself is not redistributable and is never
downloaded -- the ImageNet benchmarks take an ``image_net_root`` pointing at a local copy.
"""

import logging
import os
from typing import ClassVar, Dict, List, Optional, Tuple

import torch
from torchvision.transforms import Compose

from torch.utils.data import ConcatDataset, Dataset

from ..base import Benchmark
from pytorch_ood.dataset.img import ImageListDataset
from pytorch_ood.utils import ToRGB

log = logging.getLogger(__name__)


# Google-Drive file ids for OpenOOD's per-dataset image archives, from
# https://github.com/Jingkang50/OpenOOD scripts/download/download.py. Each archive
# extracts (bare) into ``<root>/<name>/``.
_OPENOOD_GDRIVE_IDS: Dict[str, str] = {
    "cifar10": "1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1",
    "cifar100": "1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_",
    "tin": "1PZ-ixyx52U989IKsMA2OT-24fToTrelC",
    "mnist": "1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb",
    "svhn": "1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI",
    "texture": "1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam",
    "places365": "1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay",
    "ssb_hard": "1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE",
    "ninco": "1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6",
    "inaturalist": "1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj",
    "openimage_o": "1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE",
}


# Google-Drive file id for OpenOOD's ``benchmark_imglist.zip`` (~28 MB, all splits for every
# OpenOOD benchmark), from https://github.com/Jingkang50/OpenOOD scripts/download/download.py.
# Extracts to ``<cache>/benchmark_imglist/<subdir>/<fname>``.
_IMGLIST_GDRIVE_ID = "1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1"


def _ensure_imglists(download: bool) -> str:
    """
    Ensure OpenOOD's ``benchmark_imglist`` bundle is available locally and return the
    directory containing the per-benchmark imglist subdirs (so a split resolves to
    ``<returned>/<subdir>/<fname>``).

    The bundle is fetched once from OpenOOD's own Google-Drive distribution and cached
    under ``torch.hub``'s cache dir, shared across every benchmark instance -- the imglists
    are therefore never shipped with the package (they total ~100 MB uncompressed). No-op
    if already present.
    """
    cache = os.path.join(torch.hub.get_dir(), "openood_benchmark_imglist")
    imglist_dir = os.path.join(cache, "benchmark_imglist")
    if os.path.isdir(imglist_dir) and os.listdir(imglist_dir):
        return imglist_dir

    if not download:
        raise RuntimeError(
            f"OpenOOD imglists not found under {imglist_dir!r} and download=False. "
            "Pass download=True to fetch them (requires gdown), or place OpenOOD's "
            "benchmark_imglist/ directory there manually."
        )

    try:
        import gdown  # optional dependency
    except ImportError as e:  # pragma: no cover - exercised only without gdown
        raise RuntimeError(
            "Downloading OpenOOD imglists requires the 'gdown' package (pip install gdown), "
            "or provide benchmark_imglist/ manually and pass download=False."
        ) from e

    import zipfile

    os.makedirs(cache, exist_ok=True)
    zip_path = os.path.join(cache, "benchmark_imglist.zip")
    log.info("Downloading OpenOOD benchmark_imglist -> %s", cache)
    gdown.download(id=_IMGLIST_GDRIVE_ID, output=zip_path, quiet=False)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache)
    os.remove(zip_path)
    return imglist_dir


def _download_openood_dataset(name: str, root: str) -> None:
    """
    Fetch OpenOOD's archive for ``name`` and extract it into ``<root>/<name>/``,
    matching OpenOOD's own ``download_dataset``. No-op if the target already exists.
    """
    dest = os.path.join(root, name)
    if os.path.isdir(dest) and os.listdir(dest):
        return

    gid = _OPENOOD_GDRIVE_IDS.get(name)
    if gid is None:
        raise RuntimeError(
            f"No OpenOOD download id known for '{name}'. Provide the data manually under "
            f"{dest!r}, or set download=False."
        )

    try:
        import gdown  # optional dependency
    except ImportError as e:  # pragma: no cover - exercised only without gdown
        raise RuntimeError(
            "Downloading OpenOOD datasets requires the 'gdown' package "
            "(pip install gdown), or provide the data manually and pass download=False."
        ) from e

    import zipfile

    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, f"{name}.zip")
    log.info("Downloading OpenOOD dataset '%s' -> %s", name, dest)
    gdown.download(id=gid, output=zip_path, quiet=False)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    os.remove(zip_path)


class _OpenOOD_Imglist(Benchmark):
    """
    Shared base for imglist-driven OpenOOD benchmarks. Subclasses declare the imglist
    filenames and the near-/far-OOD grouping; this base wires them into
    :class:`ImageListDataset` splits and (optionally) downloads the referenced data.
    """

    _subdir: ClassVar[str]
    _train_imglist: ClassVar[str]
    _val_id_imglist: ClassVar[str]
    _test_id_imglist: ClassVar[str]
    _val_ood_imglist: ClassVar[str]
    #: ``(display_name, imglist_filename)`` per OOD test set, near-OOD first then far-OOD
    _ood_imglists: ClassVar[List[Tuple[str, str]]]
    _near_ood: ClassVar[List[str]]
    _far_ood: ClassVar[List[str]]
    #: top-level dataset dirs (under ``root``) to auto-download when ``download=True``
    _download_names: ClassVar[List[str]]

    def __init__(self, root, transform, download: bool = True) -> None:
        """
        :param root: directory the imglist paths resolve against (OpenOOD's
            ``images_classic`` / ``images_largescale`` layout)
        :param transform: transform applied to every image (``ToRGB`` is prepended, so
            grayscale OOD sets like MNIST work with 3-channel models)
        :param download: if ``True``, fetch any missing referenced dataset from OpenOOD's
            distribution into ``root`` (requires ``gdown``)
        """
        self.root = root
        # Prepend ToRGB so callers can pass a model's own transform directly, matching
        # the sweep, and so grayscale sources (MNIST/FashionMNIST) load as 3-channel.
        self.transform = Compose([ToRGB(), transform])

        if download:
            self._download(root)

        imglist_dir = _ensure_imglists(download)

        def _load(fname: str) -> ImageListDataset:
            return ImageListDataset(
                root=root,
                imglist_path=os.path.join(imglist_dir, self._subdir, fname),
                transform=self.transform,
            )

        self.train_in = _load(self._train_imglist)
        self.val_in = _load(self._val_id_imglist)
        self.test_in = _load(self._test_id_imglist)
        self.val_ood = _load(self._val_ood_imglist)
        self.test_oods = [_load(fname) for _, fname in self._ood_imglists]

        self.ood_names: List[str] = [name for name, _ in self._ood_imglists]

    def _download(self, root: str) -> None:
        """Download any missing referenced datasets into ``root``."""
        os.makedirs(root, exist_ok=True)
        for name in self._download_names:
            _download_openood_dataset(name, root)

    def train_set(self) -> Dataset:
        """Training dataset (in-distribution)."""
        return self.train_in

    def validation_set(self) -> Dataset:
        """
        Mixed ID + OOD validation split used by OpenOOD for hyperparameter tuning
        (APS). ID samples keep their class labels; OOD samples are labelled ``-1``.
        Suitable as the ``val_loader`` for :class:`pytorch_ood.utils.GridSearch`.

        Class-conditional calibration detectors (TemperatureScaling, KLMatching) drop
        the ``-1`` samples / never look up the ``-1`` class, so this is also safe to use
        as their calibration set -- and the ID half covers every class.
        """
        return ConcatDataset([self.val_in, self.val_ood])

    def test_sets(self, known=True, unknown=True) -> List[Dataset]:
        """
        List of the different test datasets.
        If known and unknown are true, each dataset contains ID and OOD data.

        :param known: include ID
        :param unknown: include OOD
        """
        if known and unknown:
            return [ConcatDataset([self.test_in, other]) for other in self.test_oods]

        if known and not unknown:
            return [self.test_in]

        if not known and unknown:
            return self.test_oods

        raise ValueError()


class _OpenOOD_ImageNet(_OpenOOD_Imglist):
    """
    Base for the ImageNet-scale OpenOOD benchmarks. ImageNet-1K itself is not
    redistributable, so its images come from a user-provided ``image_net_root`` rather
    than the downloader; only the OOD sets are fetched. The imglists reference
    ``imagenet_1k/train/<wnid>/...`` and flat ``imagenet_1k/val/ILSVRC2012_val_*.JPEG``.
    """

    def __init__(self, root, transform, image_net_root: Optional[str] = None,
                 download: bool = True) -> None:
        """
        :param root: directory the OOD imglist paths resolve against; ``imagenet_1k`` is
            set up here from ``image_net_root``
        :param transform: transform applied to every image
        :param image_net_root: local ImageNet-1K directory (torchvision layout, with
            ``train/<wnid>/`` and the flat validation images available)
        :param download: fetch missing OOD sets into ``root``
        """
        if image_net_root is not None:
            self._link_imagenet(root, image_net_root)
        super().__init__(root, transform, download=download)

    @staticmethod
    def _link_imagenet(root: str, image_net_root: str) -> None:
        """
        Point ``<root>/imagenet_1k/train`` at the ImageNet train dir. The flat
        ``imagenet_1k/val`` split (``ILSVRC2012_val_*.JPEG``) must already be present --
        it is derived from the ImageNet devkit and not created here.
        """
        dest = os.path.join(root, "imagenet_1k")
        os.makedirs(dest, exist_ok=True)
        train_link = os.path.join(dest, "train")
        train_src = os.path.join(image_net_root, "train")
        if not os.path.exists(train_link) and os.path.isdir(train_src):
            os.symlink(train_src, train_link)


class CIFAR10_OpenOOD(_OpenOOD_Imglist):
    """
    Exact OpenOOD v1.5 CIFAR-10 benchmark (imglist-driven, auto-downloading).

    Near-OOD: CIFAR-100, TinyImageNet. Far-OOD: MNIST, SVHN, Textures, Places365.
    The hyperparameter-tuning validation split is a held-out 1000-image slice of the
    CIFAR-10 test set (all 10 classes) plus a disjoint TinyImageNet OOD subset.

    :see Paper: `OpenOOD v1.5 <https://arxiv.org/abs/2306.09301>`__
    """

    _subdir = "cifar10"
    _train_imglist = "train_cifar10.txt"
    _val_id_imglist = "val_cifar10.txt"
    _test_id_imglist = "test_cifar10.txt"
    _val_ood_imglist = "val_tin.txt"
    _ood_imglists = [
        ("CIFAR100", "test_cifar100.txt"),
        ("TinyImageNet", "test_tin.txt"),
        ("MNIST", "test_mnist.txt"),
        ("SVHN", "test_svhn.txt"),
        ("Textures", "test_texture.txt"),
        ("Places365", "test_places365.txt"),
    ]
    _near_ood = ["CIFAR100", "TinyImageNet"]
    _far_ood = ["MNIST", "SVHN", "Textures", "Places365"]
    _download_names = ["cifar10", "cifar100", "tin", "mnist", "svhn", "texture", "places365"]


class CIFAR100_OpenOOD(_OpenOOD_Imglist):
    """
    Exact OpenOOD v1.5 CIFAR-100 benchmark (imglist-driven, auto-downloading).

    Near-OOD: CIFAR-10, TinyImageNet. Far-OOD: MNIST, SVHN, Textures, Places365.

    :see Paper: `OpenOOD v1.5 <https://arxiv.org/abs/2306.09301>`__
    """

    _subdir = "cifar100"
    _train_imglist = "train_cifar100.txt"
    _val_id_imglist = "val_cifar100.txt"
    _test_id_imglist = "test_cifar100.txt"
    _val_ood_imglist = "val_tin.txt"
    _ood_imglists = [
        ("CIFAR10", "test_cifar10.txt"),
        ("TinyImageNet", "test_tin.txt"),
        ("MNIST", "test_mnist.txt"),
        ("SVHN", "test_svhn.txt"),
        ("Textures", "test_texture.txt"),
        ("Places365", "test_places365.txt"),
    ]
    _near_ood = ["CIFAR10", "TinyImageNet"]
    _far_ood = ["MNIST", "SVHN", "Textures", "Places365"]
    _download_names = ["cifar100", "cifar10", "tin", "mnist", "svhn", "texture", "places365"]


class ImageNet200_OpenOOD(_OpenOOD_ImageNet):
    """
    Exact OpenOOD v1.5 ImageNet-200 benchmark (imglist-driven).

    In-distribution is the 200-class ImageNet-R subset of ImageNet-1K. Near-OOD:
    SSB-Hard, NINCO. Far-OOD: iNaturalist, Textures, OpenImage-O. The OOD sets
    auto-download; ImageNet-1K comes from ``image_net_root``.

    :see Paper: `OpenOOD v1.5 <https://arxiv.org/abs/2306.09301>`__
    """

    _subdir = "imagenet200"
    _train_imglist = "train_imagenet200.txt"
    _val_id_imglist = "val_imagenet200.txt"
    _test_id_imglist = "test_imagenet200.txt"
    _val_ood_imglist = "val_openimage_o.txt"
    _ood_imglists = [
        ("SSBHard", "test_ssb_hard.txt"),
        ("NINCO", "test_ninco.txt"),
        ("iNaturalist", "test_inaturalist.txt"),
        ("Textures", "test_textures.txt"),
        ("OpenImagesO", "test_openimage_o.txt"),
    ]
    _near_ood = ["SSBHard", "NINCO"]
    _far_ood = ["iNaturalist", "Textures", "OpenImagesO"]
    _download_names = ["ssb_hard", "ninco", "inaturalist", "texture", "openimage_o"]


class ImageNet_OpenOOD(_OpenOOD_ImageNet):
    """
    Exact OpenOOD v1.5 ImageNet-1K benchmark (imglist-driven).

    In-distribution is full 1000-class ImageNet-1K (from ``image_net_root``). Near-OOD:
    SSB-Hard, NINCO. Far-OOD: iNaturalist, Textures, OpenImage-O. The validation split
    (5 images per class, all 1000 classes, plus a held-out OpenImage-O OOD subset)
    covers every class, so class-conditional calibration detectors fit all classes.

    :see Paper: `OpenOOD v1.5 <https://arxiv.org/abs/2306.09301>`__
    """

    _subdir = "imagenet"  # OpenOOD names the 1K benchmark's imglist dir "imagenet"
    _train_imglist = "train_imagenet.txt"
    _val_id_imglist = "val_imagenet.txt"
    _test_id_imglist = "test_imagenet.txt"
    _val_ood_imglist = "val_openimage_o.txt"
    _ood_imglists = [
        ("SSBHard", "test_ssb_hard.txt"),
        ("NINCO", "test_ninco.txt"),
        ("iNaturalist", "test_inaturalist.txt"),
        ("Textures", "test_textures.txt"),
        ("OpenImagesO", "test_openimage_o.txt"),
    ]
    _near_ood = ["SSBHard", "NINCO"]
    _far_ood = ["iNaturalist", "Textures", "OpenImagesO"]
    _download_names = ["ssb_hard", "ninco", "inaturalist", "texture", "openimage_o"]


#: Backwards-compatible alias
ImageNet1K_OpenOOD = ImageNet_OpenOOD
