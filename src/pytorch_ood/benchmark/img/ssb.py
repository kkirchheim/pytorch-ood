"""
Semantic Split Benchmark (SSB) from *Dissecting Out-of-Distribution Detection and Open-Set
Recognition: A Critical Analysis of Methods and Benchmarks* (IJCV 2024).

SSB provides fine-grained evaluation of OOD detection on fine-grained visual datasets.
Each dataset is partitioned into ID and OOD classes, with OOD classes further split by semantic
similarity: far-OOD (Easy, maximally dissimilar) and near-OOD (Hard/Medium, visually similar to ID).
This enables nuanced evaluation of OOD detection methods under varying difficulty levels.

:see Paper: `ArXiv <https://arxiv.org/abs/2408.16757>`__
:see Repository: `Visual-AI/Dissect-OOD-OSR <https://github.com/Visual-AI/Dissect-OOD-OSR>`__
"""

import logging
import os
import pickle
from os.path import join
from typing import Any, Callable, List, Optional, Tuple

import scipy.io
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, download_url

from pytorch_ood.benchmark import Benchmark
from pytorch_ood.utils import ToUnknown

log = logging.getLogger(__name__)


# ─── SSB Split Files ──────────────────────────────────────────────────────────

_SSB_SPLIT_BASE = (
    "https://raw.githubusercontent.com/Visual-AI/Dissect-OOD-OSR/main/data/open_set_splits"
)
_SSB_SPLIT_FILES = {
    "cub": "cub_osr_splits.pkl",
    "aircraft": "aircraft_osr_splits.pkl",
    "scars": "scars_osr_splits.pkl",
}


def load_ssb_splits(dataset: str, root: str) -> dict:
    """
    Download (once) and return the SSB class splits for a given dataset.

    Files are cached under ``<root>/ssb_splits/``.

    :param dataset: one of ``"cub"``, ``"aircraft"``, ``"scars"``
    :param root: directory used for caching the split file
    :returns: dict with keys:

        - ``known_classes`` — list of 0-indexed class IDs for the ID split
        - ``unknown_classes`` — dict with keys ``"Easy"``, ``"Medium"``, ``"Hard"``
          each mapping to a list of 0-indexed class IDs
    """
    if dataset not in _SSB_SPLIT_FILES:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(_SSB_SPLIT_FILES)}")

    fname = _SSB_SPLIT_FILES[dataset]
    cache_dir = join(root, "ssb_splits")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = join(cache_dir, fname)

    if not os.path.isfile(local_path):
        url = f"{_SSB_SPLIT_BASE}/{fname}"
        log.info("Downloading SSB splits for %s from %s", dataset, url)
        download_url(url, cache_dir, filename=fname)

    with open(local_path, "rb") as f:
        raw = pickle.load(f)

    known = list(raw["known_classes"])
    unk_raw = raw["unknown_classes"]
    unknown = {k: list(v) for k, v in unk_raw.items()}

    # scars pkl uses 1-indexed class IDs — normalise to 0-indexed
    if dataset == "scars":
        known = [c - 1 for c in known]
        unknown = {k: [c - 1 for c in v] for k, v in unknown.items()}

    return {"known_classes": known, "unknown_classes": unknown}


# ─── Private Dataset Classes ──────────────────────────────────────────────────


class _CUB200(VisionDataset):
    """
    CUB-200-2011 Caltech-UCSD Birds dataset.

    Expects the extracted ``CUB_200_2011/`` directory to be present under ``root``.
    Use ``download=True`` to fetch and extract the archive automatically.
    """

    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    base_folder = "CUB_200_2011"

    def __init__(
        self,
        root: str,
        split: str = "train",
        classes: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: root directory of the dataset
        :param split: ``"train"`` or ``"test"``
        :param classes: 0-indexed class IDs to include; labels are remapped to
            ``0..len(classes)-1``. All classes are used when ``None``.
        :param download: if ``True``, download and extract if not already present
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self._download()

        base = join(root, self.base_folder)
        if not os.path.isdir(base):
            raise RuntimeError(
                f"CUB-200-2011 not found at {base}. "
                "Pass download=True to download it automatically."
            )

        with open(join(base, "images.txt")) as f:
            id_to_path = {int(r[0]): r[1] for r in (line.split() for line in f)}

        with open(join(base, "image_class_labels.txt")) as f:
            # 1-indexed in file → 0-indexed here
            id_to_cls = {int(r[0]): int(r[1]) - 1 for r in (line.split() for line in f)}

        with open(join(base, "train_test_split.txt")) as f:
            id_to_split = {int(r[0]): int(r[1]) for r in (line.split() for line in f)}

        is_train = split == "train"
        split_val = 1 if is_train else 0

        self.data: List[Tuple[str, int]] = [
            (join(base, "images", id_to_path[img_id]), id_to_cls[img_id])
            for img_id in sorted(id_to_path)
            if id_to_split[img_id] == split_val
        ]

        if classes is not None:
            classes_set = set(classes)
            class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
            self.data = [
                (path, class_to_idx[cls]) for path, cls in self.data if cls in classes_set
            ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.data[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def _download(self) -> None:
        if os.path.isdir(join(self.root, self.base_folder)):
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename)


class _StanfordCars(VisionDataset):
    """
    Stanford Cars dataset.

    .. note::

        Auto-download is not supported because the original Stanford host is no
        longer available. Download the dataset manually from
        https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
        and extract it so that ``<root>/stanford_cars/`` contains
        ``cars_train/``, ``cars_test/``, and ``devkit/``.
    """

    base_folder = "stanford_cars"

    def __init__(
        self,
        root: str,
        split: str = "train",
        classes: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        :param root: root directory of the dataset
        :param split: ``"train"`` or ``"test"``
        :param classes: 0-indexed class IDs to include; labels are remapped to
            ``0..len(classes)-1``. All classes are used when ``None``.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        base = join(root, self.base_folder)
        if not os.path.isdir(base):
            raise RuntimeError(
                f"Stanford Cars not found at {base}. "
                "Download from https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset "
                f"and extract so that {base}/cars_train/, {base}/cars_test/, and "
                f"{base}/devkit/ are present."
            )

        if split == "train":
            annos_path = join(base, "devkit", "cars_train_annos.mat")
            img_dir = join(base, "cars_train")
        else:
            annos_path = join(base, "devkit", "cars_test_annos_withlabels.mat")
            img_dir = join(base, "cars_test")

        # Array-level field access matches torchvision's StanfordCars and is robust
        # to the nested structure of real Stanford Cars .mat struct arrays.
        annos = scipy.io.loadmat(annos_path, squeeze_me=True)["annotations"]
        self.data: List[Tuple[str, int]] = [
            (join(img_dir, str(fname)), int(cls) - 1)
            for fname, cls in zip(annos["fname"], annos["class"])
        ]

        if classes is not None:
            classes_set = set(classes)
            class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
            self.data = [
                (path, class_to_idx[cls]) for path, cls in self.data if cls in classes_set
            ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.data[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class _FGVCAircraft(VisionDataset):
    """
    FGVC-Aircraft dataset (variant-level, 100 classes).

    Use ``download=True`` to fetch and extract the archive automatically.
    """

    url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    filename = "fgvc-aircraft-2013b.tar.gz"
    base_folder = "fgvc-aircraft-2013b"

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        classes: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: root directory of the dataset
        :param split: one of ``"train"``, ``"val"``, ``"trainval"``, ``"test"``
        :param classes: 0-indexed variant class IDs to include; labels are
            remapped to ``0..len(classes)-1``. All classes are used when ``None``.
        :param download: if ``True``, download and extract if not already present
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self._download()

        base = join(root, self.base_folder, "data")
        if not os.path.isdir(base):
            raise RuntimeError(
                f"FGVC-Aircraft not found at {base}. "
                "Pass download=True to download it automatically."
            )

        # Build variant → 0-indexed class ID mapping.
        # Sort alphabetically to match the np.unique() order used by the SSB reference code
        # when it builds class_to_idx from the annotation file.
        variants_file = join(base, "variants.txt")
        with open(variants_file) as f:
            variants = sorted(line.strip() for line in f if line.strip())
        variant_to_idx = {v: i for i, v in enumerate(variants)}

        annos_file = join(base, f"images_variant_{split}.txt")
        with open(annos_file) as f:
            entries = [line.strip().split(maxsplit=1) for line in f if line.strip()]

        img_dir = join(base, "images")
        self.data: List[Tuple[str, int]] = [
            (join(img_dir, img_id + ".jpg"), variant_to_idx[variant])
            for img_id, variant in entries
        ]

        if classes is not None:
            classes_set = set(classes)
            class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
            self.data = [
                (path, class_to_idx[cls]) for path, cls in self.data if cls in classes_set
            ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.data[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def _download(self) -> None:
        if os.path.isdir(join(self.root, self.base_folder)):
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename)


# ─── Benchmark Base ───────────────────────────────────────────────────────────


class _SSBBase(Benchmark):
    """Shared structure for SSB benchmarks."""

    def train_set(self):
        return self._train

    def test_sets(self, known=True, unknown=True):
        if known and unknown:
            return [self._test_id + self._test_easy, self._test_id + self._test_hard]
        if known and not unknown:
            return [self._test_id, self._test_id]
        if not known and unknown:
            return [self._test_easy, self._test_hard]
        raise ValueError()


# ─── Public Benchmark Classes ─────────────────────────────────────────────────


class CUB_SSB(_SSBBase):
    """
    The benchmark partitions CUB-200-2011 into 100 ID classes and 100 OOD classes.
    OOD classes are split by semantic similarity to the ID classes:

    - **Easy** (32 classes) — far-OOD; most dissimilar to ID classes
    - **Hard** (34 Hard + 34 Medium classes) — near-OOD; most visually similar to ID classes

    ``test_sets()`` returns two combined datasets:
    ``[ID_test + Easy_OOD, ID_test + Hard_OOD]`` with ``ood_names = ["Easy", "Hard"]``.

    :see Paper: `ArXiv <https://arxiv.org/abs/2408.16757>`__
    :see Repository: `Visual-AI/Dissect-OOD-OSR <https://github.com/Visual-AI/Dissect-OOD-OSR>`__
    """

    def __init__(self, root: str, transform: Callable, download: bool = False) -> None:
        """
        :param root: directory containing CUB-200-2011 data and used for caching splits
        :param transform: image transform applied to all samples
        :param download: if ``True``, download CUB-200-2011 if not already present
        """
        splits = load_ssb_splits("cub", root)
        known = splits["known_classes"]
        unk = splits["unknown_classes"]
        hard_and_medium = unk["Hard"] + unk["Medium"]

        kwargs = dict(transform=transform, download=download)
        self._train = _CUB200(root, split="train", classes=known, **kwargs)
        self._test_id = _CUB200(root, split="test", classes=known, **kwargs)
        self._test_easy = _CUB200(
            root,
            split="test",
            classes=unk["Easy"],
            transform=transform,
            target_transform=ToUnknown(),
            download=download,
        )
        self._test_hard = _CUB200(
            root,
            split="test",
            classes=hard_and_medium,
            transform=transform,
            target_transform=ToUnknown(),
            download=download,
        )
        self.ood_names: List[str] = ["Easy", "Hard"]


class StanfordCars_SSB(_SSBBase):
    """
    The benchmark partitions Stanford Cars into 98 ID classes and 98 OOD classes.
    OOD classes are split by semantic similarity to the ID classes:

    - **Easy** (76 classes) — far-OOD; most dissimilar to ID classes
    - **Hard** (7 Hard + 15 Medium classes) — near-OOD; most visually similar to ID classes

    ``test_sets()`` returns two combined datasets:
    ``[ID_test + Easy_OOD, ID_test + Hard_OOD]`` with ``ood_names = ["Easy", "Hard"]``.

    .. note::

        Stanford Cars cannot be downloaded automatically. See :class:`_StanfordCars`
        for manual download instructions.

    :see Paper: `ArXiv <https://arxiv.org/abs/2408.16757>`__
    :see Repository: `Visual-AI/Dissect-OOD-OSR <https://github.com/Visual-AI/Dissect-OOD-OSR>`__
    """

    def __init__(self, root: str, transform: Callable) -> None:
        """
        :param root: directory containing Stanford Cars data and used for caching splits
        :param transform: image transform applied to all samples
        """
        splits = load_ssb_splits("scars", root)
        known = splits["known_classes"]
        unk = splits["unknown_classes"]
        hard_and_medium = unk["Hard"] + unk["Medium"]

        self._train = _StanfordCars(root, split="train", classes=known, transform=transform)
        self._test_id = _StanfordCars(root, split="test", classes=known, transform=transform)
        self._test_easy = _StanfordCars(
            root,
            split="test",
            classes=unk["Easy"],
            transform=transform,
            target_transform=ToUnknown(),
        )
        self._test_hard = _StanfordCars(
            root,
            split="test",
            classes=hard_and_medium,
            transform=transform,
            target_transform=ToUnknown(),
        )
        self.ood_names: List[str] = ["Easy", "Hard"]


class Aircraft_SSB(_SSBBase):
    """
    The benchmark partitions FGVC-Aircraft (variant level, 100 classes) into
    50 ID classes and 50 OOD classes.
    OOD classes are split by semantic similarity to the ID classes:

    - **Easy** (20 classes) — far-OOD; most dissimilar to ID classes
    - **Hard** (13 Hard + 17 Medium classes) — near-OOD; most visually similar to ID classes

    ``test_sets()`` returns two combined datasets:
    ``[ID_test + Easy_OOD, ID_test + Hard_OOD]`` with ``ood_names = ["Easy", "Hard"]``.

    :see Paper: `ArXiv <https://arxiv.org/abs/2408.16757>`__
    :see Repository: `Visual-AI/Dissect-OOD-OSR <https://github.com/Visual-AI/Dissect-OOD-OSR>`__
    """

    def __init__(self, root: str, transform: Callable, download: bool = False) -> None:
        """
        :param root: directory containing FGVC-Aircraft data and used for caching splits
        :param transform: image transform applied to all samples
        :param download: if ``True``, download FGVC-Aircraft if not already present
        """
        splits = load_ssb_splits("aircraft", root)
        known = splits["known_classes"]
        unk = splits["unknown_classes"]
        hard_and_medium = unk["Hard"] + unk["Medium"]

        kwargs = dict(transform=transform, download=download)
        self._train = _FGVCAircraft(root, split="trainval", classes=known, **kwargs)
        self._test_id = _FGVCAircraft(root, split="test", classes=known, **kwargs)
        self._test_easy = _FGVCAircraft(
            root,
            split="test",
            classes=unk["Easy"],
            transform=transform,
            target_transform=ToUnknown(),
            download=download,
        )
        self._test_hard = _FGVCAircraft(
            root,
            split="test",
            classes=hard_and_medium,
            transform=transform,
            target_transform=ToUnknown(),
            download=download,
        )
        self.ood_names: List[str] = ["Easy", "Hard"]
