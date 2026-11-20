import logging
from os.path import exists, isdir, join
from typing import Any, Callable, Dict, Optional

from PIL import Image
from torchvision.datasets import VisionDataset

from ...api import DatasetInfo, Paper, Role, Task

log = logging.getLogger(__name__)


class Kaputt(VisionDataset):
    """
    Kaputt is a large-scale dataset for visual defect detection in retail logistics, with more than 230,000
    images of more than 48,000 distinct items, of which more than 29,000 show a defect. This class loads the
    roughly 100,000 annotated query images. The items in the splits do not overlap. The unannotated reference
    images are not loaded.

    Each sample is a dictionary that contains everything that is available for a query image:

    ==================== ======================================================================================
    Key                  Value
    ==================== ======================================================================================
    ``image``            full image of the item in its tray, 2048 x 2048 (:class:`PIL.Image.Image`)
    ``crop``             crop of the item, usually square, with a size that depends on the item
                         (:class:`PIL.Image.Image`)
    ``mask``             item segmentation mask for ``image`` (:class:`PIL.Image.Image`)
    ``capture_id``       ID of the image (``str``)
    ``item_identifier``  ID of the item (``str``)
    ``item_material``    one of :attr:`materials` (``str``)
    ``defect``           whether the item has a minor or major defect (``bool``)
    ``major_defect``     whether the item has a major defect (``bool``)
    ``defect_types``     comma-separated values from :attr:`defect_types`, empty if there is no defect (``str``)
    ==================== ======================================================================================

    Images have to be converted to tensors of the same size before they can be batched, so a
    :class:`~torch.utils.data.DataLoader` needs a ``transform``. It receives the whole dictionary and can
    return what is needed. Images that it does not use are not decoded. For example, to get the crop and
    ``-1`` as the label of OOD samples for defective items:

    .. code-block:: python

        preprocess = ToTensor()

        def transform(sample):
            return preprocess(sample["crop"].resize((224, 224))), -1 if sample["defect"] else 0

        dataset = Kaputt(root="data/kaputt", split="test", transform=transform)

    The data is gated and cannot be downloaded automatically. To get it:

    #. Accept the terms of use on the `Hugging Face page <https://huggingface.co/datasets/amazon/kaputt>`__.
    #. Download ``datasets.tar.gz``, ``query-image.tar.gz``, ``query-crop.tar.gz`` and ``query-mask.tar.gz``
       from the ``kaputt-release`` folder (about 140 GB).
    #. Extract all of them into ``root``, which then contains ``query-<split>.parquet`` and ``data/<split>/``.

    Reading the parquet files requires ``pyarrow``.

    :see Website: `Kaputt <https://www.kaputt-dataset.com>`__
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.BENCHMARK},
        license="CC-BY-NC-ND-4.0 with additional terms of use",
        paper=Paper(
            title="Kaputt: A Large-Scale Dataset for Visual Defect Detection",
            venue="ICCV",
            year=2025,
            url="https://arxiv.org/abs/2510.05903",
        ),
        homepage="https://www.kaputt-dataset.com",
        ai_coded=True,
    )

    #: available splits
    splits = ["train", "validation", "test"]
    #: possible values of ``item_material``
    materials = [
        "cardboard",
        "plastic_loose_bag",
        "plastic_hard",
        "plastic_tight_wrap",
        "plastic_bubble_wrap",
        "book_paper",
        "book_other",
        "book_plastic_tight_wrap",
        "paper",
        "other",
    ]
    #: possible values in ``defect_types``
    defect_types = [
        "actuation",
        "deconstruction",
        "deformation",
        "missing_unit",
        "penetration",
        "spillage",
        "superficial",
    ]

    def __init__(self, root: str, split: str, transform: Optional[Callable] = None) -> None:
        """
        :param root: directory that contains the extracted data
        :param split: one of ``train``, ``validation`` or ``test``
        :param transform: function applied to the sample dictionary
        :raises ValueError: if ``split`` is invalid
        :raises RuntimeError: if the data is not found in ``root`` or ``pyarrow`` is not installed
        """
        super().__init__(root, transform=transform)

        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}, possible values: {self.splits}")
        self.split = split

        metadata = join(self.root, f"query-{split}.parquet")
        if not self._check_integrity(metadata):
            raise RuntimeError(
                f"Dataset not found in {self.root}. The data has to be downloaded manually from "
                "https://huggingface.co/datasets/amazon/kaputt, see the documentation of this class."
            )

        try:
            from pyarrow.parquet import read_table
        except ImportError:
            raise RuntimeError("You have to install 'pyarrow' to use this dataset")

        self.samples = read_table(metadata).to_pylist()
        log.debug(f"Loaded {len(self.samples)} samples")

    def _check_integrity(self, metadata: str) -> bool:
        data = join(self.root, "data", self.split, "query-data")
        return exists(metadata) and all(isdir(join(data, d)) for d in ("image", "crop", "mask"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        """
        :param index: index
        :return: the sample dictionary, or the output of ``transform`` if one was given
        """
        row = self.samples[index]

        sample: Dict[str, Any] = {
            "image": Image.open(join(self.root, row["query_image"])),
            "crop": Image.open(join(self.root, row["query_crop"])),
            "mask": Image.open(join(self.root, row["query_mask"])),
            "capture_id": row["capture_id"],
            "item_identifier": row["item_identifier"],
            "item_material": row["item_material"],
            "defect": row["defect"],
            "major_defect": row["major_defect"],
            "defect_types": row["defect_types"],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
