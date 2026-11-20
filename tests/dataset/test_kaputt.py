import sys
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from pytorch_ood.dataset.img import Kaputt

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _create_split(root: Path, split: str, rows):
    for folder in ("image", "crop", "mask"):
        (root / "data" / split / "query-data" / folder).mkdir(parents=True)

    records = []
    for capture_id, material, defect, major_defect, defect_types in rows:
        paths = {
            "query_image": f"data/{split}/query-data/image/{capture_id}.jpg",
            "query_crop": f"data/{split}/query-data/crop/{capture_id}.jpg",
            "query_mask": f"data/{split}/query-data/mask/{capture_id}.png",
        }
        Image.new("RGB", (32, 32)).save(root / paths["query_image"])
        Image.new("RGB", (16, 16)).save(root / paths["query_crop"])
        Image.new("L", (32, 32)).save(root / paths["query_mask"])
        records.append(
            {
                "capture_id": capture_id,
                "item_material": material,
                "item_identifier": f"item-{capture_id}",
                "defect": defect,
                "major_defect": major_defect,
                "defect_types": defect_types,
                **paths,
            }
        )

    pq.write_table(pa.Table.from_pylist(records), root / f"query-{split}.parquet")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _create_split(
        tmp_path,
        "test",
        [
            ("a", "cardboard", False, False, ""),
            ("b", "plastic_hard", True, False, "deformation"),
            ("c", "paper", True, True, "penetration,spillage"),
        ],
    )
    return tmp_path


def test_returns_all_data(root: Path):
    dataset = Kaputt(str(root), split="test")
    assert len(dataset) == 3

    sample = dataset[2]
    assert set(sample) == {
        "image",
        "crop",
        "mask",
        "capture_id",
        "item_identifier",
        "item_material",
        "defect",
        "major_defect",
        "defect_types",
    }
    assert sample["image"].size == (32, 32)
    assert sample["crop"].size == (16, 16)
    assert sample["mask"].mode == "L"
    assert sample["capture_id"] == "c"
    assert sample["item_identifier"] == "item-c"
    assert sample["item_material"] == "paper"
    assert sample["defect"] is True
    assert sample["major_defect"] is True
    assert sample["defect_types"] == "penetration,spillage"


def test_transform_receives_sample(root: Path):
    dataset = Kaputt(str(root), split="test", transform=lambda s: (s["capture_id"], s["defect"]))
    assert dataset[1] == ("b", True)


def test_unused_images_are_not_decoded(root: Path, monkeypatch):
    loaded = []
    monkeypatch.setattr(ImageFile.ImageFile, "load", lambda self: loaded.append(self))
    dataset = Kaputt(str(root), split="test", transform=lambda s: s["defect"])

    assert dataset[0] is False
    assert loaded == []


def test_dataloader_batches_transformed_samples(root: Path):
    def transform(sample):
        return {
            "crop": to_tensor(sample["crop"]),
            "defect": sample["defect"],
            "defect_types": sample["defect_types"],
        }

    dataset = Kaputt(str(root), split="test", transform=transform)
    batch = next(iter(DataLoader(dataset, batch_size=3)))

    assert batch["crop"].shape == (3, 3, 16, 16)
    assert torch.equal(batch["defect"], torch.tensor([False, True, True]))
    assert batch["defect_types"] == ["", "deformation", "penetration,spillage"]


def test_invalid_split(root: Path):
    with pytest.raises(ValueError, match="Invalid split"):
        Kaputt(str(root), split="val")


def test_missing_metadata_links_to_download(root: Path):
    with pytest.raises(RuntimeError, match="huggingface.co/datasets/amazon/kaputt"):
        Kaputt(str(root), split="train")


def test_missing_images_link_to_download(root: Path):
    mask_dir = root / "data" / "test" / "query-data" / "mask"
    for f in mask_dir.iterdir():
        f.unlink()
    mask_dir.rmdir()

    with pytest.raises(RuntimeError, match="huggingface.co/datasets/amazon/kaputt"):
        Kaputt(str(root), split="test")


def test_missing_pyarrow(root: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)

    with pytest.raises(RuntimeError, match="pyarrow"):
        Kaputt(str(root), split="test")
