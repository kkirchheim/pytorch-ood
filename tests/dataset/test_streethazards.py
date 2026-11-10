from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pytorch_ood.dataset.img import StreetHazards


@pytest.mark.parametrize("root", ["data", "my_images"])
def test_mask_is_found_when_the_root_contains_images(root: str, tmp_path: Path, monkeypatch):
    for folder, image in [
        ("images", Image.new("RGB", (2, 1))),
        ("annotations", Image.fromarray(np.array([[1, 14]], dtype=np.uint8))),
    ]:
        path = tmp_path / root / "streethazards" / "test" / folder / "t5"
        path.mkdir(parents=True)
        image.save(path / "0.png")
    monkeypatch.setattr(StreetHazards, "_check_integrity", lambda self: True)

    _, target = StreetHazards(str(tmp_path / root), subset="test")[0]

    assert torch.equal(target, torch.tensor([[0, -1]]))
