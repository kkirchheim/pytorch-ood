import json

import numpy as np
import pytest
import torch
from PIL import Image

from pytorch_ood.augment.img import COCO, InsertCOCO


def _insert_coco(tmp_path, **kwargs) -> InsertCOCO:
    """InsertCOCO on an empty, synthetic COCO directory (no images, no annotations)."""
    (tmp_path / "annotations").mkdir()
    (tmp_path / "train2017").mkdir()
    with open(tmp_path / "annotations" / "instances_train2017.json", "w") as f:
        json.dump({"images": [], "annotations": [], "categories": []}, f)
    return InsertCOCO(coco_dir=str(tmp_path), p=1.0, **kwargs)


def _rgba_object(opaque: np.ndarray, color) -> Image.Image:
    """RGBA object of the given colour; pixels where ``opaque`` is False are fully transparent."""
    arr = np.zeros(opaque.shape + (4,), dtype=np.uint8)
    arr[opaque] = list(color) + [255]
    return Image.fromarray(arr, mode="RGBA")


def test_pasted_pixels_are_ood_and_other_labels_unchanged(tmp_path, monkeypatch):
    """With two overlapping objects, the OOD pixels of the target are exactly the union of the
    non-transparent object pixels, and every other label (also >= 128 and the void label 255)
    is returned unchanged."""
    h, w = 10, 12
    labels = np.array([0, 1, 127, 128, 200, 254, 255])
    target = torch.tensor(labels[np.arange(h * w) % len(labels)].reshape(h, w))

    opaque_a = np.ones((4, 5), dtype=bool)
    opaque_a[1, 2] = False  # a hole in the first object
    opaque_b = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool)
    objects = iter(
        [
            (_rgba_object(opaque_a, (255, 0, 0)), 2, 1),
            (_rgba_object(opaque_b, (0, 255, 0)), 4, 3),  # overlaps the first object
        ]
    )
    insert = _insert_coco(tmp_path, n=2, ood_mask_value=-1)
    monkeypatch.setattr(insert, "_random_pos_and_scale", lambda orig_img_dim: next(objects))

    img, out = insert(Image.new("RGB", (w, h), (0, 0, 255)), target.clone())

    pasted = np.zeros((h, w), dtype=bool)
    pasted[1:5, 2:7] |= opaque_a
    pasted[3:6, 4:7] |= opaque_b
    expected = torch.where(torch.from_numpy(pasted), -1, target).long()
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, expected)
    # the image was changed in exactly the same pixels
    changed = np.any(np.asarray(img) != np.array([0, 0, 255]), axis=-1)
    np.testing.assert_array_equal(changed, pasted)


def test_target_format_without_insertion_and_for_large_objects(tmp_path, monkeypatch):
    """The target is always an int64 tensor of the input shape, also when nothing is inserted,
    and objects larger than the image are cropped instead of raising."""
    target = torch.full((20, 20), 255, dtype=torch.uint8)
    insert = _insert_coco(tmp_path, ood_mask_value=-1)

    insert.ood_rate = 0.0
    _, out = insert(Image.new("RGB", (20, 20)), target)
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, target.long())

    insert.ood_rate = 1.0
    big = np.full((100, 100, 4), 255, dtype=np.uint8)  # rescaled to at least 28 x 28 pixels
    monkeypatch.setattr(insert, "_load_coco_annotation_dynamic", lambda: big)
    _, out = insert(Image.new("RGB", (20, 20)), target)
    assert out.shape == (20, 20) and out.dtype == torch.int64
    assert out[10, 10] == -1  # inside the inscribed circle of the object for any rotation
    assert set(out.unique().tolist()) <= {-1, 255}


@pytest.mark.parametrize(
    "counts,size,expected",
    [
        # COCO RLE: runs alternate 0s and 1s, starting with 0s, in column-major order
        ([1, 2, 2, 1], [2, 3], [[0, 1, 0], [1, 0, 1]]),
        ([0, 3, 3], [3, 2], [[1, 0], [1, 0], [1, 0]]),
        ([6], [2, 3], [[0, 0, 0], [0, 0, 0]]),
    ],
)
def test_rle_crowd_annotation_to_mask(counts, size, expected):
    """Uncompressed RLE (used by COCO crowd annotations) decodes to the hand-computed 0/1 mask."""
    ann = {"segmentation": {"counts": counts, "size": size}}
    mask = COCO().annToMask(ann, tuple(size))
    np.testing.assert_array_equal(mask, np.array(expected, dtype=np.uint8))
