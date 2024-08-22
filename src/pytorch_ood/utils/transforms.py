"""

..  autoclass:: pytorch_ood.utils.ToUnknown
    :members:

..  autoclass:: pytorch_ood.utils.ToRGB
    :members:

..  autoclass:: pytorch_ood.utils.TargetMapping
    :members:

..  autoclass:: pytorch_ood.utils.InsertCOCO
    :members:

..  autoclass:: pytorch_ood.utils.COCO
    :members:

"""
from typing import Set

import os
import random
from os.path import join

import numpy as np
from PIL import Image, ImageDraw
import torch
from collections import defaultdict
import json
from torchvision.datasets.utils import download_and_extract_archive


class ToUnknown(object):
    """
    Callable that returns a negative number, used in pipelines to mark specific datasets as OOD or unknown.
    """

    def __init__(self):
        pass

    def __call__(self, y):
        return -1


class ToRGB(object):
    """
    Convert Image to RGB, if it is not already.
    """

    def __call__(self, x):
        try:
            return x.convert("RGB")
        except Exception as e:
            return x


class TargetMapping(object):
    """
    Maps known classes to index in :math:`[0,n]`, unknown classes to values in :math:`[-\\infty, -1]`.
    Required for open set simulations.

    **Example:**
    If we split up a dataset so that the classes 2,3,4,9 are considered *known* or *IN*, these class
    labels have to be remapped to 0,1,2,3 to be able to train
    using cross entropy with 1-of-K-vectors. All other classes have to be mapped to values :math:`<0`
    to be marked as OOD.

    Target mappings have to be known at evaluation time.
    """

    def __init__(self, known: Set, unknown: Set):
        self._map = dict()
        self._map.update({clazz: index for index, clazz in enumerate(set(known))})
        # mapping train_out classes to < 0
        self._map.update({clazz: (-clazz) for index, clazz in enumerate(set(unknown))})

    def __call__(self, target):
        if isinstance(target, torch.Tensor):
            return self._map.get(target.item(), -1)

        return self._map.get(target, -1)

    def __getitem__(self, item):
        if isinstance(item, torch.Tensor):
            return self._map[item.item()]

        return self._map[item]

    def items(self):
        return self._map.items()

    def __repr__(self):
        return str(self._map)


class InsertCOCO:
    """
    Transform to insert coco image snippets as OOD objects into the image.

    :param coco_dir: Directory to store the coco dataset
    :param prohibet_classes: List of classes that should not be used for the OOD generation
    :param probability_of_ood: Probability of adding an OOD object to the image
    :param ood_per_image: Number of OOD objects per image
    :param annotation_per_coco_image: Number of different annotation that are used for the ood object per coco image. (e.g. that are two elefants on one coco image, if this parameter is one only one one lephant ist used)
    :param ood_mask_value: Value of the OOD mask
    :param upscale: Upscale factor for the OOD object
    :param year: Year of the coco dataset
    :param min_size_of_img: Minimum size of the used coco image
    """

    def __init__(
        self,
        coco_dir: str,
        prohibet_classes: list[str],
        probability_of_ood: float = 0.1,
        ood_per_image: int = 1,
        annotation_per_coco_image: int = 1,
        ood_mask_value: int = -1,
        upscale: float = 1.4150357439499515,
        year: int = 2017,
        min_size_of_img: int = 480,
    ):
        self.coco_dir = coco_dir
        # check if coco_dir exists
        if not os.path.exists(self.coco_dir):
            os.makedirs(self.coco_dir)
        if type(prohibet_classes) is not list:
            if prohibet_classes == "bddAnomaly":
                self.prohibet_classes = ["train", "bicycle", "motorcycle"]
            elif prohibet_classes == "Streethazards":
                self.prohibet_classes = [
                    "traffic light",
                    "stop sign",
                    "vase",
                    "refrigerator",
                    "sink",
                    "toaster",
                    "oven",
                    "dining table",
                    "chair",
                    "tennis racket",
                ]
        else:
            self.prohibet_classes = prohibet_classes
        self.year = year
        self.upscale = upscale
        self.ood_rate = probability_of_ood
        self.ood_mask_value = ood_mask_value
        self.ood_per_image = ood_per_image
        self.annotation_per_coco_image = annotation_per_coco_image
        self.in_class_label = 0
        self.out_class_label = 254
        self.min_size_of_img = min_size_of_img
        # download 2017 trainset
        self.img_url = "http://images.cocodataset.org/zips/train2017.zip"
        self.images_dir = join(self.coco_dir, f"train{str(self.year)}")

        # http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        self.annottations_url = (
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        )
        self.annotation_dir = join(
            self.coco_dir, f"annotations/instances_train{str(self.year)}.json"
        )

        self.download()
        self.tools = COCO(join(self.coco_dir, f"annotations/instances_train{str(self.year)}.json"))

        self.usable_image_ids = self._init_ids()

    # inspired from https://github.com/tla93/InpaintingOutlierSynthesis/blob/main/src/train_coco.py
    def __call__(self, img, segm) -> tuple:
        """
        check if OOD should be added and add it with the given probability
        :param img: image
        :param segm: segmentation
        :return: img, segm
        """
        if random.random() <= self.ood_rate:
            segm = Image.fromarray(np.array(segm, dtype=np.uint8))
            img, segm = self.add_ood(img, segm)
            segm = torch.tensor(segm, dtype=torch.int64)

        return img, segm

    def add_ood(self, img, segm) -> tuple:
        """
        Add OOD objects to the image and manipulate the segmentation in the same way

        :param img: image
        :param segm: segmentation
        :return: img, segm
        """
        for elem in range(self.ood_per_image):
            # insert one OOD object
            w, h = segm.size
            rotated_ood_image, x_pos, y_pos = self._random_pos_and_scale(orig_img_dim=[h, w])

            # insert the clip image into the original one
            img.paste(rotated_ood_image, (x_pos, y_pos), rotated_ood_image)
            rotated_ood_image_arr = np.asarray(rotated_ood_image)
            segm_arr = np.asarray(segm, dtype=np.int8)

            for i in range(rotated_ood_image_arr.shape[0]):
                for j in range(rotated_ood_image_arr.shape[1]):
                    # if != png pixel is not empty
                    if not np.array_equal(
                        rotated_ood_image_arr[i, j],
                        np.zeros(rotated_ood_image_arr.shape[2]),
                    ):
                        segm_arr[i + y_pos, j + x_pos] = self.ood_mask_value

            segm = Image.fromarray(segm_arr)

        return img, segm_arr

    def _random_pos_and_scale(self, orig_img_dim) -> tuple:
        """
        Load random coco image and scale it to a random size and rotate it by a random angle
        :param orig_img_dim: original image dimensions
        :return: rotated_ood_image, x_pixel, y_pixel
        """
        clip_image = Image.fromarray(self.load_coco_annotation_dynamic())

        # we rescale since COCO images can be of different size
        scale_range = [int(20 * self.upscale), int(50 * self.upscale)]
        rotation = random.randint(0, 359)

        scale = random.randint(scale_range[0], scale_range[1]) / 100
        # scale the clip image by the desired amount
        new_width = int(clip_image.size[0] * scale)
        new_height = int(clip_image.size[1] * scale)
        # scale the clip image by the desired amount
        resized_image = clip_image.resize((new_width, new_height))
        # rotate the clip image by the desired amount
        rotated_ood_image = resized_image.rotate(rotation)

        # 10 pixel away from the edge
        pos_range_x = [10, orig_img_dim[1] - new_width - 10]
        pos_range_y = [10, orig_img_dim[0] - new_height - 10]

        x_pixel = random.randint(pos_range_x[0], pos_range_x[1])
        y_pixel = random.randint(pos_range_y[0], pos_range_y[1])
        # random flip
        if np.random.choice([0, 1]):
            rotated_ood_image = rotated_ood_image.transpose(Image.FLIP_LEFT_RIGHT)

        return rotated_ood_image, x_pixel, y_pixel

    # Parts of this function is inspired from https://github.com/robin-chan/meta-ood/blob/master/preparation/prepare_coco_segmentation.py
    def load_coco_annotation_dynamic(self) -> np.ndarray:
        """
        Load a random coco image and return the snipped of the coco image with the ood object

        :return: snipped of the ood object
        """

        img_id = self.usable_image_ids[np.random.randint(0, len(self.usable_image_ids))]
        img = self.tools.loadImgs(int(img_id))[0]
        # load annotations from annotation id (based on image id)
        annotations = self.tools.loadAnns(self.tools.getAnnIds(imgIds=img["id"]))
        mask = np.ones((img["height"], img["width"]), dtype="uint8") * self.in_class_label

        # get masks
        for j in range(min(len(annotations), self.annotation_per_coco_image)):
            mask = np.maximum(
                self.tools.annToMask(annotations[j], (img["height"], img["width"]))
                * self.out_class_label,
                mask,
            )

        # write mask
        for j in range(min(len(annotations), self.annotation_per_coco_image)):
            mask[
                self.tools.annToMask(annotations[j], (img["height"], img["width"])) == 1
            ] = self.out_class_label

        annott_segm_arr = np.array(mask)

        # load coco image
        path = join(self.images_dir, "{:012d}.png".format(int(img_id)).replace("png", "jpg"))
        img = Image.open(path)

        annott_img_arr = np.array(img.convert("RGBA"))

        # elim all not segmentated Pixels
        for i in range(annott_segm_arr.shape[0]):
            for j in range(annott_segm_arr.shape[1]):
                if annott_segm_arr[i, j] == 0:
                    annott_img_arr[i, j] = [0, 0, 0, 0]

        return annott_img_arr

    # Parts of this function is inspired from https://github.com/robin-chan/meta-ood/blob/master/preparation/prepare_coco_segmentation.py
    def _init_ids(self) -> list:
        """
        Determines all available ids of coco images that do not contain any of the prohibet classes
        :return: list of usable image ids
        """
        prohibet_image_ids = []
        # Iterate overall overlap categories to find all prohibed image ids
        for id in self.tools.getCatIds(catNms=self.prohibet_classes):
            prohibet_image_ids.append(self.tools.getImgIds(catIds=id))
        # Eliminate duplications
        prohibet_image_ids = [item for sublist in prohibet_image_ids for item in sublist]
        prohibet_image_ids = set(prohibet_image_ids)

        # find all usable images
        usable_image_ids = []
        for image in os.listdir(self.images_dir):
            img_id = image[:-4]
            if int(img_id) not in prohibet_image_ids:
                img = self.tools.loadImgs(int(img_id))[0]
                # check size of the image
                if img["height"] >= self.min_size_of_img and img["width"] >= self.min_size_of_img:
                    # append image id
                    usable_image_ids.append(img_id)
        return usable_image_ids

    def download(self) -> None:
        """
        Donwload the coco dataset if not already downloaded
        """
        # check if train images exist
        if not os.path.exists(self.images_dir):
            download_and_extract_archive(
                self.img_url, self.coco_dir, filename=f"train{str(self.year)}.zip"
            )
        # check if annotation file exists
        if not os.path.exists(self.annotation_dir):
            download_and_extract_archive(
                self.annottations_url,
                self.coco_dir,
                filename=f"annotations_trainval{str(self.year)}.zip",
            )


def _isArrayLike(obj):
    """
    Check if an object is array-like (list, tuple, or other iterable).

    :param obj: The object to check.
    :return: True if the object is array-like, False otherwise.
    """
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes))


def read_coco_annotations(annotations_file):
    """
    Load COCO annotations from a JSON file.

    :param annotations_file: Path to the COCO annotations JSON file.
    :return: Parsed COCO data. (dict)
    """
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)
    return coco_data


def rle_decode(rle, img_shape):
    """
    Decode RLE encoded mask into a binary mask.

    :param rle: Dictionary with 'counts' and 'size' for RLE encoding.
    :param img_shape: Tuple (height, width) of the image size.
    :return: Binary mask as a numpy array. (np.ndarray)
    """
    height, width = img_shape
    mask = np.zeros(height * width, dtype=np.uint8)

    counts = rle["counts"]
    size = rle["size"]

    if not size or size[0] != height or size[1] != width:
        print(
            f"Warning: RLE size {size} does not match the provided image dimensions {img_shape}."
        )
        raise ValueError("RLE size does not match the provided image dimensions.")

    # Convert counts to a flat mask array
    rle_array = np.array(counts, dtype=np.uint32)
    positions = np.concatenate(
        [
            np.arange(start, start + length)
            for start, length in zip(np.cumsum(rle_array[:-1]), rle_array[1:])
        ]
    )
    # Ensure positions are integers
    positions = positions.astype(int)
    # print(positions)
    mask[positions] = 1
    mask = mask.reshape((height, width))

    return mask


def create_mask_from_segmentation(segmentation, image_size):
    """
    Create a binary mask from segmentation data, which can be either polygons or RLE.

    :param segmentation: List of polygons or RLE data representing the segmentation. (list of lists or dict)
    :param image_size: Size of the image as (width, height). (tuple of int)
    :return: Binary mask as a numpy array. (np.ndarray)
    """
    if isinstance(segmentation, dict):
        # Handle RLE segmentation
        if "counts" in segmentation and "size" in segmentation:
            img_shape = (image_size[1], image_size[0])  # (height, width)
            mask = rle_decode(segmentation, img_shape)
            mask = Image.fromarray(mask * 255)  # Convert binary mask to Image format
        else:
            raise ValueError("Unexpected RLE format.")
    else:
        # Handle polygon segmentation
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)

        for polygon in segmentation:
            # Ensure polygon coordinates are in the correct format
            if isinstance(polygon, list) and all(isinstance(p, (int, float)) for p in polygon):
                if len(polygon) % 2 != 0:
                    raise ValueError(
                        "Polygon coordinates list should have an even number of elements."
                    )
                polygon = np.array(polygon).reshape(-1, 2).astype(int)
                polygon = [tuple(p) for p in polygon]
                draw.polygon(polygon, outline=1, fill=1)
            else:
                raise ValueError("Unexpected format for polygon coordinates.")

        mask = np.array(mask)

    return np.array(mask)


def generate_masks(coco_data, image_id):
    """
    Generate masks for a specific image ID from COCO annotations.

    :param coco_data: Parsed COCO data. (dict)
    :param image_id: ID of the image to generate masks for. (int)
    :return: List of binary masks as numpy arrays. (list of np.ndarray)
    """
    # print(f"Image ID: {image_id}")
    masks = []
    annotations = coco_data["annotations"]
    image_info = next(item for item in coco_data["images"] if item["id"] == image_id)
    image_size = (image_info["width"], image_info["height"])

    for annotation in annotations:
        if annotation["image_id"] == image_id:
            segmentation = annotation["segmentation"]
            mask = create_mask_from_segmentation(segmentation, image_size)
            masks.append(mask)

    return masks


class COCO:
    """
    A simplified version of the COCO (Common Objects in Context) class,
    used for handling COCO dataset annotations without relying on the pycocotools library.
    """

    def __init__(self, annotation_file=None):
        """
        Initialize the COCO object, optionally loading annotations from a file.

        :param annotation_file: Path to the COCO annotation JSON file.
        """
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.annotation_file = annotation_file
        if annotation_file is not None:
            # Load annotations from the provided file
            import json

            with open(annotation_file, "r") as f:
                self.dataset = json.load(f)
            self.createIndex()

    def createIndex(self):
        """
        Create indices for quick lookup of annotations, images, and categories.
        """
        for ann in self.dataset.get("annotations", []):
            self.anns[ann["id"]] = ann
            self.imgToAnns[ann["image_id"]].append(ann)
        for img in self.dataset.get("images", []):
            self.imgs[img["id"]] = img
        for cat in self.dataset.get("categories", []):
            self.cats[cat["id"]] = cat
        for ann in self.dataset.get("annotations", []):
            self.catToImgs[ann["category_id"]].append(ann["image_id"])

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        """
        Get annotation IDs that satisfy given filter conditions.

        :param imgIds: List of image IDs or a single image ID to filter. (int or list of int)
        :param catIds: List of category IDs or a single category ID to filter. (int or list of int)
        :param areaRng: Range of area sizes to filter annotations. (list of int)
        :return: List of annotation IDs that match the conditions. (list of int)
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        annIds = []
        for imgId in imgIds:
            anns = self.imgToAnns[imgId]
            for ann in anns:
                if (not catIds or ann["category_id"] in catIds) and (
                    not areaRng or areaRng[0] <= ann["area"] <= areaRng[1]
                ):
                    annIds.append(ann["id"])
        return annIds

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        Get category IDs that satisfy given filter conditions.

        :param catNms: List of category names to filter. (list of str)
        :param supNms: List of supercategory names to filter. (list of str)
        :param catIds: List of category IDs or a single category ID to filter. (int or list of int)
        :return: List of category IDs that match the conditions. (list of int)
        """
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        catIdsList = []
        for catId, cat in self.cats.items():
            if (
                (not catNms or cat["name"] in catNms)
                and (not supNms or cat["supercategory"] in supNms)
                and (not catIds or catId in catIds)
            ):
                catIdsList.append(catId)
        return catIdsList

    def getImgIds(self, imgIds=[], catIds=[]):
        """
        Get image IDs that satisfy the given filter conditions.

        :param imgIds: List of image IDs or a single image ID to filter. (int or list of int)
        :param catIds: List of category IDs or a single category ID to filter. (int or list of int)
        :return: List of image IDs that match the conditions. (list of int)
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        imgIdsList = set(imgIds) if imgIds else set(self.imgs.keys())

        if catIds:
            catIdsList = set()
            for catId in catIds:
                catIdsList.update(self.catToImgs[catId])
            imgIdsList &= catIdsList

        return list(imgIdsList)

    def loadAnns(self, ids=[]):
        """
        Load annotations with the specified IDs.

        :param ids: List of annotation IDs or a single annotation ID to load. (int or list of int)
        :return: List of annotation dictionaries corresponding to the IDs. (list of dict)
        """
        ids = ids if _isArrayLike(ids) else [ids]

        return [self.anns[id] for id in ids]

    def loadImgs(self, ids=[]):
        """
        Load images with the specified IDs.

        :param ids: List of image IDs or a single image ID to load. (int or list of int)
        :return: List of image dictionaries corresponding to the IDs. (list of dict)
        """
        ids = ids if _isArrayLike(ids) else [ids]

        return [self.imgs[id] for id in ids]

    def annToMask(self, ann, img_size):
        """
        Convert annotation data to a binary mask.

        :param ann: Annotation data. (dict)
        :param img_size: Size of the image as (width, height). (tuple of int)
        :return: Binary mask as a numpy array. (np.ndarray)
        """
        return create_mask_from_segmentation(ann["segmentation"], (img_size[1], img_size[0]))
