from collections import defaultdict
import numpy as np

# from ._dev.mask_utils import decode, merge, area
from .mask_gen import read_coco_annotations, generate_masks, create_mask_from_segmentation


def _isArrayLike(obj):
    """
    Check if an object is array-like (list, tuple, or other iterable).

    :param obj: The object to check.
    :return: True if the object is array-like, False otherwise.
    """
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes))


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

    def MYannToMask(self,ann, img_size):
        # print(ann)
        return create_mask_from_segmentation(ann["segmentation"], (img_size[1],img_size[0]))
    # def annToMask(self, ann):
    #     """
    #     Convert an annotation into a binary mask.

    #     :param ann: Annotation dictionary containing segmentation data. (dict)
    #     :return: Binary mask corresponding to the annotation. (np.ndarray)
    #     """
    #     rle = self.annToRLE(ann)
    #     return decode(rle)

    # def annToRLE(self, ann):
    #     """
    #     Convert an annotation's segmentation into RLE (Run-Length Encoding).

    #     :param ann: Annotation dictionary containing segmentation data. (dict)
    #     :return: RLE representation of the segmentation. (dict)
    #     """
    #     segm = ann["segmentation"]
    #     if isinstance(segm, list):
    #         # Polygon segmentation
    #         rle = merge([self.frPyObjects(p, ann["image_id"], ann["category_id"]) for p in segm])
    #     elif isinstance(segm["counts"], list):
    #         # RLE segmentation
    #         rle = self.frPyObjects(segm, ann["image_id"], ann["category_id"])
    #     else:
    #         # Already in RLE format
    #         rle = segm
    #     return rle

    def getMaskFromId(self, image_id):
        coco_data = read_coco_annotations(self.annotation_file)
        masks = generate_masks(coco_data, image_id)
        return masks

    def frPyObjects(self, pyobj, h, w):
        """
        Convert polygon segmentation data into RLE format.

        :param pyobj: Polygon points. (list)
        :param h: Height of the image. (int)
        :param w: Width of the image. (int)
        :return: RLE representation of the polygon. (dict)
        """
        if isinstance(pyobj, list):
            # Convert polygon to binary mask
            mask = np.zeros((h, w), dtype=np.uint8)
            return mask  # Replace with actual RLE logic
        return pyobj
