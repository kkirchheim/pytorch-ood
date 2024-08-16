import os
import random
from os.path import join

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class InsertCOCO:
    def __init__(
        self,
        coco_dir,
        dataset,
        ood_rate=0.1,
        ood_per_image=1,
        ood_mask_value=-1,
        upscale=1.4150357439499515,
        year=2017,
        in_class_label=0,
        out_class_label=254,
        min_size_of_img=480,
    ):
        self.coco_dir = coco_dir
        # check if coco_dir exists
        if not os.path.exists(self.coco_dir):
            os.makedirs(self.coco_dir)
        # check if dataset is known
        # TODO: oder klassen als array bekommen
        if dataset not in ["bddAnomaly", "Streethazards"]:
            raise ValueError(
                "Unknown dataset! Please choose between 'bddAnomaly' and 'Streethazards'!"
            )
        self.dataset = dataset
        self.year = year
        self.upscale = upscale
        self.ood_rate = ood_rate
        self.ood_mask_value = ood_mask_value
        self.ood_per_image = ood_per_image
        self.in_class_label = in_class_label
        self.out_class_label = out_class_label
        self.min_size_of_img = min_size_of_img
        # download 2017 trainset
        # link http://images.cocodataset.org/zips/train2017.zip
        self.img_url = "http://images.cocodataset.org/zips/train2017.zip"
        self.images_dir = join(self.coco_dir, f"train{str(self.year)}")

        # http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        self.annottations_url = (
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        )
        self.annotation_dir = join(
            self.coco_dir, f"/annotations/instances_train{str(self.year)}.json"
        )

        # if data not prepared
        # if not self.check_dataset():
        self.download_prepare_data()
        self.files = os.listdir(join(self.coco_dir, f"train{str(self.year)}"))
        self.annott = join(
            self.coco_dir, f"/annotations/for_{self.dataset}_seg_train{str(self.year)}"
        )

    # inspired from https://github.com/tla93/InpaintingOutlierSynthesis/blob/main/src/train_coco.py
    def __call__(self, img, segm):
        if random.random() <= self.ood_rate:
            segm = Image.fromarray(np.array(segm, dtype=np.uint8))
            img, segm = self.add_ood(img, segm)
            segm = torch.tensor(segm, dtype=torch.int64)

        return img, segm

    def add_ood(self, img, segm):
        """ """
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

    def _random_pos_and_scale(self, orig_img_dim):
        """ """
        clip_image = Image.fromarray(self.load_coco_ood())

        # scale_range=[20,50]
        # we rescale since COCO images can be of different size
        # upscale=1.4150357439499515

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

        # 10 pixel vom rand weg
        pos_range_x = [10, orig_img_dim[1] - new_width - 10]
        pos_range_y = [10, orig_img_dim[0] - new_height - 10]

        x_pixel = random.randint(pos_range_x[0], pos_range_x[1])
        y_pixel = random.randint(pos_range_y[0], pos_range_y[1])
        # new: random flip
        if np.random.choice([0, 1]):
            rotated_ood_image = rotated_ood_image.transpose(Image.FLIP_LEFT_RIGHT)

        return rotated_ood_image, x_pixel, y_pixel

    def load_coco_ood(self) -> np.ndarray:
        """ """

        number = self.files[np.random.randint(0, len(self.files))]

        segm = Image.open(join(self.annott, number))
        annott_segm_arr = np.array(segm)

        # load coco image
        path = join(self.coco_dir, number.replace("png", "jpg"))
        img = Image.open(path)

        annott_img_arr = np.array(img.convert("RGBA"))

        # elim all not segmentated Pixels
        for i in range(annott_segm_arr.shape[0]):
            for j in range(annott_segm_arr.shape[1]):
                if annott_segm_arr[i, j] == 0:
                    annott_img_arr[i, j] = [0, 0, 0, 0]

        return annott_img_arr

    # TODO md5 sum
    def check_dataset(self):
        return os.path.exists(
            join(self.coco_dir, f"annotations/instances_train{str(self.year)}.json")
        )

    def download_prepare_data(self):
        if not self.check_dataset():
            self.download()

        tools = COCO(join(self.coco_dir, f"annotations/instances_train{str(self.year)}.json"))
        save_dir = join(
            self.coco_dir, f"/annotations/for_{self.dataset}_seg_train{str(self.year)}"
        )
        print(f"Creating segmentation masks for {self.dataset} in {save_dir}")
        # Classes that are also in the main dataset --> don't use these overlap_classes for coco outlier
        if self.dataset == "bddAnomaly":
            overlap_classes = ["train", "bicycle", "motorcycle"]
        elif self.dataset == "Streethazards":
            overlap_classes = [
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

        prohibet_image_ids = []
        # Iterate overall overlap categories to find all prohibed image ids
        for id in tools.getCatIds(catNms=overlap_classes):
            prohibet_image_ids.append(tools.getImgIds(catIds=id))
        # Eliminate duplications
        prohibet_image_ids = [item for sublist in prohibet_image_ids for item in sublist]
        prohibet_image_ids = set(prohibet_image_ids)

        # find all usable images
        usable_image_ids = []
        for image in os.listdir(self.images_dir):
            img_id = image[:-4]
            if int(img_id) not in prohibet_image_ids:
                img = tools.loadImgs(int(img_id))[0]
                # check size of the image
                if img["height"] >= self.min_size_of_img and img["width"] >= self.min_size_of_img:
                    # append image id
                    usable_image_ids.append(img_id)

        # start ground truth segmentaion mask creation
        os.makedirs(save_dir, exist_ok=True)
        for i, img_id in enumerate(usable_image_ids):
            img = tools.loadImgs(img_id)[0]
            # load annotations from annotation id (based on image id)
            annotations = tools.loadAnns(tools.getAnnIds(imgIds=img["id"], iscrowd=None))
            mask = np.ones((img["height"], img["width"]), dtype="uint8") * self.in_class_label
            for j in range(len(annotations)):
                mask = np.maximum(tools.annToMask(annotations[j]) * self.out_class_label, mask)

            # write mask
            for j in range(len(annotations)):
                mask[tools.annToMask(annotations[j]) == 1] = self.out_class_label

            image = Image.fromarray(mask)
            save_path = join(save_dir, "{:012d}.png".format(img_id))
            image.save(save_path)

    def _check_integrity(self) -> bool:
        fpath = os.path.join(self.root, self.filename)
        return check_integrity(fpath, self.md5hash)

    # TODO hübsch machen
    def download(self) -> None:
        # if self._check_integrity():
        #     # log.debug("Files already downloaded and verified")
        #     print("Files already downloaded and verified")
        #     return
        download_and_extract_archive(
            self.img_url, self.coco_dir, filename=f"train{str(self.year)}.zip"
        )
        download_and_extract_archive(
            self.annottations_url,
            self.coco_dir,
            filename=f"annotations_trainval{str(self.year)}.zip",
        )
