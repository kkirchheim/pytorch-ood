import json
import logging
import os
from os.path import join
from typing import Any, Callable, Optional, Tuple

import scipy
import tqdm
from PIL import Image
from torchvision.transforms.functional import to_tensor

from pytorch_ood.dataset.img.base import ImageDatasetBase

log = logging.getLogger(__name__)

# inspired by https://github.com/tla93/InpaintingOutlierSynthesis/blob/main/src/bddanomaly.py


class BDDAnomaly(ImageDatasetBase):
    """
    Benchmark Dataset for Anomaly Segmentation.

    From the paper *Scaling Out-of-Distribution Detection for Real-World Settings*

    """

    subset_list = ["test", "train", "validation"]

    root_dir_name = "bdd100k"

    filename_list = {
        "test": "./data/bdd100k/anom.odgt",
        "train": "./data/bdd100k/train.odgt",
        "validation": "./data/bdd100k/validation.odgt",
    }

    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable[[Tuple], Tuple]] = None,
        download: bool = False,
        prefix_img="./",
    ) -> None:
        """
        :param root: root path for dataset
        :param subset: one of ``train``, ``test``, ``validation``
        :param transform: transformations to apply to images and masks, will get tuple as argument
        :param download: if dataset should be downloaded automatically
        :param prefix_img: TODO
        """
        root = join(root, self.root_dir_name)
        super(ImageDatasetBase, self).__init__(root, transform=transform)

        with open(self.filename_list[subset], "r") as f:
            data = f.read()

        odgt = json.loads(data)

        if download:
            self.download()

        if subset not in self.subset_list:
            raise ValueError(f"Invalid subset: {subset}")

        all_img = [join(root, prefix_img, elem["fpath_img"]) for elem in odgt]
        all_segm = [join(root, prefix_img, elem["fpath_segm"]) for elem in odgt]

        self.all_img, self.all_segm = all_img, all_segm

    def __len__(self) -> int:
        return len(self.all_segm)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: index
        :returns: (image, target) where target is the annotation of the image.
        """
        file, target = self.all_img[index], self.all_segm[index]

        # to return a PIL Image
        img = Image.open(file)
        target = to_tensor(Image.open(target)).squeeze(0)
        target = (target * 255).long() - 1  # labels to integer
        target[target >= 17] = -1  # negative labels for outliers

        # get img_id
        id = self.all_img[index].split("/")[-1]  # get img_name
        id = id.split(".")[0]  # get id_name

        if self.transform is not None:
            # # id is necassary for Inpainting outliers
            # img, target = self.transform(img, target, id)
            # id is unnecassary for Inpainting outliers
            img, target = self.transform(img, target)

        return img, target

    def preparation(self):
        """
        Prepare the dataset for training
        """
        # download data
        self.download()
        # prepare data
        self.prepare_data()

    def prepare_data(self):
        """
        Prepare the data for training
        """
        # TODO

        pass

    @staticmethod
    # code snippet from https://github.com/hendrycks/anomaly-seg/blob/master/create_dataset.py
    def convert_bdd(root_dir, ann_dir):
        count = 0
        for img_loc in tqdm(os.listdir(root_dir + ann_dir)):
            img = Image.open(root_dir + ann_dir + img_loc)
            if img.ndim <= 1:
                continue
            # swap 255 with -1
            # 16 -> 19
            # 18 -> 16
            # 19 -> 18
            # add 1 to whole array
            loc = img == 255
            img[loc] = -1
            loc = img == 16
            img[loc] = 19
            loc = img == 18
            img[loc] = 16
            loc = img == 19
            img[loc] = 18
            img += 1
            scipy.misc.toimage(img, cmin=0, cmax=255).save(root_dir + ann_dir + img_loc)
