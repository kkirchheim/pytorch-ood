import io
import json
import logging
import os
import shutil
import zipfile
from os.path import join
from typing import Any, Callable, Optional, Tuple

import numpy as np
import requests
import scipy
import tqdm
from PIL import Image
from scipy.misc import imread, imsave
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

    root_dir_offset = "/data/bdd100k/"

    filename_list = {
        "test": "anom.odgt",
        "train": "train.odgt",
        "validation": "validation.odgt",
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
        root = join(root, self.root_dir_offset)
        super(ImageDatasetBase, self).__init__(root, transform=transform)

        if subset not in self.subset_list:
            raise ValueError(f"Invalid subset: {subset}")
        if download:
            self.download_and_preprocessing()
        try:
            with open(self.filename_list[subset], "r") as f:
                data = f.read()
        except:
            raise ValueError(
                f"File {self.filename_list[subset]} not found! Pleae download the dataset first with download=True."
            )

        odgt = json.loads(data)

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
            # id is unnecassary
            img, target = self.transform(img, target)

        return img, target

    def download_and_preprocessing(self):
        """
        Download and preprocess the dataset
        """
        # download data
        self.download()
        # prepare data
        self.prepare_data()
        # create odgt files
        self.create_odgt_for_all()

    def download(self):
        """
        Download the dataset
        """
        # TODO download folder for bdd100k annotations to root
        # https://github.com/hendrycks/anomaly-seg/tree/4e7b1de5049500c30c07cef6925a96dd5304791b/seg
        # URL of the repository
        # URL of the repository
        repo_url = "https://github.com/hendrycks/anomaly-seg"

        # Folder you want to download
        folder_path = "anomaly-seg-master/seg"

        # Download the repository as a zip file
        response = requests.get(repo_url + "/archive/master.zip")

        # Make sure the request was successful
        assert response.status_code == 200

        # Open the zip file in memory
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        # Extract the specific folder to the current directory
        for file in zip_file.namelist():
            # print(file)
            if file.startswith(folder_path):
                zip_file.extract(join(self.root, file))  #
        os.rename(join(self.root, "anomaly-seg-master/seg"), join(self.root, "seg"))
        # clean up
        shutil.rmtree(join(self.root, "anomaly-seg-master"))
        # TODO download dataset from http://bdd-data.berkeley.edu/download.html
        # currently not possible because of registration
        pass

    def prepare_data(self):
        """
        Prepare the data from dataset after downloading
        """

        self.convert_bdd("seg/train_labels/train/")
        # TODO habe ich den validation spaß überhaupt gemacht???
        self.convert_bdd("seg/train_labels/val/")

    def create_odgt_for_all(self):
        """
        Create odgt files for all subsets
        """
        self.create_odgt("seg/images/train/", "seg/train_labels/train/", self.subset_list["train"])
        self.create_odgt(
            "seg/images/val/", "seg/train_labels/val/", self.subset_list["validation"]
        )

    def create_odgt(
        self,
        file_dir,
        ann_dir,
        odgt_file,
    ):
        """
        Create odgt files
        """
        # check if anom.odgt exists
        if os.path.exists(join(self.root, self.subset_list["test"])):
            anom_files = json.load(open(join(self.root, self.subset_list["test"])))
        else:
            anom_files = []
        _files = []

        in_count = 0
        out_count = 0
        img_files = sorted(os.listdir(self.root + file_dir))
        for img in img_files:
            ann_file = img[:-4] + "_train_id.png"
            ann_file_path = join(self.root, ann_dir) + ann_file
            if os.path.exists(ann_file_path):
                dict_entry = {
                    # "dbName": "BDD100k",
                    # "width": 1280,
                    # "height": 720,
                    "fpath_img": file_dir + img,
                    "fpath_segm": ann_dir + ann_file,
                }
                img = imread(ann_file_path)
                cond1 = np.logical_or((img == 18), (img == 19))
                if np.any(np.logical_or(cond1, (img == 20))):
                    out_count += 1
                    anom_files.append(dict_entry)
                else:
                    in_count += 1
                    _files.append(dict_entry)

        print("total images in = {} and out =  {}".format(in_count, out_count))

        # save odgt files
        with open(odgt_file, "w") as outfile:
            json.dump(_files, outfile)
        with open(join(self.root, self.subset_list["test"], "w")) as outfile:
            json.dump(anom_files, outfile)

        # TODO check if numbers of images per odgt is correct
        pass

    # code snippet from https://github.com/hendrycks/anomaly-seg/blob/master/create_dataset.py
    def convert_bdd(self, ann_dir):
        for img_loc in tqdm(os.listdir(join(self.root, ann_dir))):
            img = Image.open(join(self.root, ann_dir) + img_loc)
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
            scipy.misc.toimage(img, cmin=0, cmax=255).save(join(self.root, ann_dir) + img_loc)
