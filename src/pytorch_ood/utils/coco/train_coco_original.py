import os
import random
from os.path import join

import numpy as np
import torch
from PIL import Image


class TrainCocoTransform(object):
    """ """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self.coco_dir = cfg.ood.directory
        self.annott = cfg.ood.annotations
        self.files = os.listdir(self.annott)

    def __call__(self, img, segm):
        if random.random() <= self.cfg.ood.ood_rate:
            segm = Image.fromarray(np.array(segm, dtype=np.uint8))
            img, segm = self.add_ood(img, segm)
            segm = torch.tensor(segm, dtype=torch.int64)

        return img, segm

    def add_ood(self, img, segm):
        """ """
        for elem in range(self.cfg.ood.ood_per_image):
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
                        segm_arr[i + y_pos, j + x_pos] = self.cfg.ood.mask_value

            segm = Image.fromarray(segm_arr)

        return img, segm_arr

    def _random_pos_and_scale(self, orig_img_dim):
        """ """
        clip_image = Image.fromarray(self.load_coco_ood())

        # scale_range=[20,50]
        # we rescale since COCO images can be of different size
        # upscale=1.4150357439499515
        upscale = self.cfg.ood.upscale
        scale_range = [int(20 * upscale), int(50 * upscale)]
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
