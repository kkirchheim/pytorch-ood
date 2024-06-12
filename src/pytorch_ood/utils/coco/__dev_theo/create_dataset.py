import json
import os
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
import scipy.misc
from PIL import Image
from tqdm import tqdm

# Replace the colors with our colors
# This is only used for visualization purposes
# color_mat = sio.loadmat("data_ADE/color150.mat")

# StreetHazards colors
# colors = np.array([[ 0,   0,   0],# // unlabeled     =   0,
#        [ 70,  70,  70],# // building      =   1,
#        [190, 153, 153],# // fence         =   2,
#        [250, 170, 160],# // other         =   3,
#        [220,  20,  60],# // pedestrian    =   4,
#        [153, 153, 153],# // pole          =   5,
#        [157, 234,  50],# // road line     =   6,
#        [128,  64, 128],# // road          =   7,
#        [244,  35, 232],# // sidewalk      =   8,
#        [107, 142,  35],# // vegetation    =   9,
#        [  0,   0, 142],# // car           =  10,
#        [102, 102, 156],# // wall          =  11,
#        [220, 220,   0],# // traffic sign  =  12,
#        [ 60, 250, 240],# // anomaly       =  13,
#
#        ])

# color_mat["colors"] = colors
# sio.savemat("data/color150.mat", color_mat)


#####
# create the train and val obgt


def create_odgt(root_dir, file_dir, ann_dir, out_dir, anom_files=None):
    if anom_files is None:
        anom_files = []
    _files = []

    count = total = 0
    town_names = sorted(os.listdir(root_dir + file_dir))

    for town in town_names:
        img_files = sorted(os.listdir(os.path.join(root_dir, file_dir, town)))
        total += len(img_files)
        for img in img_files:
            ann_file = img
            ann_file_path = os.path.join(root_dir, ann_dir, town, ann_file)
            if os.path.exists(ann_file_path):
                dict_entry = {
                    "dbName": "StreetHazards",
                    "width": 1280,
                    "height": 720,
                    "fpath_img": os.path.join(file_dir, town, img),
                    "fpath_segm": os.path.join(ann_dir, town, ann_file),
                }
                # If converting BDD100K uncomment out the following
                # img = Image.open(ann_file_path)
                # if np.any(np.logical_or( (img == 19), (img == 20) )):
                #    anom_files.append(dict_entry)
                # else:
                count += 1
                _files.append(dict_entry)

    print("total images in = {} and out =  {}".format(total, count))

    with open(out_dir, "w") as outfile:
        json.dump(_files, outfile)

    # If converting BDD100K uncomment out the following
    # with open(root_dir + "anom.odgt", "w") as outfile:
    #    json.dump(anom_files, outfile)

    return anom_files


out_dir = "data/training.odgt"
# modify root directory to reflect the location of where the streethazards_train was extracted to.
root_dir = "data/"
train_dir = "train/images/training/"
ann_dir = "train/annotations/training/"
anom_files = create_odgt(root_dir, train_dir, ann_dir, out_dir)


out_dir = "data/validation.odgt"
train_dir = "train/images/validation/"
ann_dir = "train/annotations/validation/"
create_odgt(root_dir, train_dir, ann_dir, out_dir, anom_files=anom_files)


out_dir = "data/test.odgt"
val_dir = "test/images/test/"
ann_dir = "test/annotations/test/"
create_odgt(root_dir, val_dir, ann_dir, out_dir)


# BDD100K label map
# colors = np.array(
#    [0,    # road
#    1,     #sidewalk
#    2,     # building
#    3,     # wall
#    4,     # fence
#    5,     # pole
#    6,     # traffic light
#    7,     # traffic sign
#    8,     # vegetation
#    9,     # terrain
#    10,    # sky
#    11,    # person
#    12,    # rider
#    13,    # car
#    14,    # truck
#    15,    # bus
#    16,    # train
#    17,    # motorcycle
#    18,    # bicycle
#    255,]) # other

### convert BDD100K semantic segmentation images to correct labels


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


# root_dir = "data/"
# ann_dir = "seg/train_labels/train/"
# convert the BDD100K semantic segmentation images.
# convert_bdd(root_dir, ann_dir)
