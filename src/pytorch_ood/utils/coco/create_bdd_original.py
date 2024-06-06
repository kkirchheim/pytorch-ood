import json
import os
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
import scipy.misc
from scipy.misc import imread, imsave
from tqdm import tqdm

# replace the colors with our colors
# a = sio.loadmat("data_ADE/color150.mat")
# print(a)

colors = np.array(
    [
        0,  # road
        1,  # sidewalk
        2,  # building
        3,  # wall
        4,  # fence
        5,  # pole
        6,  # traffic light
        7,  # traffic sign
        8,  # vegetation
        9,  # terrain
        10,  # sky
        11,  # person
        12,  # rider
        13,  # car
        14,  # truck
        15,  # bus
        16,  # train
        17,  # motorcycle
        18,  # bicycle
        255,
    ]
)  # other

# swap 255 with -1
# add 2 to whole array

# a["colors"] = colors
# print(a)
# sio.savemat("data/color150.mat", a)


#####
# create the train and val obgt

## To view the structure of their obgt file uncomment
## the lines below
# odgt = "data_ADE/train.odgt"
#
# with open(odgt) as fp:
#     a = json.loads(fp.read())
#     print(a, type(a))
#
# a = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
# print(a, type(a), type(a[0]), len(a), "\n\n", a[0])


def create_odgt(root_dir, file_dir, ann_dir, out_dir, anom_files=None):
    if anom_files is None:
        anom_files = []
    _files = []

    count1 = 0
    count2 = 0
    img_files = sorted(os.listdir(root_dir + file_dir))
    for img in img_files:
        # print(img, img[-5])
        # this line is because all of train images
        # are saved as "type5.png"
        # ann_file = img[:-5] + "5" + img[-4:]
        ann_file = img[:-4] + "_train_id.png"
        # print(ann_file)
        ann_file_path = root_dir + ann_dir + ann_file
        if os.path.exists(ann_file_path):
            # print("exists")
            dict_entry = {
                "dbName": "BDD100k",
                "width": 1280,
                "height": 720,
                "fpath_img": file_dir + img,
                "fpath_segm": ann_dir + ann_file,
            }
            img = imread(ann_file_path)
            cond1 = np.logical_or((img == 18), (img == 19))
            if np.any(np.logical_or(cond1, (img == 20))):
                count2 += 1
                anom_files.append(dict_entry)
            else:
                count1 += 1
                _files.append(dict_entry)

    print("total images in = {} and out =  {}".format(count1, count2))

    with open(out_dir, "w") as outfile:
        json.dump(_files, outfile)
    with open(root_dir + "anom.odgt", "w") as outfile:
        json.dump(anom_files, outfile)

    # for i in training_files:
    #     json.dumps(i, outfile)
    return anom_files


# do train first
out_dir = "data/train.odgt"
root_dir = "data/"
train_dir = "seg/images/train/"
ann_dir = "seg/train_labels/train/"
anom_files = create_odgt(root_dir, train_dir, ann_dir, out_dir)

out_dir = "data/validation.odgt"
root_dir = "data/"
train_dir = "seg/images/val/"
ann_dir = "seg/train_labels/val/"
create_odgt(root_dir, train_dir, ann_dir, out_dir, anom_files=anom_files)


# sanity check to make sure it can be loaded back
# a = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

# print(a)
# print(a, type(a), type(a[0]), len(a[0]), "\n\n",)


### convert annotation images to correct labels


def convert_cityscapes_to_uint(root_dir, ann_dir):
    count = 0
    for img_loc in tqdm(os.listdir(root_dir + ann_dir)):
        img = imread(root_dir + ann_dir + img_loc)
        if img.ndim <= 1:
            continue
        # img = img[:,:,:3]
        # print(img.shape, img[0],)
        # swap 255 with -1
        # add 2 to whole array
        loc = img == 255
        img[loc] = -1
        img += 2
        # plt.imshow(new_img)
        # plt.show()
        # imsave(root_dir+ann_dir+img_loc, new_img)  # SCIPY RESCALES from 0-255 on its own
        scipy.misc.toimage(img, cmin=0, cmax=255).save(root_dir + ann_dir + img_loc)


root_dir = "data/"
ann_dir = "seg/train_labels/train/"
# convert the training images
# convert_cityscapes_to_uint(root_dir, ann_dir)

root_dir = "data/"
ann_dir = "seg/train_labels/val/"
# convert the anomaly images
# convert_cityscapes_to_uint(root_dir, ann_dir)

# convert the val images
# ann_dir = "annotations/validation/"
# convert_cityscapes_to_uint(root_dir, ann_dir)
