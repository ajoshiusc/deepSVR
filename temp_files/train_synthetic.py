
import monai
import torch
import itk
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import os.path
import tempfile

from utils import (
    preview_image, preview_3D_vector_field, preview_3D_deformation,
    jacobian_determinant, plot_against_epoch_numbers
)

monai.config.print_config()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=2938649572)

use_synthetic_data = True
directory = '/home/ajoshi/monai_data' #os.environ.get("MONAI_DATA_DIRECTORY")

root_dir = tempfile.mkdtemp() if directory is None else directory
data_dir = os.path.join(root_dir, "synthetic_data")

save_img = monai.transforms.SaveImage(output_dir=data_dir, output_postfix="img", print_log=False)
save_seg = monai.transforms.SaveImage(output_dir=data_dir, output_postfix="seg", print_log=False)

# Set the amount of synthetic data here
num_img_seg_pairs_to_generate = 50
num_segmentation_classes = 5
num_segs_to_select = 6

for i in range(num_img_seg_pairs_to_generate):
    img, seg = monai.data.synthetic.create_test_image_3d(
        64, 64, 64,  # image size
        num_objs=5,
        rad_max=30, rad_min=4,
        noise_max=0.5,
        num_seg_classes=num_segmentation_classes - 1,  # background is not counted
        channel_dim=0,
        random_state=None
    )
    save_img(img)
    if i < num_segs_to_select:  # pretend that only a few segmentations are available
        save_seg(seg)

image_paths = glob.glob(os.path.join(data_dir, "*/*img.nii.gz"))
segmentation_paths = glob.glob(os.path.join(data_dir, "*/*seg.nii.gz"))

print(segmentation_paths)








device = torch.device("cuda:0")


# Function to extract an image or segmentation ID from its path
def path_to_id(path):
    if use_synthetic_data:
        return os.path.basename(path).split('_')[0]
    else:
        return os.path.basename(path).strip('OAS1_')[:8]


seg_ids = list(map(path_to_id, segmentation_paths))
img_ids = map(path_to_id, image_paths)
data = []
for img_index, img_id in enumerate(img_ids):
    data_item = {'img': image_paths[img_index]}
    if img_id in seg_ids:
        data_item['seg'] = segmentation_paths[seg_ids.index(img_id)]
    data.append(data_item)




data_seg_available = list(filter(lambda d: 'seg' in d.keys(), data))
data_seg_unavailable = list(filter(lambda d: 'seg' not in d.keys(), data))

data_seg_available_train, data_seg_available_valid = \
    monai.data.utils.partition_dataset(data_seg_available, ratios=(8, 2))
# Validation of the segmentation network only makes sense if you have enough segmentation labels.
# E.g. definitely skip validation here if there's just one segmentation label.
#     

svr_net = monai.networks.nets.UNet(
    3,  # spatial dims
    1,  # input channels
    1,  # output channels
    (8, 16, 16, 32, 32, 64, 64),  # channel sequence
    (1, 2, 1, 2, 1, 2),  # convolutional strides
    dropout=0.2,
    norm='batch'
)

resize = 96 if not use_synthetic_data else None

data_seg_available = list(filter(lambda d: 'seg' in d.keys(), data))
data_seg_unavailable = list(filter(lambda d: 'seg' not in d.keys(), data))

data_seg_available_train, data_seg_available_valid = \
    monai.data.utils.partition_dataset(data_seg_available, ratios=(8, 2))
# Validation of the segmentation network only makes sense if you have enough segmentation labels.
# E.g. definitely skip validation here if there's just one segmentation label.

transform_seg_available = monai.transforms.Compose(
    transforms=[
        monai.transforms.LoadImageD(keys=['img', 'seg'], image_only=True),
        monai.transforms.TransposeD(keys=['img', 'seg'], indices=(2, 1, 0)),
        monai.transforms.EnsureChannelFirstD(keys=['img', 'seg']),
        monai.transforms.ResizeD(
            keys=['img', 'seg'],
            spatial_size=(resize, resize, resize),
            mode=['trilinear', 'nearest'],
            align_corners=[False, None]
        ) if resize is not None else monai.transforms.Identity()
    ]
)


# Supress the many warnings related to deprecation of the Analyze file format
# (without this, we would see warnings when the LoadImage transform calls itk to load Analyze files)
itk.ProcessObject.SetGlobalWarningDisplay(False)

# Uncomment the following lines to preview a random image with the transform above applied
# data_item = transform_seg_available(random.choice(data_seg_available))
# preview_image(data_item['img'][0])
# preview_image(data_item['seg'][0])


dataset_seg_available_train = monai.data.CacheDataset(
    data=data_seg_available_train,
    transform=transform_seg_available,
    cache_num=16
)

dataset_seg_available_valid = monai.data.CacheDataset(
    data=data_seg_available_valid,
    transform=transform_seg_available,
    cache_num=16
)


# During the joint/alternating training process, we will use reuse data_seg_available_valid
# for validating the segmentation network.
# So we should not let the registration or segmentation networks see these images in training.
data_without_seg_valid = data_seg_unavailable + data_seg_available_train  # Note the order

# For validation of the registration network, we prefer not to use the precious data_seg_available_train,
# if that's possible. The following split tries to use data_seg_unavailable for the
# the validation set, to the extent possible.
data_valid, data_train = monai.data.utils.partition_dataset(
    data_without_seg_valid,  # Note the order
    ratios=(2, 8),  # Note the order
    shuffle=False
)


def take_data_pairs(data, symmetric=True):
    """Given a list of dicts that have keys for an image and maybe a segmentation,
    return a list of dicts corresponding to *pairs* of images and maybe segmentations.
    Pairs consisting of a repeated image are not included.
    If symmetric is set to True, then for each pair that is included, its reverse is also included"""
    data_pairs = []
    for i in range(len(data)):
        j_limit = len(data) if symmetric else i
        for j in range(j_limit):
            if j == i:
                continue
            d1 = data[i]
            d2 = data[j]
            pair = {
                'img1': d1['img'],
                'img2': d2['img']
            }
            if 'seg' in d1.keys():
                pair['seg1'] = d1['seg']
            if 'seg' in d2.keys():
                pair['seg2'] = d2['seg']
            data_pairs.append(pair)
    return data_pairs


data_pairs_valid = take_data_pairs(data_valid)
data_pairs_train = take_data_pairs(data_train)


def subdivide_list_of_data_pairs(data_pairs_list):
    out_dict = {'00': [], '01': [], '10': [], '11': []}
    for d in data_pairs_list:
        if 'seg1' in d.keys() and 'seg2' in d.keys():
            out_dict['11'].append(d)
        elif 'seg1' in d.keys():
            out_dict['10'].append(d)
        elif 'seg2' in d.keys():
            out_dict['01'].append(d)
        else:
            out_dict['00'].append(d)
    return out_dict


data_pairs_valid_subdivided = subdivide_list_of_data_pairs(data_pairs_valid)
data_pairs_train_subdivided = subdivide_list_of_data_pairs(data_pairs_train)

# print some useful counts to be aware of

num_train_reg_net = len(data_pairs_train)
num_valid_reg_net = len(data_pairs_valid)
num_train_both = len(data_pairs_train_subdivided['01']) +\
    len(data_pairs_train_subdivided['10']) +\
    len(data_pairs_train_subdivided['11'])


print(f"""We have {num_train_both} pairs to train reg_net and seg_net together,
  and an additional {num_train_reg_net - num_train_both} to train reg_net alone.""")
print(f"We have {num_valid_reg_net} pairs for reg_net validation.")

# The following are dictionaries that map segmentation availability labels 00,10,01,11 to MONAI datasets

transform_pair = monai.transforms.Compose(
    transforms=[
        monai.transforms.LoadImageD(keys=['img1', 'seg1', 'img2', 'seg2'], image_only=True, allow_missing_keys=True),
        monai.transforms.TransposeD(keys=['img1', 'seg1', 'img2', 'seg2'], indices=(2, 1, 0), allow_missing_keys=True),
        monai.transforms.EnsureChannelFirstD(keys=['img1', 'seg1', 'img2', 'seg2'], allow_missing_keys=True),
        monai.transforms.ConcatItemsD(keys=['img1', 'img2'], name='img12', dim=0),
        monai.transforms.DeleteItemsD(keys=['img1', 'img2']),
        monai.transforms.ResizeD(
            keys=['img12', 'seg1', 'seg2'],
            spatial_size=(resize, resize, resize),
            mode=['trilinear', 'nearest', 'nearest'],
            allow_missing_keys=True,
            align_corners=[False, None, None]
        ) if resize is not None else monai.transforms.Identity()
    ]
)
dataset_pairs_train_subdivided = {
    seg_availability: monai.data.CacheDataset(
        data=data_list,
        transform=transform_pair,
        cache_num=32
    )
    for seg_availability, data_list in data_pairs_train_subdivided.items()
}


dataset_pairs_valid_subdivided = {
    seg_availability: monai.data.CacheDataset(
        data=data_list,
        transform=transform_pair,
        cache_num=32
    )
    for seg_availability, data_list in data_pairs_valid_subdivided.items()
}





data_item = random.choice(dataset_seg_available_train)
svr_net_example_output = svr_net(data_item['img'].unsqueeze(0))
print(f"Segmentation classes: {torch.unique(data_item['seg']).as_tensor()}")
print(f"Shape of ground truth label: {data_item['seg'].unsqueeze(0).shape}")
print(f"Shape of seg_net output: {svr_net_example_output.shape}")


# Trying out warping function

warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="border")
warp_nearest = monai.networks.blocks.Warp(mode="nearest", padding_mode="border")

# Use example reg_net output to apply warp
example_warped_image = warp(
    data_item['img'][[0], :, :, :].unsqueeze(0),  # moving image
    svr_net_example_output  # warping
)

# preview_3D_vector_field(reg_net_example_output.detach()[0])


# Uncomment to preview warped image from forward pass example above
preview_image(example_warped_image[0,0].detach())