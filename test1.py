from monai.utils import set_determinism, first
from monai.transforms import (
    EnsureChannelFirstD,
    Compose, ConcatItemsd,Resize,
    LoadImageD,CopyItemsd,
    RandRotateD,RandAffined,
    RandZoomD,CropForegroundd,
    GaussianSmoothd, RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd, Resized
)
from tqdm import tqdm
from network_mods import GlobalNetRigid
from monai.data import DataLoader, Dataset, CacheDataset
from monai.config import print_config, USE_COMPILED
from monai.networks.blocks import Warp
from monai.apps import MedNISTDataset
from monai.networks.nets import unet
import numpy as np
import torch
from torch.nn import MSELoss, CrossEntropyLoss
#import matplotlib.pyplot as plt
import os
import tempfile
from glob import glob
from monai.data.nifti_writer import write_nifti
from easy_transforms import RandMakeStackd


set_determinism(42)


#sublist_full = glob('./feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')

sublist_full = glob('/home/ajoshi/T1w*.nii.gz')


# training files

subfiles_train = sublist_full[0:1]


training_datadict = [{"image": item} for item in subfiles_train]




randstack_transforms = Compose(
    [
        LoadImageD(keys=["image"]),
        EnsureChannelFirstD(keys=["image"]),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=[64, 64, 64]),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=2, upper=98, b_min=0.0, b_max=10.0, clip=True),
        # make stacks
        RandAffined(mode=("bilinear"), prob=1.0, translate_range=(5, 5, 5), rotate_range=(
            np.pi/4, np.pi/4, np.pi/4), padding_mode="zeros", keys=["image"]),
        CopyItemsd(keys=["image", "image", "image", "image", "image", "image"], names=[
                   "stack0", "stack1", "stack2", "stack3", "stack4", "stack5"]),

        RandMakeStackd(keys=["stack0", "stack1"], stack_axis=0),
        RandMakeStackd(keys=["stack2", "stack3"], stack_axis=1),
        RandMakeStackd(keys=["stack4", "stack5"], stack_axis=2),

        Resized(keys=["stack0", "stack1", "stack2", "stack3",
                "stack4", "stack5"], spatial_size=[64, 64, 64]),

        ConcatItemsd(keys=["stack0", "stack1", "stack2",
                     "stack3", "stack4", "stack5"], name='stacks'),
        # Resized(keys=["image", "stack0", "stack1", "stack2","stack3","stack4","stack5"],spatial_size=[32,32,32]),
    ]
)

train_ds = CacheDataset(data=training_datadict, transform=randstack_transforms,
                        cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_data = first(train_loader)
image = batch_data["image"].to(device)
stacks = batch_data["stacks"].to(device)

write_nifti(image[0,0],'outsvr/deepsvr_orig_image.nii.gz')

write_nifti(stacks[0,0],'outsvr/deepsvr_orig_stack0.nii.gz')
write_nifti(stacks[0,1],'outsvr/deepsvr_orig_stack1.nii.gz')
write_nifti(stacks[0,2],'outsvr/deepsvr_orig_stack2.nii.gz')
write_nifti(stacks[0,3],'outsvr/deepsvr_orig_stack3.nii.gz')
write_nifti(stacks[0,4],'outsvr/deepsvr_orig_stack4.nii.gz')
write_nifti(stacks[0,5],'outsvr/deepsvr_orig_stack5.nii.gz')
