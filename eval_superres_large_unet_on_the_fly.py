#!python
#AUM Shree Ganeshaya Namha

from monai.utils import set_determinism, first
from monai.transforms import (
    EnsureChannelFirstD,
    Compose, ConcatItemsd,
    LoadImageD,CopyItemsd,
    RandRotateD,RandAffined,
    RandZoomD,CropForegroundd,
    GaussianSmoothd, RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd, Resized
)
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
from transforms import RandMakeStackd


MODEL_FILE = '/project/ajoshi_27/code_farm/deepSVR/model_64_unet_large_lrem4_hcp/epoch_1840.pth'
#MODEL_FILE = '/home/ajoshi/epoch_370.pth'

print_config()
set_determinism(42)


image_files = glob('./normal_mris_data/sub*/sub-*_T2w_image.nii.gz')

sublist_full = glob(
    './feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')

sublist_full = glob('/project/ajoshi_27/HCP_All/*/T1w/T1*.nii.gz')


# training files

subfiles_train = sublist_full[:60]
subfiles_val = sublist_full[60:70]
subfiles_test = sublist_full[70:]

# '/deneb_disk/feta_2022/feta_2.2/sub-029/anat/sub-029_rec-mial_T2w.nii.gz'

training_datadict = [{"image": item} for item in subfiles_train]
valid_datadict = [{"image": item} for item in subfiles_val]


#training_datadict = d0 + d1 + d2 + d3 + d4 + d5

#print("\n first training items: ", training_datadict)


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

        RandMakeStackd(keys=["stack0", "stack1", "stack2",
                       "stack3", "stack4", "stack5"], stack_axis=0),

        Resized(keys=["stack0", "stack1", "stack2", "stack3",
                "stack4", "stack5"], spatial_size=[64, 64, 64]),

        ConcatItemsd(keys=["stack0", "stack1", "stack2",
                     "stack3", "stack4", "stack5"], name='stacks'),
        # Resized(keys=["image", "stack0", "stack1", "stack2","stack3","stack4","stack5"],spatial_size=[32,32,32]),
    ]
)


check_ds = Dataset(data=training_datadict, transform=randstack_transforms)
check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
check_data = first(check_loader)
image = check_data["image"]
stack = check_data["stacks"]

print(f"image shape: {image.shape}")
print(f"stack shape: {stack.shape}")
""" 
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[0, 0, 32], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("stack")
plt.imshow(stack[0, 0, 32], cmap="gray")
plt.savefig('sample_data.png')

plt.show() """

train_ds = CacheDataset(data=training_datadict, transform=randstack_transforms,
                        cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)


device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = unet.UNet(
    spatial_dims=3,
    in_channels=6,  # moving and fixed
    out_channels=1,
    channels=(16, 32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2, 2),
    kernel_size = 5,
    up_kernel_size= 5,
    num_res_units=3).to(device)

'''#small u net
model = unet.UNet(
    spatial_dims=3,
    in_channels=6,  # moving and fixed
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2).to(device)
'''

image_loss = MSELoss()

optimizer = torch.optim.Adam(model.parameters(), 1e-3)

max_epochs = 4880
epoch_loss_values = []


# load weights
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()


batch_size=16
val_ds = CacheDataset(data=training_datadict[0:16], transform=randstack_transforms,
                      cache_rate=1.0, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, num_workers=0)
for batch_data in val_loader:
    image = batch_data["image"].to(device)
    stacks = batch_data["stacks"]
    pred_image = model(stacks)
    break

image = image.numpy()[:, 0]
stack0 = stacks.numpy()[:, 0]
pred_image = pred_image.detach().numpy()[:, 0]

for i in range(batch_size):
    write_nifti(image[i],'image'+str(i)+'.nii.gz')
    write_nifti(stack0[i],'stack0'+str(i)+'.nii.gz')
    write_nifti(pred_image[i],'pred_img'+str(i)+'.nii.gz')
