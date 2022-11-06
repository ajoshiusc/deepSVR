#!python
#AUM Shree Ganeshaya Namha

from monai.utils import set_determinism, first
from monai.transforms import (
    EnsureChannelFirstD,
    Compose, ConcatItemsd,
    LoadImageD,
    RandRotateD,
    RandZoomD,
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
import matplotlib.pyplot as plt
import os
import tempfile
from glob import glob
from monai.data.nifti_writer import write_nifti

MODEL_FILE = '/home/ajoshi/projects/deepSVR/model_64_unet_lrem4/epoch_550.pth'

print_config()
set_determinism(42)

image_files = glob('./feta_syn_data/sub-*_T2w/sub-*_T2w_image.nii.gz')


training_datadict = [{"image": item, "stack0": item[:-13]+'_stack_x_0.nii.gz',
                      "stack1": item[:-13]+'_stack_x_1.nii.gz', "stack2": item[:-13]+'_stack_y_0.nii.gz',
                      "stack3": item[:-13]+'_stack_y_1.nii.gz', "stack4": item[:-13]+'_stack_z_0.nii.gz',
                      "stack5": item[:-13]+'_stack_z_1.nii.gz'} for item in image_files]


#training_datadict = d0 + d1 + d2 + d3 + d4 + d5

#print("\n first training items: ", training_datadict)


train_transforms = Compose(
    [
        LoadImageD(keys=["image", "stack0", "stack1",
                   "stack2", "stack3", "stack4", "stack5"]),
        EnsureChannelFirstD(
            keys=["image", "stack0", "stack1", "stack2", "stack3", "stack4", "stack5"]),
        Resized(keys=["stack0", "stack1", "stack2", "stack3",
                "stack4", "stack5"], spatial_size=[64, 64, 64]),
        ConcatItemsd(keys=["stack0", "stack1", "stack2",
                     "stack3", "stack4", "stack5"], name='stacks'),
        #Resized(keys=["image", "stack0", "stack1", "stack2","stack3","stack4","stack5"],spatial_size=[32,32,32]),
        ScaleIntensityRangePercentilesd(keys=["image", "stack0", "stack1", "stack2", "stack3", "stack4", "stack5", "stacks"],lower=2,upper=98,b_min=0.0, b_max=10.0, clip=True),

    ]
)

check_ds = Dataset(data=training_datadict, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
check_data = first(check_loader)
image = check_data["image"]
stack = check_data["stacks"]

print(f"image shape: {image.shape}")
print(f"stack shape: {stack.shape}")

plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[0, 0, 32], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("stack")
plt.imshow(stack[0, 0, 32], cmap="gray")
plt.savefig('sample_data.png')

plt.show()

train_ds = CacheDataset(data=training_datadict, transform=train_transforms,
                        cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)


device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = unet.UNet(
    spatial_dims=3,
    in_channels=6,  # moving and fixed
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2).to(device)
image_loss = MSELoss()

optimizer = torch.optim.Adam(model.parameters(), 1e-3)

max_epochs = 4880
epoch_loss_values = []


# load weights
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()


batch_size=16
val_ds = CacheDataset(data=training_datadict[0:16], transform=train_transforms,
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
