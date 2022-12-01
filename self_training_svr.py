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
from tqdm.notebook import tqdm
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


image_files = glob('/project/ajoshi_27/code_farm/deepSVR/normal_mris_data/sub*/sub-*_T2w_image.nii.gz')

sublist_full = glob('./feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')



# training files

subfiles_train = sublist_full[7:8]


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_data = first(train_loader)
image = batch_data["image"].to(device)
stacks = batch_data["stacks"].to(device)

reg = GlobalNetRigid(
    image_size=(64, 64, 64),
    spatial_dims=3,
    in_channels=2,  # moving and fixed
    num_channel_initial=16,
    depth=4).to(device)

if USE_COMPILED:
    warp_layer = Warp(3, "border").to(device)
else:
    warp_layer = Warp("bilinear", "border").to(device)
    

def masked_mse(moving, slicevol):
    msk = slicevol > 0
    err = (moving-slicevol)*msk
    loss = torch.sqrt(torch.sum(err**2))
    return loss
    
image_loss = masked_mse

superres = unet.UNet(
    spatial_dims=3,
    in_channels=6,  # moving and fixed
    out_channels=1,
    channels=(16, 32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2, 2),
    kernel_size=5,
    up_kernel_size=5,
    num_res_units=3).to(device)

#reg.load_state_dict(torch.load('/project/ajoshi_27/code_farm/deepSVR/model_64_slice2vol_reg/epoch_3980.pth'));
reg.train()
#superres.load_state_dict(torch.load('/project/ajoshi_27/code_farm/deepSVR/model_64_unet_large_lrem4_hcp_easy/epoch_1070.pth'));
superres.train()
optimizerR = torch.optim.Adam(reg.parameters(), 1e-4)
optimizerS = torch.optim.Adam(superres.parameters(), 1e-4)

max_epochs = 500000
for epoch in range(max_epochs):

    vol_loss = 0

    for sliceno in range(int(64/2)):

        #valid_moving = image.detach().to(device)
        batch_size = stacks.shape[0]

        slice_vol = torch.zeros(batch_size, 1, 64, 64, 64).to(device)
        slice_ind = torch.tensor(range(2*sliceno, 2*(sliceno+1)))

        for d in range(6):

            optimizerR.zero_grad()
            optimizerS.zero_grad()

            recon_image = superres(stacks)

            if int(d/2)==0:
                slice_vol[0, :, slice_ind, :, :] = stacks[0, d, slice_ind, :, :].to(
                device)

            elif int(d/2)==1:
                slice_vol[0, :, :, slice_ind, :] = stacks[0, d, :, slice_ind, :].to(
                    device)
            elif int(d/2)==2:
                slice_vol[0, :, :, :, slice_ind] = stacks[0, d, :, :, slice_ind].to(
                    device)


            ddf = reg(torch.cat((recon_image, slice_vol), dim=1))
            recon_image_moved = warp_layer(recon_image, ddf)

            slice_loss = image_loss(recon_image_moved, slice_vol)

            slice_loss.backward()
            optimizerR.step()
            optimizerS.step()

        vol_loss += slice_loss

    print(f'epoch_loss:{vol_loss} for epoch:{epoch}')    
   
    

