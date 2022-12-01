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

sublist_full = glob(
    '/project/ajoshi_27/code_farm/deepSVR/feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')



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

        Resized(keys=["stack0", "stack1"], spatial_size=[32, 64, 64]),
        Resized(keys=["stack2", "stack3"], spatial_size=[64, 32, 64]),
        Resized(keys=["stack4", "stack5"], spatial_size=[64, 64, 32]),

        Resized(keys=["stack0", "stack1", "stack2", "stack3",
                "stack4", "stack5"], spatial_size=[64, 64, 64]),

        ConcatItemsd(keys=["stack0", "stack1", "stack2",
                     "stack3", "stack4", "stack5"], name='stacks'),
    ]
)

resize_x_down = Resize(spatial_size=[32, 64, 64])
resize_y_down = Resize(spatial_size=[64, 32, 64])
resize_z_down = Resize(spatial_size=[64, 64, 32])
resize_up = Resize(spatial_size=[64, 64, 64])



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

reg.load_state_dict(torch.load('/project/ajoshi_27/code_farm/deepSVR/model_64_slice2vol_reg/epoch_3980.pth'));
reg.train()
superres.load_state_dict(torch.load('/project/ajoshi_27/code_farm/deepSVR/model_64_unet_large_lrem4_hcp_easy/epoch_1070.pth'));
superres.train()
optimizerR = torch.optim.Adam(reg.parameters(), 1e-5)
optimizerS = torch.optim.Adam(superres.parameters(), 1e-5)

max_epochs = 5
for epoch in range(max_epochs):
    
    image = superres(stacks.detach())
    
    valid_loss = 0
    valid_step = 0
    for sliceno in tqdm(range(int(64/2))):
            valid_step += 1

            valid_moving = image.detach().to(device)
            batch_size = stacks.shape[0]

            valid_fixed = torch.zeros(batch_size, 1, 64, 64, 64).to(device)
            slice_ind = torch.tensor(range(2*sliceno, 2*(sliceno+1)))

            for s in range(batch_size):
                for d in range(6):
                    if int(d/2)==0:
                        valid_fixed[s, :, slice_ind, :, :] = stacks[s, d, slice_ind, :, :].to(
                        device)

                    elif int(d/2)==1:
                        valid_fixed[s, :, :, slice_ind, :] = stacks[s, d, :, slice_ind, :].to(
                            device)
                    elif int(d/2)==2:
                        valid_fixed[s, :, :, :, slice_ind] = stacks[s, d, :, :, slice_ind].to(
                            device)

            new_loss = image_loss(valid_moving, valid_fixed).item()
            loss = new_loss + 1
            n=1
            while new_loss < loss:
                optimizerR.zero_grad()

                ddf = reg(torch.cat((valid_moving, valid_fixed), dim=1))
                valid_out_image = warp_layer(valid_moving, ddf)

                loss = new_loss
                new_loss = image_loss(valid_out_image, valid_fixed).item()
                new_loss_TBU = image_loss(valid_out_image, valid_fixed)
                new_loss_TBU.backward()
                optimizerR.step()

                n+=1

                valid_moving = valid_out_image.detach()



            valid_loss += loss
            
            
    valid_loss = 0
    valid_step = 0       
    for sliceno in tqdm(range(int(64/2))):
            optimizerS.zero_grad()
            image = superres(stacks.detach())
            
            valid_step += 1

            valid_moving = image.to(device)
            batch_size = stacks.shape[0]

            valid_fixed = torch.zeros(batch_size, 1, 64, 64, 64).to(device)
            slice_ind = torch.tensor(range(2*sliceno, 2*(sliceno+1)))

            for s in range(batch_size):
                for d in range(6):
                    if int(d/2)==0:
                        valid_fixed[s, :, slice_ind, :, :] = stacks[s, d, slice_ind, :, :].to(
                        device)

                    elif int(d/2)==1:
                        valid_fixed[s, :, :, slice_ind, :] = stacks[s, d, :, slice_ind, :].to(
                            device)
                    elif int(d/2)==2:
                        valid_fixed[s, :, :, :, slice_ind] = stacks[s, d, :, :, slice_ind].to(
                            device)

            ddf = reg(torch.cat((valid_moving, valid_fixed), dim=1))
            output = warp_layer(valid_moving, ddf)
            superres_loss = image_loss(output, valid_fixed)
            
            superres_loss.backward()
            optimizerS.step()
    

