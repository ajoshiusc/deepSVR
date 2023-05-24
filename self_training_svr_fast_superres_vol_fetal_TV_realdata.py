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


sublist_full = glob('./feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')

#sublist_full = glob('/home/ajoshi/T1w*.nii.gz')


# training files
#x[1,4,5,8]
#y[3,7,9,11]
#z[0,2,6,10]
subfiles_train = [f'rstack{i}.nii.gz' for i in (1,4,3,7,0,2)]


training_datadict = [{f"image": item} for item in subfiles_train]




randstack_transforms_old = Compose(
    [
        LoadImageD(keys=["image"]),
        EnsureChannelFirstD(keys=["image"]),
        CopyItemsd(keys=["image", "image", "image", "image", "image", "image"], names=[
                   "stack0", "stack1", "stack2", "stack3", "stack4", "stack5"]),

        ConcatItemsd(keys=["stack0", "stack1", "stack2",
                     "stack3", "stack4", "stack5"], name='stacks'),
        # Resized(keys=["image", "stack0", "stack1", "stack2","stack3","stack4","stack5"],spatial_size=[32,32,32]),
    ]
)



randstack_transforms = Compose(
    [
        LoadImageD(keys=["image"]),
        EnsureChannelFirstD(keys=["image"]),
        Resized(keys=["image"], spatial_size=[64, 64, 64]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=2, upper=98, b_min=0.0, _max=10.0, clip=True),
        CopyItemsd(keys=["image"], names=["stack0"]),
        #LoadImageD(keys=["image"]),
        #EnsureChannelFirstD(keys=["image"]),
        #CopyItemsd(keys=["image"], names=["stack1"]),

        #ConcatItemsd(keys=["stack0", "stack1"], name='stacks'),
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

resize_x_down = Resize(spatial_size=[32, 64, 64])
resize_y_down = Resize(spatial_size=[64, 32, 64])
resize_z_down = Resize(spatial_size=[64, 64, 32])
resize_up = Resize(spatial_size=[64, 64, 64])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_data = first(train_loader)
image = batch_data["image"].to(device)+.001
stacks = batch_data["image"].to(device)+.001
stacks = torch.swapaxes(stacks,0,1)
#stacks = batch_data["stacks"].to(device)+.001

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
    
def tv_loss(img):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (B, C, D, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.abs(img[:,:,:,:,:-1] - img[:,:,:,:,1:]))
    h_variance = torch.sum(torch.abs(img[:,:,:,:-1,:] - img[:,:,:,1:,:]))
    d_variance = torch.sum(torch.abs(img[:,:,:-1,:,:] - img[:,:,1:,:,:]))
    loss =  (h_variance + w_variance + d_variance)
    return loss


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

#reg.load_state_dict(torch.load('/ImagePTE1/ajoshi/code_farm/deepSVR/trained_models/reg/epoch_3980.pth'));
superres.load_state_dict(torch.load('/home/ajoshi/projects/deepSVR/trained_models/sup/epoch_2600.pth'));
#superres.load_state_dict(torch.load('/ImagePTE1/ajoshi/code_farm/deepSVR/trained_models/epoch_2020.pth'))

reg.load_state_dict(torch.load('/home/ajoshi/projects/deepSVR/trained_models/reg/epoch_5370.pth'));
#reg.load_state_dict(torch.load('/project/ajoshi_27/code_farm/deepSVR/model_64_slice2vol_reg/epoch_5370.pth'))
reg.train()
#superres.load_state_dict(torch.load('/home/ajoshi/epoch_2020.pth'))
#superres.load_state_dict(torch.load('/project/ajoshi_27/code_farm/deepSVR/model_64_unet_large_lrem4_hcp_easy/epoch_2600.pth'));
superres.train()
recon_image = superres(stacks)

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, img):
        super(Model, self).__init__()
        self.recon_img = nn.Parameter(img)
    
    def forward(self):
        return self.recon_img
    

superres = Model(recon_image)

optimizerR = torch.optim.Adam(reg.parameters(), 1e-6)
optimizerS = torch.optim.Adam(superres.parameters(), 1e-1)





write_nifti(recon_image[0,0],f'outsvr_fast_fetal_real_data/deepsvr_recon_orig.nii.gz')

write_nifti(image[0,0],'outsvr_fast_fetal_real_data/deepsvr_orig.nii.gz')
write_nifti(stacks[0,0],'outsvr_fast_fetal_real_data/deepsvr_stack0.nii.gz')
write_nifti(stacks[0,1],'outsvr_fast_fetal_real_data/deepsvr_stack1.nii.gz')
write_nifti(stacks[0,2],'outsvr_fast_fetal_real_data/deepsvr_stack2.nii.gz')
write_nifti(stacks[0,3],'outsvr_fast_fetal_real_data/deepsvr_stack3.nii.gz')
write_nifti(stacks[0,4],'outsvr_fast_fetal_real_data/deepsvr_stack4.nii.gz')
write_nifti(stacks[0,5],'outsvr_fast_fetal_real_data/deepsvr_stack5.nii.gz')

stacks.to(device)

tv_weight=.01
max_epochs = 5000000
for epoch in range(max_epochs):

    vol_loss = 0
    optimizerS.zero_grad()
    recon_image = superres()

    for d in tqdm(range(6)):

        for sliceno in range(int(64/2)):

            #valid_moving = image.detach().to(device)
            batch_size = stacks.shape[0]
            slice_ind = torch.tensor(range(2*sliceno, 2*(sliceno+1)))


            optimizerR.zero_grad()

            slice_vol = torch.zeros(batch_size, 1, 64, 64, 64,device=device)#.to(device)

            if int(d/2)==0:
                slice_vol[0, :, slice_ind, :, :] = stacks[0, d, slice_ind, :, :]#stacks[0, d, :, :, slice_ind].swapaxes(-1,-3) #
            elif int(d/2)==1:
                slice_vol[0, :, :, slice_ind, :] = stacks[0, d, :, :, slice_ind, :]#stacks[0, d, :, :, slice_ind].swapaxes(-1,-2) #
            elif int(d/2)==2:
                slice_vol[0, :, :, :, slice_ind] = stacks[0, d, :, :, slice_ind] #.to(device)


            ddf = reg(torch.cat((recon_image, slice_vol), dim=1))
            recon_image_moved = warp_layer(recon_image, ddf)

            if int(d/2)==0:
                temp = resize_x_down(recon_image_moved[0])
                temp2 = resize_up(temp)
            elif int(d/2)==1:
                temp = resize_y_down(recon_image_moved[0])
                temp2 = resize_up(temp)
            elif int(d/2)==2:
                temp = resize_z_down(recon_image_moved[0])
                temp2 = resize_up(temp)

            slice_loss = image_loss(temp2, slice_vol[0]) 
            vol_loss += slice_loss

            slice_loss.backward()
            optimizerR.step()


    tv = tv_weight*tv_loss(recon_image)
    tv.backward()
    #vol_loss = vol_loss + tv

    optimizerS.step()

    if np.mod(epoch, 10) == 0:



        write_nifti(recon_image[0,0],f'outsvr_fast_fetal_real_data/deepsvr_recon_{epoch}_l1.nii.gz')

    print(f'epoch_loss:{vol_loss} for epoch:{epoch}')    
   
    

