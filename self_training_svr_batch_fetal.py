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
from easy_transforms import RandMakeStackd, stacks2batchvol


set_determinism(42)




sublist_full = glob('./feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')

#sublist_full = glob('/home/ajoshi/T1w*.nii.gz')


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

#resize_x = Compose([EnsureChannelFirstD(keys=["image"]), Resized(keys=["image"],size_mode='all',spatial_size=[32, 64, 64])]) 
#                    Resized(keys=["image"],spatial_size=[64, 64, 64])])
#resize_x = Resized(keys=["image"],size_mode='all',spatial_size=[32, 64, 64])

resize_x_down = Resize(spatial_size=[32, 64, 64])
resize_y_down = Resize(spatial_size=[64, 32, 64])
resize_z_down = Resize(spatial_size=[64, 64, 32])
resize_up = Resize(spatial_size=[64, 64, 64])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_data = first(train_loader)
image = batch_data["image"].to(device)+.001
stacks = batch_data["stacks"].to(device)+.001

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


reg.load_state_dict(torch.load('/home/ajoshi/Desktop/epoch_3980.pth'));
#reg.load_state_dict(torch.load('/project/ajoshi_27/code_farm/deepSVR/model_64_slice2vol_reg/epoch_5370.pth'))
reg.train()
superres.load_state_dict(torch.load('/home/ajoshi/epoch_2020.pth'))
#superres.load_state_dict(torch.load('/project/ajoshi_27/code_farm/deepSVR/model_64_unet_large_lrem4_hcp_easy/epoch_2600.pth'));
superres.train()
optimizerR = torch.optim.Adam(reg.parameters(), 1e-6)
optimizerS = torch.optim.Adam(superres.parameters(), 1e-6)


recon_image = superres(stacks)
write_nifti(recon_image[0,0],f'outsvr_fast_fetal/deepsvr_recon_orig.nii.gz')

write_nifti(image[0,0],'outsvr_fast_fetal/deepsvr_orig.nii.gz')
write_nifti(stacks[0,0],'outsvr_fast_fetal/deepsvr_stack0.nii.gz')
write_nifti(stacks[0,1],'outsvr_fast_fetal/deepsvr_stack1.nii.gz')
write_nifti(stacks[0,2],'outsvr_fast_fetal/deepsvr_stack2.nii.gz')
write_nifti(stacks[0,3],'outsvr_fast_fetal/deepsvr_stack3.nii.gz')
write_nifti(stacks[0,4],'outsvr_fast_fetal/deepsvr_stack4.nii.gz')
write_nifti(stacks[0,5],'outsvr_fast_fetal/deepsvr_stack5.nii.gz')

stacks.to(device)

slice_vols_batch_x1 = stacks2batchvol(stacks[0,0],dir=0,device=device)
slice_vols_batch_x2 = stacks2batchvol(stacks[0,1],dir=0,device=device)
slice_vols_batch_y1 = stacks2batchvol(stacks[0,2],dir=1,device=device)
slice_vols_batch_y2 = stacks2batchvol(stacks[0,3],dir=1,device=device)
slice_vols_batch_z1 = stacks2batchvol(stacks[0,4],dir=2,device=device)
slice_vols_batch_z2 = stacks2batchvol(stacks[0,5],dir=2,device=device)

slice_vols_batch = torch.cat((slice_vols_batch_x1,slice_vols_batch_x2,slice_vols_batch_y1,slice_vols_batch_y2,slice_vols_batch_z1,slice_vols_batch_z2),dim=0)
del slice_vols_batch_z1, slice_vols_batch_x1, slice_vols_batch_y1
del slice_vols_batch_z2, slice_vols_batch_x2, slice_vols_batch_y2

num_slices = slice_vols_batch.shape[0]

sub_batch_size = 12
max_epochs = 5000000
for epoch in range(max_epochs):

    vol_loss = 0
    #optimizerS.zero_grad()
    #optimizerR.zero_grad()
    optimizerS.zero_grad()

    
    for i in range(0,192,sub_batch_size):
        optimizerR.zero_grad()

        recon_image = superres(stacks)
        recon_image = recon_image.repeat([sub_batch_size,1,1,1,1])

        s = slice_vols_batch[i:i+sub_batch_size,:,:,:,:]
        input_data = torch.cat((recon_image, s), dim=1)
        ddf = reg(input_data)
        recon_image_moved = warp_layer(recon_image, ddf)



        #for i in range(sub_batch_size):
        #    recon_image_moved[i] = resize_up(resize_x_down(recon_image_moved[i]))

        #recon_image_moved = [resize_x_down(resize_up(item)) for item in recon_image_moved]

        #recon_image_moved = [Resize(spatial_size=[64,64,64])(Resize(spatial_size=[32,64,64])(item)) for item in recon_image_moved]

        #recon_image_moved_dict = [{"image": item} for item in recon_image_moved[:,:,:,:,:]]
        #recon_image_moved_dict = [dict(zip("image", item)) for item in recon_image_moved[:,:,:,:,:]]

        #temp2 = resize_x(recon_image_moved_dict)
        #temp2 = resize_up(temp)
        
        slice_loss = image_loss(recon_image_moved, s)

        slice_loss.backward()
        optimizerR.step()

        vol_loss += slice_loss
    optimizerS.step()

    print(f'epoch_loss:{vol_loss} for epoch:{epoch}')    

    if np.mod(epoch, 100) == 0:

        torch.save(reg.state_dict(),'./outsvr_fast_fetal/epoch_reg_batch_'+str(epoch)+'.pth')
        torch.save(superres.state_dict(),'./outsvr_fast_fetal/epoch_superres_batch_'+str(epoch)+'.pth')
        write_nifti(recon_image[0,0],f'outsvr_fast_fetal/deepsvr_recon_batch_{epoch}.nii.gz')

   
    

