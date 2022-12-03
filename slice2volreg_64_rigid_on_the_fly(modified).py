from monai.utils import set_determinism, first
from monai.transforms import (
    EnsureChannelFirstD,
    Compose, ConcatItemsd,
    LoadImageD, CopyItemsd,
    RandRotateD, RandAffined,
    RandZoomD, CropForegroundd,
    GaussianSmoothd, RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd, Resized
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.config import print_config, USE_COMPILED
from network_mods import GlobalNetRigid
from monai.networks.blocks import Warp
from monai.apps import MedNISTDataset
from transforms import RandMakeStackd
from tqdm.notebook import tqdm
import numpy as np
import torch
from torch.nn import MSELoss, CrossEntropyLoss
#import matplotlib.pyplot as plt
import os
import tempfile
from glob import glob
from monai.data.nifti_writer import write_nifti
device = torch.device("cuda:0")

model = GlobalNetRigid(
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

sublist_full = glob('/project/ajoshi_27/code_farm/deepSVR/feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')

# training files

subfiles_train = sublist_full[:60]
subfiles_val = sublist_full[60:70]
subfiles_test = sublist_full[70:]

# '/deneb_disk/feta_2022/feta_2.2/sub-029/anat/sub-029_rec-mial_T2w.nii.gz'

training_datadict = [{"image": item} for item in subfiles_train]
valid_datadict = [{"image": item} for item in subfiles_val]

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


check_ds = Dataset(data=training_datadict, transform=randstack_transforms)
check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
check_data = first(check_loader)
image = check_data["image"][0][0]
stack = check_data["stack"][0][0]

print(f"image shape: {image.shape}")
print(f"stack shape: {stack.shape}")

""" plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 32], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("stack")
plt.imshow(stack[:, :, 32], cmap="gray")
plt.savefig('sample_data.png')
plt.show()
 """
train_ds = CacheDataset(data=training_datadict, transform=randstack_transforms,
                        cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)


valid_ds = CacheDataset(data=valid_datadict, transform=randstack_transforms,
                        cache_rate=1.0, num_workers=4)
valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=True, num_workers=2)



for valid_batch_data in tqdm(valid_loader):
    valid_loss = 0
    valid_step = 0
    for sliceno in tqdm(range(int(64/4))):
                    valid_step += 1

                    valid_moving = valid_batch_data["image"].to(device)
                    batch_size = valid_batch_data['stack'].shape[0]

                    valid_fixed = torch.zeros(batch_size, 1, 64, 64, 64).to(device)
                    slice_ind = torch.tensor(range(4*sliceno, 4*(sliceno+1)))

                    for s in range(batch_size):
                        dir = 0  # batch_data['dir'][s]
                        if dir == 0:
                            valid_fixed[s, :, slice_ind, :, :] = valid_batch_data['stack'][s, :, slice_ind, :, :].to(
                                device)
                        elif dir == 1:
                            valid_fixed[s, :, :, slice_ind, :] = valid_batch_data['stack'][s, :, :, slice_ind, :].to(
                                device)
                        elif dir == 2:
                            valid_fixed[s, :, :, :, slice_ind] = valid_batch_data['stack'][s, :, :, :, slice_ind].to(
                                device)

                    model.load_state_dict(torch.load('/project/ajoshi_27/code_farm/deepSVR/model_64_slice2vol_reg/epoch_3980.pth'));
                    model.train()

                    new_loss = image_loss(valid_moving, valid_fixed).item()
                    loss = new_loss + 1
                    n=1
                    while new_loss < loss:
                        optimizer.zero_grad()

                        ddf = model(torch.cat((valid_moving, valid_fixed), dim=1))
                        valid_out_image = warp_layer(valid_moving, ddf)

                        loss = new_loss
                        new_loss = image_loss(valid_out_image, valid_fixed).item()
                        new_loss_TBU = image_loss(valid_out_image, valid_fixed)
                        new_loss_TBU.backward()
                        optimizer.step()

                        n+=1

                        valid_moving = valid_out_image.detach()



                    valid_loss += loss

    valid_loss /= valid_step                
    print(valid_loss)
    print(n)