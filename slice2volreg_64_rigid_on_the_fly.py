from monai.utils import set_determinism, first
from monai.transforms import (
    EnsureChannelFirstD,
    Compose,ConcatItemsd,
    LoadImageD,CopyItemsd,
    RandRotateD,RandAffined,
    RandZoomD,CropForegroundd,
    GaussianSmoothd,RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd, Resized
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.config import print_config, USE_COMPILED
from network_mods import GlobalNetRigid
from monai.networks.blocks import Warp
from monai.apps import MedNISTDataset

import numpy as np
import torch
from torch.nn import MSELoss, CrossEntropyLoss
#import matplotlib.pyplot as plt
import os
import tempfile
from glob import glob
from monai.data.nifti_writer import write_nifti

def masked_mse(moving,slicevol):
    msk = slicevol>0
    err = (moving-slicevol)*msk
    loss = torch.sqrt(torch.sum(err**2))
    return loss



print_config()
#directory = os.environ.get("MONAI_DATA_DIRECTORY")


sublist_full = glob(
    './feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')


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
        RandAffined(mode=("bilinear"), prob=1.0, translate_range=(.2, 1, 1), rotate_range=(
            np.pi / 16, np.pi / 32, np.pi / 32), padding_mode="border", keys=["stack0"]),
        RandAffined(mode=("bilinear"), prob=1.0, translate_range=(.2, 1, 1), rotate_range=(
            np.pi / 16, np.pi / 32, np.pi / 32), padding_mode="border", keys=["stack1"]),

        RandAffined(mode=("bilinear"), prob=1.0, translate_range=(1, .2, 1), rotate_range=(
            np.pi / 32, np.pi / 16, np.pi / 32), padding_mode="border", keys=["stack2"]),
        RandAffined(mode=("bilinear"), prob=1.0, translate_range=(1, .2, 1), rotate_range=(
            np.pi / 32, np.pi / 16, np.pi / 32), padding_mode="border", keys=["stack3"]),

        RandAffined(mode=("bilinear"), prob=1.0, translate_range=(1, 1, .2), rotate_range=(
            np.pi / 32, np.pi / 32, np.pi / 16), padding_mode="border", keys=["stack0"]),
        RandAffined(mode=("bilinear"), prob=1.0, translate_range=(1, 1, .2), rotate_range=(
            np.pi / 32, np.pi / 32, np.pi / 16), padding_mode="border", keys=["stack1"]),

        Resized(keys=["stack0"], spatial_size=[16, 64, 64]), Resized(
            keys=["stack0"], spatial_size=[64, 64, 64]),
        Resized(keys=["stack1"], spatial_size=[16, 64, 64]), Resized(
            keys=["stack1"], spatial_size=[64, 64, 64]),
        Resized(keys=["stack2"], spatial_size=[16, 64, 64]), Resized(
            keys=["stack2"], spatial_size=[64, 64, 64]),
        Resized(keys=["stack3"], spatial_size=[64, 16, 64]), Resized(
            keys=["stack3"], spatial_size=[64, 64, 64]),
        Resized(keys=["stack4"], spatial_size=[16, 64, 64]), Resized(
            keys=["stack4"], spatial_size=[64, 64, 64]),
        Resized(keys=["stack5"], spatial_size=[64, 64, 16]), Resized(
            keys=["stack5"], spatial_size=[64, 64, 64]),

        ConcatItemsd(keys=["stack0", "stack1", "stack2",
                     "stack3", "stack4", "stack5"], name='stacks'),
        # Resized(keys=["image", "stack0", "stack1", "stack2","stack3","stack4","stack5"],spatial_size=[32,32,32]),
    ]
)

check_ds = Dataset(data=training_datadict, transform=randstack_transforms)
check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
check_data = first(check_loader)
image = check_data["image"][0][0]
stack = check_data["stack0"][0][0]

print(f"image shape: {image.shape}")
print(f"stack shape: {stack.shape}")
'''
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 32], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("stack")
plt.imshow(stack[:, :, 32], cmap="gray")
plt.savefig('sample_data.png')

plt.show()
'''
train_ds = CacheDataset(data=training_datadict, transform=randstack_transforms,
                        cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)


valid_ds = CacheDataset(data=valid_datadict, transform=randstack_transforms,
                        cache_rate=1.0, num_workers=4)
valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=True, num_workers=2)

device = torch.device("cuda:0")
model = GlobalNetRigid(
    image_size=(64, 64, 64),
    spatial_dims=3,
    in_channels=2,  # moving and fixed
    num_channel_initial=16,
    depth=4).to(device)
image_loss = masked_mse #(image,slicevol)MSELoss()
if USE_COMPILED:
    warp_layer = Warp(3, "border").to(device)
else:
    warp_layer = Warp("bilinear", "border").to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-5)

max_epochs = 500
epoch_loss_values = []
epoch_loss_valid = []


for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss, step = 0, 0
    for batch_data in train_loader:
        for sliceno in range(int(64/4)):
            step += 1
            optimizer.zero_grad()

            moving = batch_data["image"].to(device)
            batch_size = moving.shape[0]

            fixed = torch.zeros(batch_size,1,64,64,64)
            slice_ind = torch.tensor(range(4*(sliceno),4*sliceno+1))
 
            for s in range(batch_size):
                dir = batch_data['dir'][s]
                if dir == 0:
                    fixed[s,:,slice_ind,:,:] = batch_data['stack'][s,:,slice_ind,:,:]
                elif dir == 1:
                    fixed[s,:,:,slice_ind,:] = batch_data['stack'][s,:,:,slice_ind,:]
                elif dir == 2:
                    fixed[s,:,:,:,slice_ind] = batch_data['stack'][s,:,:,:,slice_ind]

            fixed = fixed.to(device)

            ddf = model(torch.cat((moving, fixed), dim=1))
            pred_image = warp_layer(moving, ddf)

            loss = image_loss(pred_image,fixed)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        #print(f"{step}/{len(train_ds) // train_loader.batch_size}, "f"train_loss: {loss.item():.4f}")


    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    if np.mod(epoch, 10) == 0:
        torch.save(model.state_dict(), './model_64_slice2vol/epoch_'+str(epoch)+'.pth')
        valid_loss = 0
        for valid_batch_data in valid_loader:
            valid_image = valid_batch_data["image"].to(device)
            valid_stacks = valid_batch_data["stacks"].to(device)
            valid_out_image = model(valid_stacks)
            valid_loss += image_loss(valid_image, valid_out_image).item()
            epoch_loss_valid.append(valid_loss)

        print(f"validation loss: {valid_loss:.4f}")


    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

  
#plt.plot(epoch_loss_values)

batch_size=16
val_ds = CacheDataset(data=training_datadict[0:16], transform=train_transforms,
                      cache_rate=1.0, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, num_workers=0)
for batch_data in val_loader:
    moving = batch_data["image"].to(device)
    fixed_full = batch_data["stack"]
    dir = batch_data['dir']

    fixed = torch.zeros(batch_size,1,64,64,64)
    fixed[:,0,16:18,:,:]=fixed_full[:,0,16:18,:,:]
    fixed = fixed.to(device)
    ddf = model(torch.cat((moving, fixed), dim=1))
    pred_image = warp_layer(moving, ddf)
    break

fixed_image = fixed.detach().cpu().numpy()[:, 0]
moving_image = moving.detach().cpu().numpy()[:, 0]
pred_image = pred_image.detach().cpu().numpy()[:, 0]

for i in range(batch_size):
    write_nifti(fixed_image[i],'fixed'+str(i)+'.nii.gz')
    write_nifti(moving_image[i],'moving'+str(i)+'.nii.gz')
    write_nifti(pred_image[i],'pred'+str(i)+'.nii.gz')

#batch_size = 5
'''plt.subplots(batch_size, 3, figsize=(8, 10))
for b in range(batch_size):
    # moving image
    plt.subplot(batch_size, 3, b * 3 + 1)
    plt.axis('off')
    plt.title("moving image")
    plt.imshow(moving_image[b][:,:,16], cmap="gray")
    # fixed image
    plt.subplot(batch_size, 3, b * 3 + 2)
    plt.axis('off')
    plt.title("fixed image")
    plt.imshow(fixed_image[b][:,:,16], cmap="gray")
    # warped moving
    plt.subplot(batch_size, 3, b * 3 + 3)
    plt.axis('off')
    plt.title("predicted image")
    plt.imshow(pred_image[b][:,:,16], cmap="gray")
plt.axis('off')
plt.show()

'''
print('Done!')







