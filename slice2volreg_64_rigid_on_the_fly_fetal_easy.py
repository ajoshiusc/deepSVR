from monai.utils import set_determinism, first
from monai.transforms import (
    EnsureChannelFirstD,
    Compose, ConcatItemsd,
    LoadImageD, CopyItemsd,
    RandRotateD, RandAffined,
    RandZoomD, CropForegroundd,
    GaussianSmoothd, Resize,
    ScaleIntensityRangePercentilesd, Resized
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.config import print_config, USE_COMPILED
from network_mods import GlobalNetRigid
from monai.networks.blocks import Warp
from monai.apps import MedNISTDataset
from easy_transforms import RandMakeStackd
slice_thickness = 2# 4 for hard transforms, 2 for easy transforms

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import MSELoss, CrossEntropyLoss
#import matplotlib.pyplot as plt
import os
import tempfile
from glob import glob
from monai.data.nifti_writer import write_nifti



def masked_mse(moving, slicevol):
    msk = slicevol > 0
    err = (moving-slicevol)*msk
    loss = torch.sqrt(torch.sum(err**2))
    return loss


print_config()
#directory = os.environ.get("MONAI_DATA_DIRECTORY")


sublist_full = glob('./feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')


PRETRAINED = False

if PRETRAINED:
    start_epoch = 2580
else:
    start_epoch = 0

trained_model_file = './model_64_slice2vol_reg_fetal/epoch_'+str(start_epoch)+'.pth'


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

        CopyItemsd(keys=["image","image","image"], names=["stackx", "stacky", "stackz"]),
        RandMakeStackd(keys=["stackx"], stack_axis=0),RandMakeStackd(keys=["stacky"], stack_axis=1),RandMakeStackd(keys=["stackz"], stack_axis=2),
        Resized(keys=["stackx","stacky","stackz"], spatial_size=[64, 64, 64]),

        #ConcatItemsd(keys=["stack0", "stack1", "stack2", "stack3", "stack4", "stack5"], name='stacks'),
        # Resized(keys=["image", "stack0", "stack1", "stack2","stack3","stack4","stack5"],spatial_size=[32,32,32]),
    ]
)


resize_x_down = Resize(spatial_size=[int(64/slice_thickness), 64, 64])
resize_y_down = Resize(spatial_size=[64, int(64/slice_thickness), 64])
resize_z_down = Resize(spatial_size=[64, 64, int(64/slice_thickness)])
resize_up = Resize(spatial_size=[64, 64, 64])


check_ds = Dataset(data=training_datadict, transform=randstack_transforms)
check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
check_data = first(check_loader)
image = check_data["image"][0][0]
stack = check_data["stackx"][0][0]

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

device = torch.device("cuda:0")
model = GlobalNetRigid(
    image_size=(64, 64, 64),
    spatial_dims=3,
    in_channels=2,  # moving and fixed
    num_channel_initial=16,
    depth=4).to(device)
image_loss = masked_mse  # (image,slicevol)MSELoss()
if USE_COMPILED:
    warp_layer = Warp(3, "border").to(device)
else:
    warp_layer = Warp("bilinear", "border").to(device)

optimizer = torch.optim.Adam(model.parameters(), 1e-7)

max_epochs = 50000
epoch_loss_values = []
epoch_loss_valid = []

if PRETRAINED:
    model.load_state_dict(torch.load(trained_model_file))

for epoch in range(start_epoch+1, max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss, step = 0, 0
    for batch_data in tqdm(train_loader):
        for sliceno in range(int(64/slice_thickness)):

            step += 1
            optimizer.zero_grad()

            image_vol = batch_data["image"].to(device)
            batch_size = image_vol.shape[0]

            slice_vol = torch.zeros(batch_size, 3, 64, 64, 64)
            slice_ind = torch.tensor(range(slice_thickness*sliceno, slice_thickness*(sliceno+1))) # 4 for hard transform

            #for s in range(batch_size):
            slice_vol[:, 0, slice_ind, :, :] = batch_data['stackx'][:, 0, slice_ind, :, :]
            slice_vol[:, 1, :, slice_ind, :] = batch_data['stacky'][:, 0, :, slice_ind, :]
            slice_vol[:, 2, :, :, slice_ind] = batch_data['stackz'][:, 0, :, :, slice_ind]

            slice_vol = slice_vol.to(device)

            ddf = model(torch.cat((image_vol, slice_vol[:,0:1]), dim=1))
            pred_imagex = warp_layer(image_vol, ddf)
            temp = resize_x_down(pred_imagex[0])
            temp2 = resize_up(temp)
            slice_loss = image_loss(temp2, slice_vol[:,0:1])
            slice_loss.backward()
            optimizer.step()
            epoch_loss += slice_loss.item()

            ddf = model(torch.cat((image_vol, slice_vol[:,1:2]), dim=1))
            pred_imagey = warp_layer(image_vol, ddf)
            temp = resize_y_down(pred_imagey[0])
            temp2 = resize_up(temp)
            slice_loss = image_loss(temp2, slice_vol[:,1:2])
            slice_loss.backward()
            optimizer.step()
            epoch_loss += slice_loss.item()

            ddf = model(torch.cat((image_vol, slice_vol[:,2:3]), dim=1))
            pred_imagez = warp_layer(image_vol, ddf)
            temp = resize_z_down(pred_imagez[0])
            temp2 = resize_up(temp)
            slice_loss = image_loss(temp2, slice_vol[:,2:3])
            slice_loss.backward()
            optimizer.step()
            epoch_loss += slice_loss.item()

        #print(f"{step}/{len(train_ds) // train_loader.batch_size}, "f"train_loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    if np.mod(epoch, 10) == 0:

        torch.save(model.state_dict(),
                   './model_64_slice2vol_reg_fetal_easy/epoch_'+str(epoch)+'.pth')
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    np.savez('svr_reg_epoch_loss_values_on_the_fly_fetal_easy.npz',
             epoch_loss_values=epoch_loss_values, epoch_loss_valid=epoch_loss_valid)


print('Done!')
