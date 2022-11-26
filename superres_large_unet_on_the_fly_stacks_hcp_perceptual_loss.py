from monai.utils import set_determinism, first
from monai.transforms import (
    EnsureChannelFirstD,
    Compose, ConcatItemsd,
    LoadImageD,
    RandRotateD,
    RandZoomD,
    GaussianSmoothd, RandGaussianSmoothd, CopyItemsd, CropForegroundd,
    ScaleIntensityRangePercentilesd, Resized, RandAffined
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
from vgg_perceptual_loss import VGGPerceptualLoss


print_config()
# set_determinism(42)

#sublist_full = glob('/project/ajoshi_27/HCP_All/*/T1w/T1*.nii.gz')
sublist_full = glob('./feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')

percp_loss = VGGPerceptualLoss()
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
image = check_data["image"][0][0]
stack = check_data["stack0"][0][0]

print(f"image shape: {image.shape}")
print(f"stack shape: {stack.shape}")

'''plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 32], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("stack0")
plt.imshow(stack[:, :, 32], cmap="gray")
plt.savefig('sample_data.png')

plt.show()
'''

train_ds = CacheDataset(data=training_datadict, transform=randstack_transforms,
                        cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)


valid_ds = CacheDataset(data=valid_datadict, transform=randstack_transforms,
                        cache_rate=1.0, num_workers=4)
valid_loader = DataLoader(valid_ds, batch_size=10, shuffle=True, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = unet.UNet(
    spatial_dims=3,
    in_channels=6,  # moving and fixed
    out_channels=1,
    channels=(16, 32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2, 2),
    kernel_size=5,
    up_kernel_size=5,
    num_res_units=3).to(device)
image_loss = percp_loss

optimizer = torch.optim.Adam(model.parameters(), 1e-4)

max_epochs = 5000
epoch_loss_values = []
epoch_loss_valid = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss, step = 0, 0
    for batch_data in train_loader:
        step += 1
        optimizer.zero_grad()

        image = batch_data["image"].to(device)
        stacks = batch_data["stacks"].to(device)

        out_image = model(stacks)

        loss = image_loss(image, out_image)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #print(f"{step}/{len(train_ds) // train_loader.batch_size}, "f"train_loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    if np.mod(epoch, 10) == 0:
        torch.save(model.state_dict(),
                   './model_64_unet_large_lrem4_hcp/epoch_'+str(epoch)+'.pth')

        # run validation

        valid_step = 0
        for valid_batch_data in valid_loader:
            valid_step += 1
            valid_loss = 0
            valid_image = valid_batch_data["image"].to(device)
            valid_stacks = valid_batch_data["stacks"].to(device)

            model.eval()
            with torch.no_grad():
                valid_out_image = model(valid_stacks)

            valid_loss += image_loss(valid_image, valid_out_image).item()

        valid_loss /= valid_step
        epoch_loss_valid.append(valid_loss)

        print(f"validation loss: {valid_loss:.4f}")

    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    np.savez('large_unet_on_the_fly_loss_values_hcp_perceptual_loss.npz',
             epoch_loss_values=epoch_loss_values, epoch_loss_valid=epoch_loss_valid)
'''plt.plot(epoch_loss_values)
plt.savefig('epochs1em4.png')

plt.show()
'''
print("done")
