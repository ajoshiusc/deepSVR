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
image = check_data["image"][0][0]
stack = check_data["stacks"][0][4]

print(f"image shape: {image.shape}")
print(f"stack shape: {stack.shape}")

plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 32], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("stack")
plt.imshow(stack[:, :, 32], cmap="gray")
plt.savefig('sample_data.png')

plt.show()

train_ds = CacheDataset(data=training_datadict, transform=train_transforms,
                        cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = unet.UNet(
    spatial_dims=3,
    in_channels=6,  # moving and fixed
    out_channels=1,
    channels=(16, 32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2, 2),
    kernel_size = 5,
    up_kernel_size= 5,
    num_res_units=3).to(device)
image_loss = MSELoss()

optimizer = torch.optim.Adam(model.parameters(), 1e-4)

max_epochs = 5000
epoch_loss_values = []


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
                   './model_64_unet_large_lrem4/epoch_'+str(epoch)+'.pth')

    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

plt.plot(epoch_loss_values)
plt.savefig('epochs1em4.png')

plt.show()

print("done")
