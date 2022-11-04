from monai.utils import set_determinism, first
from monai.transforms import (
    EnsureChannelFirstD,
    Compose,
    LoadImageD,
    RandRotateD,
    RandZoomD,
    GaussianSmoothd,RandGaussianSmoothd,
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
    "stack1": item[:-13]+'_stack_x_1.nii.gz',"stack2": item[:-13]+'_stack_y_0.nii.gz',
    "stack3": item[:-13]+'_stack_y_1.nii.gz',"stack4": item[:-13]+'_stack_z_0.nii.gz',
    "stack5": item[:-13]+'_stack_z_1.nii.gz'} for item in image_files]


#training_datadict = d0 + d1 + d2 + d3 + d4 + d5

#print("\n first training items: ", training_datadict)


train_transforms = Compose(
    [
        LoadImageD(keys=["image", "stack0", "stack1", "stack2","stack3","stack4","stack5"]),
        EnsureChannelFirstD(keys=["image", "stack0", "stack1", "stack2","stack3","stack4","stack5"]),
        #Resized(keys=["image", "stack0", "stack1", "stack2","stack3","stack4","stack5"],spatial_size=[32,32,32]),
        #ScaleIntensityRangePercentilesd(keys=["image", "stack"],lower=0,upper=100,b_min=0.0, b_max=1.0, clip=True),

    ]
)

check_ds = Dataset(data=training_datadict, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
check_data = first(check_loader)
image = check_data["image"][0][0]
stack0 = check_data["stack0"][0][0]

print(f"image shape: {image.shape}")
print(f"stack0 shape: {stack0.shape}")

plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:,:,32], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("stack0")
plt.imshow(stack0[:,:,32], cmap="gray")

plt.show()

train_ds = CacheDataset(data=training_datadict, transform=train_transforms,
                        cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)


device = torch.device("cuda:0")
model = GlobalNetRigid(
    image_size=(32, 32, 32),
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


for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss, step = 0, 0
    for batch_data in train_loader:
        for sliceno in range(int(32/2)):
            step += 1
            optimizer.zero_grad()

            moving = batch_data["image"].to(device)
            batch_size = moving.shape[0]

            fixed = torch.zeros(batch_size,1,32,32,32)
            slice_ind = torch.tensor(range(2*(sliceno),2*sliceno+1))
 
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

    torch.save(model.state_dict(), './model_32_slice2vol/epoch_'+str(epoch)+'.pth')

    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
plt.plot(epoch_loss_values)
