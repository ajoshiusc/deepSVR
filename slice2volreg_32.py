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
from monai.networks.nets import GlobalNet
from monai.networks.blocks import Warp
from monai.apps import MedNISTDataset

import numpy as np
import torch
from torch.nn import MSELoss, CrossEntropyLoss
import matplotlib.pyplot as plt
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
set_determinism(42)

#directory = os.environ.get("MONAI_DATA_DIRECTORY")

'''directory = "./monai_data_dir"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
'''


#train_data = MedNISTDataset(root_dir=root_dir, section="training", download=True, transform=None)

image_files = glob('./feta_syn_data_prealigned/sub-*_T2w/sub-*_T2w_image.nii.gz')



d0 = [{"image": item, "stack": item[:-13]+'_stack_x_0.nii.gz', "dir":0} for item in image_files]
d1 = [{"image": item, "stack": item[:-13]+'_stack_x_1.nii.gz', "dir":0} for item in image_files]
d2 = [{"image": item, "stack": item[:-13]+'_stack_y_0.nii.gz', "dir":1} for item in image_files]
d3 = [{"image": item, "stack": item[:-13]+'_stack_y_1.nii.gz', "dir":1} for item in image_files]
d4 = [{"image": item, "stack": item[:-13]+'_stack_z_0.nii.gz', "dir":2} for item in image_files]
d5 = [{"image": item, "stack": item[:-13]+'_stack_z_1.nii.gz', "dir":2} for item in image_files]

training_datadict = d0 + d1 + d2 + d3 + d4 + d5

print("\n first training items: ", training_datadict)


train_transforms = Compose(
    [
        LoadImageD(keys=["image", "stack"]),
        EnsureChannelFirstD(keys=["image", "stack"]),
        Resized(keys=["image","stack"],spatial_size=[32,32,32]),
        ScaleIntensityRangePercentilesd(keys=["image", "stack"],lower=0,upper=100,b_min=0.0, b_max=1.0, clip=True),

    ]
)

check_ds = Dataset(data=training_datadict, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
check_data = first(check_loader)
fixed_image = check_data["image"][0][0]
moving_image = check_data["stack"][0][0]
dir_image = check_data['dir'][0]
print(f"moving_image shape: {moving_image.shape}")
print(f"fixed_image shape: {fixed_image.shape}")
print(f"dir shape: {dir_image.shape}")

'''plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("moving_image")
plt.imshow(moving_image[:,:,16], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("fixed_image")
plt.imshow(fixed_image[:,:,16], cmap="gray")

plt.show()
'''
train_ds = CacheDataset(data=training_datadict, transform=train_transforms,
                        cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)


device = torch.device("cuda:0")
model = GlobalNet(
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
optimizer = torch.optim.Adam(model.parameters(), 1e-6)

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

batch_size=16
val_ds = CacheDataset(data=training_datadict[0:16], transform=train_transforms,
                      cache_rate=1.0, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, num_workers=0)
for batch_data in val_loader:
    moving = batch_data["image"].to(device)
    fixed_full = batch_data["stack"]
    dir = batch_data['dir']

    fixed = torch.zeros(batch_size,1,32,32,32)
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







