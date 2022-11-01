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
#import matplotlib.pyplot as plt
import os
import tempfile
from glob import glob


print_config()
set_determinism(42)

#directory = os.environ.get("MONAI_DATA_DIRECTORY")

'''directory = "./monai_data_dir"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
'''


#train_data = MedNISTDataset(root_dir=root_dir, section="training", download=True, transform=None)

image_files = glob('/project/ajoshi_27/feta_syn_data_slices/sub-*T2w/*_T2w_*.nii.gz')

'''training_datadict = [
    {"fixed_hand": item["image"], "moving_hand": item["image"]}
    for item in train_data.data if item["label"] == 4  # label 4 is for xray hands
]'''

training_datadict = [
    {"fixed_hand": item, "moving_hand": item,"fixed_orig": item, "moving_orig": item}
    for item in image_files  # label 4 is for xray hands
]    #for item in train_data.data if item["label"] == 4  # label 4 is for xray hands


print("\n first training items: ", training_datadict[:3])


train_transforms = Compose(
    [
        LoadImageD(keys=["fixed_hand", "moving_hand", "fixed_orig", "moving_orig"]),
        EnsureChannelFirstD(keys=["fixed_hand", "moving_hand", "fixed_orig", "moving_orig"]),
        Resized(keys=["fixed_hand", "moving_hand", "fixed_orig", "moving_orig"],spatial_size=[32, 32, 32]),
        GaussianSmoothd(keys=["fixed_hand", "moving_hand"],sigma=2),
        ScaleIntensityRangePercentilesd(keys=["fixed_hand", "moving_hand", "fixed_orig", "moving_orig"],lower=0,upper=100,b_min=0.0, b_max=1.0, clip=True),
        #ScaleIntensityRanged(keys=["fixed_hand", "moving_hand"], a_min=0., a_max=850, b_min=0.0, b_max=1.0, clip=True,),
        RandRotateD(keys=["moving_hand", "moving_orig"], range_x=np.pi/4, range_y=np.pi/4, range_z=np.pi/4, prob=1.0, keep_size=True,padding_mode='border'),#, mode="bicubic")
        RandZoomD(keys=["moving_hand","moving_orig"], min_zoom=0.9, max_zoom=1.1, prob=1.0),

    ]
)

check_ds = Dataset(data=training_datadict, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
check_data = first(check_loader)
fixed_image = check_data["fixed_hand"][0][0]
moving_image = check_data["moving_hand"][0][0]

print(f"moving_image shape: {moving_image.shape}")
print(f"fixed_image shape: {fixed_image.shape}")

'''plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("moving_image")
plt.imshow(moving_image[:,:,16], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("fixed_image")
plt.imshow(fixed_image[:,:,16], cmap="gray")

plt.show()
'''
train_ds = CacheDataset(data=training_datadict[:10000], transform=train_transforms,
                        cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)


device = torch.device("cuda:0")
model = GlobalNet(
    image_size=(32, 32, 32),
    spatial_dims=3,
    in_channels=2,  # moving and fixed
    num_channel_initial=4,
    depth=4).to(device)
image_loss = MSELoss()
if USE_COMPILED:
    warp_layer = Warp(3, "border").to(device)
else:
    warp_layer = Warp("bilinear", "border").to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

max_epochs = 50
epoch_loss_values = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss, step = 0, 0
    for batch_data in train_loader:
        step += 1
        optimizer.zero_grad()

        moving = batch_data["moving_hand"].to(device)
        fixed = batch_data["fixed_hand"].to(device)
        ddf = model(torch.cat((moving, fixed), dim=1))
        pred_image = warp_layer(moving, ddf)

        loss = image_loss(pred_image, fixed)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #print(f"{step}/{len(train_ds) // train_loader.batch_size}, "f"train_loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    torch.save(model.state_dict(), "best_metric_model_"+str(epoch)+".pth")

    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
'''  
plt.plot(epoch_loss_values)
'''

val_ds = CacheDataset(data=training_datadict[2110:2150], transform=train_transforms,
                      cache_rate=1.0, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, num_workers=0)
for batch_data in val_loader:
    moving = batch_data["moving_hand"].to(device)
    fixed_orig = batch_data["fixed_orig"].to(device)
    moving_orig = batch_data["moving_orig"].to(device)
    fixed = batch_data["fixed_hand"].to(device)
    ddf = model(torch.cat((moving, fixed), dim=1))
    pred_image = warp_layer(moving_orig, ddf)
    break

fixed_image = fixed_orig.detach().cpu().numpy()[:, 0]
moving_image = moving_orig.detach().cpu().numpy()[:, 0]
pred_image = pred_image.detach().cpu().numpy()[:, 0]

batch_size = 5
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







