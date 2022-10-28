from monai.transforms import (
    LoadImage, LoadImaged, EnsureChannelFirstd,
    Resized, EnsureTyped, Compose,NormalizeIntensityd, RandAffined, RandAffine, SaveImage, Resize, EnsureChannelFirst, CropForeground
)

import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

output_dir = '.'
nii_name = '/deneb_disk/feta_2022/feta_2.2/sub-080/anat/sub-080_rec-irtk_T2w.nii.gz'

num_stacks = 3

rand_affine_x = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(.2, 1, 1), 
    rotate_range=(np.pi / 16, np.pi /32, np.pi / 32), 
    padding_mode="border")

rand_affine_y = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(1, .2, 1), 
    rotate_range=(np.pi / 32, np.pi /16, np.pi / 32), 
    padding_mode="border")


rand_affine_z = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(1, 1, .2), 
    rotate_range=(np.pi / 32, np.pi /32, np.pi / 16), 
    padding_mode="border")
    
data, _ = LoadImage()(nii_name)

data = CropForeground()(data)
data = EnsureChannelFirst()(data)

data_ds = Resize(spatial_size=[64,64,64])(data)
SaveImage(output_dir=output_dir,output_postfix='image', resample=False)(data_ds)



# x stacks
for n in range(num_stacks):
    data_new = deepcopy(data_ds) #torch.zeros(data_ds.shape)

    for i in tqdm(range(data_new.shape[1])):
        temp = rand_affine_x(data_ds)
        data_new[:,i,:,:] = temp[:,i,:,:]


    SaveImage(output_dir=output_dir,output_postfix='stack_x_'+str(n), resample=False)(data_new)

# y stacks

for n in range(num_stacks):
    data_new = deepcopy(data_ds) #torch.zeros(data_ds.shape)

    for i in tqdm(range(data_new.shape[2])):
        temp = rand_affine_y(data_ds)
        data_new[:,:,i,:] = temp[:,:,i,:]


    SaveImage(output_dir=output_dir,output_postfix='stack_y_'+str(n),resample=False)(data_new)

# z stacks
for n in range(num_stacks):
    data_new = deepcopy(data_ds) #torch.zeros(data_ds.shape)

    for i in tqdm(range(data_new.shape[3])):
        temp = rand_affine_z(data_ds)
        data_new[:,:,:,i] = temp[:,:,:,i]


    SaveImage(output_dir=output_dir,output_postfix='stack_z_'+str(n),resample=False)(data_new)

print('done')

