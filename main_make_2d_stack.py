from monai.transforms import (
    LoadImage, LoadImaged, EnsureChannelFirstd,
    Resized, EnsureTyped, Compose,NormalizeIntensityd, RandAffined, RandAffine, SaveImage, Resize, EnsureChannelFirst
)

import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

output_dir = '.'
nii_name = '/deneb_disk/feta_2022/feta_2.2/sub-080/anat/sub-080_rec-irtk_T2w.nii.gz'

num_stacks = 3
rand_affine = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(1, 1, 1), 
    rotate_range=(np.pi / 32, np.pi /32, np.pi / 16), 
    padding_mode="border")
    
data, meta = LoadImage()(nii_name)

data = EnsureChannelFirst()(data)

data_ds = Resize(spatial_size=[64,64,64])(data)
SaveImage(output_dir=output_dir,output_postfix='image')(data_ds)


for n in range(num_stacks):
    data_new = deepcopy(data_ds) #torch.zeros(data_ds.shape)

    for i in tqdm(range(data_ds.shape[3])):

        temp = rand_affine(data_ds)
        data_new[:,:,:,i] = temp[:,:,:,i]


    SaveImage(output_dir=output_dir,output_postfix='stack_'+str(n))(data_new)

print('done')

