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

def slice_generator(image, direction, slice_no, profile):
    slice_volume = torch.zeros(image.size())
    slice_width = profile.size(0)
    bottom = max(slice_no-int(profile.size(0)/2), 0)
    top = bottom + profile.size(0)
    slice_ind = torch.tensor(range(bottom, top))
    
    if direction == 0:
        new = torch.sum(torch.zeros(image.size()),2, keepdim = True)
        print(new.size())
        print(image[:,:,1,:,:].size())
        for i in slice_ind:
            new += image[:,:,i:i+1,:,:]*profile[i-bottom]
        slice_volume[:,:,slice_ind,:,:] = new.repeat(1,1,profile.size(0),1,1)
        
    elif direction == 1:
        new = torch.sum(torch.zeros(image.size()),3, keepdim = True)
        for i in slice_ind:
            new += image[:,:,:,i:i+1,:]*profile[i-bottom]
        slice_volume[:,:,:,slice_ind,:] = new.repeat(1,1,1,profile.size(0),1)
        
    elif direction == 2:
        new = torch.sum(torch.zeros(image.size()),4, keepdim = True)
        for i in slice_ind:
            new += image[:,:,:,:,i:i+1]*profile[i-bottom]
        slice_volume[:,:,:,:,slice_ind] = new.repeat(1,1,1,1,profile.size(0))
    
    return slice_volume