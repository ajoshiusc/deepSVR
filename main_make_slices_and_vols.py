import glob
from monai.transforms import (
    LoadImage, RandAffine, SaveImage, Resize, EnsureChannelFirst, CropForeground, RandAffineGrid, Resample
)
import numpy as np
from copy import deepcopy
from random import randrange
from os.path import split, splitext, join


def make_slices_vols(filename, num_stacks=3, output_dir='./'):

    _, fname = split(filename)
    subname = fname[:-7]

    data, _ = LoadImage()(filename)
    '''SaveImage(output_dir=output_dir,
              output_postfix='orig', resample=False)(data)'''
    data = EnsureChannelFirst()(data)

    data = CropForeground(margin=32)(data)

    data = Resize(spatial_size=[64, 64, 64])(data)
    SaveImage(output_dir=output_dir, output_postfix='image',
              resample=False)(data)

    rand_affine = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(2, 2, 2), rotate_range=(
        np.pi / 16, np.pi / 16, np.pi / 16), padding_mode="border", cache_grid=True)
    rand_affine_grid = RandAffineGrid(translate_range=(
        2, 2, 2), rotate_range=(np.pi / 16, np.pi / 16, np.pi / 16))
    resample = Resample(mode=("bilinear"), padding_mode="border")


    for n in range(num_stacks):

        data_new = rand_affine(data)

        SaveImage(output_dir=output_dir, output_postfix='image_' +
                  str(n), resample=False)(data_new)
        slice_num = randrange(16)
        dim = randrange(3)

        if dim == 2:
            # downsample vol to slice res along slice axis
            data_new_ds = Resize(spatial_size=[64, 64, 16])(data_new)
            # generate rand number between 0-num slices and take slice of the vol. save it as h5 file
            slice = data_new_ds[:, :, :, slice_num]

            # also generate 64^3 vol with 0 everywhere except at the slice locs save it
            temp = 0*deepcopy(data_new_ds)
            temp[:, :, :, slice_num] = slice

        elif dim == 1:
            # downsample vol to slice res along slice axis
            data_new_ds = Resize(spatial_size=[64, 16, 64])(data_new)
            # generate rand number between 0-num slices and take slice of the vol. save it as h5 file
            slice = data_new_ds[:, :, slice_num, :]

            # also generate 64^3 vol with 0 everywhere except at the slice locs save it
            temp = 0*deepcopy(data_new_ds)
            temp[:, :, slice_num, :] = slice
        elif dim == 0:
            # downsample vol to slice res along slice axis
            data_new_ds = Resize(spatial_size=[16, 64, 64])(data_new)
            # generate rand number between 0-num slices and take slice of the vol. save it as h5 file
            slice = data_new_ds[:, slice_num, :, :]

            # also generate 64^3 vol with 0 everywhere except at the slice locs save it
            temp = 0*deepcopy(data_new_ds)
            temp[:, slice_num, :, :] = slice

        np.savez(join(output_dir, subname, 'slice_'+str(n)+'.npz'),
                 slice_num=slice_num, slice=slice, dim=dim)

        slice_vol = Resize(spatial_size=[64, 64, 64])(temp)

        SaveImage(output_dir=output_dir, output_postfix='slice_' +
                  str(n), resample=False)(slice_vol)
        # generate a random affine transform, apply same transform and apply to volume and slice as well

        agrid = rand_affine_grid(spatial_size=[64, 64, 64])
        data_new = resample(img=data_new, grid=agrid)
        SaveImage(output_dir=output_dir, output_postfix='image_aff' +
                  str(n), resample=False)(data_new)
        slice_vol_rot = resample(img=slice_vol, grid=agrid)
        SaveImage(output_dir=output_dir, output_postfix='slice_aff' +
                  str(n), resample=False)(slice_vol_rot)


if __name__ == '__main__':

    sublist = glob.glob('/deneb_disk/feta_2022/feta_2.2/sub*/*/*T2w.nii.gz')
    out_dir = './feta_syn_data_slices'
    #sublist =['/deneb_disk/feta_2022/feta_2.2/sub-080/anat/sub-080_rec-irtk_T2w.nii.gz']

    for nii_name in sublist:

        make_slices_vols(filename=nii_name, num_stacks=32, output_dir=out_dir)

    print('done')
