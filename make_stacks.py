import glob
from monai.transforms import (
    LoadImage, RandAffine, SaveImage, Resize, EnsureChannelFirst, CropForeground
)
import numpy as np
from copy import deepcopy


PRE_ALIGNED_STACKS = False

def make_slices(filename, num_stacks=3, output_dir='./'):

    rand_affine = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(2, 2, 2),
                             rotate_range=(
        np.pi / 16, np.pi / 16, np.pi / 16),
        padding_mode="border")

    rand_affine_x = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(.2, 1, 1),
                               rotate_range=(
                                   np.pi / 16, np.pi / 32, np.pi / 32),
                               padding_mode="border")

    rand_affine_y = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(1, .2, 1),
                               rotate_range=(
                                   np.pi / 32, np.pi / 16, np.pi / 32),
                               padding_mode="border")

    rand_affine_z = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(1, 1, .2),
                               rotate_range=(
                                   np.pi / 32, np.pi / 32, np.pi / 16),
                               padding_mode="border")

    data, _ = LoadImage()(filename)
    '''SaveImage(output_dir=output_dir,
              output_postfix='orig', resample=False)(data)'''
    data = EnsureChannelFirst()(data)

    data = CropForeground()(data)

    data_ds = Resize(spatial_size=[64, 64, 64])(data)
    SaveImage(output_dir=output_dir, output_postfix='image',
              resample=False)(data_ds)

    # x stacks
    for n in range(num_stacks):
        data_new = deepcopy(data_ds)  # torch.zeros(data_ds.shape)
        data_new = Resize(spatial_size=[16, 64, 64])(data_new)

        if PRE_ALIGNED_STACKS:
            rot_data = deepcopy(data_ds)
        else:
            rot_data = rand_affine(data_ds)

        for i in range(data_new.shape[1]):
            temp = rand_affine_x(rot_data)
            temp = Resize(spatial_size=[16, 64, 64])(temp)
            data_new[:, i, :, :] = temp[:, i, :, :]

        SaveImage(output_dir=output_dir, output_postfix='stack_x_' +
                  str(n), resample=False)(data_new)

    # y stacks

    for n in range(num_stacks):
        data_new = deepcopy(data_ds)  # torch.zeros(data_ds.shape)
        data_new = Resize(spatial_size=[64, 16, 64])(data_new)

        rot_data = rand_affine(data_ds)

        for i in range(data_new.shape[2]):
            temp = rand_affine_y(rot_data)
            temp = Resize(spatial_size=[64, 16, 64])(temp)

            data_new[:, :, i, :] = temp[:, :, i, :]

        SaveImage(output_dir=output_dir, output_postfix='stack_y_' +
                  str(n), resample=False)(data_new)

    # z stacks
    for n in range(num_stacks):
        data_new = deepcopy(data_ds)  # torch.zeros(data_ds.shape)
        data_new = Resize(spatial_size=[64, 64, 16])(data_new)

        rot_data = rand_affine(data_ds)

        for i in range(data_new.shape[3]):
            temp = rand_affine_z(rot_data)
            temp = Resize(spatial_size=[64, 64, 16])(temp)

            data_new[:, :, :, i] = temp[:, :, :, i]

        SaveImage(output_dir=output_dir, output_postfix='stack_z_' +
                  str(n), resample=False)(data_new)


if __name__ == '__main__':

    sublist = glob.glob('/deneb_disk/camcan/cc700/mri/pipeline/release004/BIDS_20190411/anat/sub-CC*/anat/sub-CC2103*_T2w.nii.gz')
    out_dir = './normal_mris_data'
    #sublist =['/deneb_disk/feta_2022/feta_2.2/sub-080/anat/sub-080_rec-irtk_T2w.nii.gz']

    for nii_name in sublist:

        make_slices(filename=nii_name, num_stacks=3, output_dir=out_dir)

    print('done')
