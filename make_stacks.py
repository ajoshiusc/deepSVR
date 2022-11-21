import glob
from monai.transforms import (
    LoadImage, RandAffine, SaveImage, Resize, EnsureChannelFirst, CropForeground
)
import numpy as np
from copy import deepcopy


PRE_ALIGNED_STACKS = False

def make_stacks(filename, num_stacks=3, output_dir='./'):

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

    sublist_full = glob.glob('/deneb_disk/feta_2022/feta_2.2/sub-*/anat/sub-*_T2w.nii.gz')
    
    # training data
    sublist = sublist_full[:60]
    out_dir = './train_fetal_data_60'
    for nii_name in sublist:
        make_stacks(filename=nii_name, num_stacks=3, output_dir=out_dir)

    # testing data
    sublist = sublist_full[60:70]
    out_dir = './test_fetal_data_10'
    for nii_name in sublist:
        make_stacks(filename=nii_name, num_stacks=3, output_dir=out_dir)


    # valid data
    sublist = sublist_full[70:]
    out_dir = './valid_fetal_data_10'
    for nii_name in sublist:
        make_stacks(filename=nii_name, num_stacks=3, output_dir=out_dir)

    print('done')
