import nilearn.image as ni
import numpy as np
import glob
import nibabel as nb
from nibabel.processing import resample_to_output

import torch
import torchvision.transforms.functional as TF


SZ = 96


def resize_3d_image(image, direction):
    # direction: 0 for dx, 1 for dy, 2 for dz

    dx, dy, dz = image.shape
    if direction == 0:
        padding = (
            (0, 0),
            ((SZ - dy) // 2, (SZ - dy) - (SZ - dy) // 2),
            ((SZ - dz) // 2, (SZ - dz) - (SZ - dz) // 2),
        )
    elif direction == 1:
        padding = (
            ((SZ - dx) // 2, (SZ - dx) - (SZ - dx) // 2),
            (0, 0),
            ((SZ - dz) // 2, (SZ - dz) - (SZ - dz) // 2),
        )
    else:
        padding = (
            ((SZ - dx) // 2, (SZ - dx) - (SZ - dx) // 2),
            ((SZ - dy) // 2, (SZ - dy) - (SZ - dy) // 2),
            (0, 0),
        )
    # Pad or crop the image based on the specified direction
    padded_image = np.pad(image, padding)

    return padded_image


msk = "/deneb_disk/fetal_scan_1_9_2023/morning/nii_files_rot/sample_real_data/fetal_scan_1_9_2023_morning_12_stacks/p28_t2_haste_sag_head_p_mask.nii.gz"

stacks = glob.glob(
    "/deneb_disk/fetal_scan_1_9_2023/morning/nii_files_rot/sample_real_data/fetal_scan_1_9_2023_morning_12_stacks/p*p.nii.gz"
)

v = ni.load_img(msk)
v2, offsets = ni.crop_img(v, return_offset=True)
v2 = v.slicer[offsets]


for i, s in enumerate(stacks):
    msk2img = ni.resample_to_img(msk, s, interpolation="nearest")
    _, offsets = ni.crop_img(msk2img, return_offset=True)

    v = ni.load_img(s)

    # Get the image data arrays
    data1 = msk2img.get_fdata()
    data2 = v.get_fdata()

    # Perform element-wise multiplication
    multiplied_data = np.multiply(data1, data2)

    # Create a new NIfTI image with the multiplied data
    v = nb.Nifti1Image(multiplied_data, affine=v.affine, header=v.header)

    v = v.slicer[offsets]
    v.to_filename(f"stack{i}.nii.gz")

    sx, sy, sz = v.header.get_zooms()
    print(sx, sy, sz)

    vc = nb.as_closest_canonical(v)
    vc.to_filename(f"rstack{i}.nii.gz")

    res = vc.header["pixdim"][1:4]

    target_voxel_size = np.array((1.5, 1.5, 1.5))
    slice_axis = np.argmax(res)
    target_voxel_size[slice_axis] = 3.0

    scaling_factors = np.diag(target_voxel_size) / np.diag(vc.affine[:3, :3])
    target_affine = np.copy(vc.affine)
    target_affine[:3, :3] = vc.affine[:3, :3] * scaling_factors

    v = nb.Nifti1Image(dataobj=vc.get_fdata(), affine=target_affine, header=vc.header)

    v.to_filename(f"cstack{i}.nii.gz")

    img = resize_3d_image(v.get_fdata(), slice_axis)

    hdr = nb.Nifti1Header()
    hdr.set_data_shape(img.shape)
    hdr.set_zooms(target_voxel_size)  # set voxel size
    hdr.set_xyzt_units(2)
    vp = nb.Nifti1Image(
        dataobj=img, affine=np.diag(np.concatenate((target_voxel_size, [0])))
    )
    vp.to_filename(f"pstack{i}_{SZ}.nii.gz")

""" 
    print(v.shape)

    print(v.header['pixdim'][1:4])

    xcent = int(np.round(v.shape[0]/2))
    ycent = int(np.round(v.shape[1]/2))
    zcent = int(np.round(v.shape[2]/2))


    target_affine = v.affine
    target_affine[:3,3]=0
    target_shape=[SZ,SZ,SZ]
    target_shape[slice_axis]=v.shape[slice_axis]

    v2 = ni.resample_img(v,target_affine=target_affine,target_shape=target_shape)
    v2.to_filename(f'pstack{i}_{SZ}.nii.gz')
 """
# offsets2 = tuple([slice(xcent-SZ/2,xcent-SZ/2),slice(ycent-SZ/2,ycent+SZ/2),slice(zcent-SZ/4,zcent+SZ/4)])
# v2 = v.slicer[offsets2]
# #nb.as_closest_canonical(v2).to_filename(f'stack{i}_64.nii.gz')
# v2.to_filename(f'stack{i}_64.nii.gz')
