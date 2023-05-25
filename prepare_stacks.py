import nilearn.image as ni
import numpy as np
import glob
import nibabel as nb
from nibabel.processing import resample_to_output

msk = '/deneb_disk/fetal_scan_1_9_2023/morning/nii_files_rot/sample_real_data/fetal_scan_1_9_2023_morning_12_stacks/p28_t2_haste_sag_head_p_mask.nii.gz'

stacks = glob.glob(
    '/deneb_disk/fetal_scan_1_9_2023/morning/nii_files_rot/sample_real_data/fetal_scan_1_9_2023_morning_12_stacks/p*p.nii.gz')

v = ni.load_img(msk)
v2, offsets = ni.crop_img(v, return_offset=True)
v2 = v.slicer[offsets]


SZ = 96

for i, s in enumerate(stacks):

    msk2img = ni.resample_to_img(msk, s, interpolation='nearest')
    _, offsets = ni.crop_img(msk2img, return_offset=True)

    v = ni.load_img(s)
    v = v.slicer[offsets]
    v.to_filename(f'stack{i}.nii.gz')

    sx, sy, sz = v.header.get_zooms()
    print(sx,sy,sz)
    
    vc = nb.as_closest_canonical(v)
    vc.to_filename(f'rstack{i}.nii.gz')


    res = vc.header['pixdim'][1:4]

    target_voxel_size = np.array((1.5, 1.5,1.5))
    im = np.argmax(res)
    target_voxel_size[im] = 3.0


    scaling_factors = np.diag(target_voxel_size)/np.diag(vc.affine[:3,:3])
    target_affine = np.copy(vc.affine)
    target_affine[:3,:3] = vc.affine[:3,:3] * scaling_factors


    v = nb.Nifti1Image(dataobj=vc.get_fdata(),affine=target_affine,header=vc.header)
    
    v.to_filename(f'cstack{i}.nii.gz')

    print(v.shape)

    print(v.header['pixdim'][1:4])

    xcent = int(np.round(v.shape[0]/2))
    ycent = int(np.round(v.shape[1]/2))
    zcent = int(np.round(v.shape[2]/2))


    target_affine = v.affine
    target_affine[:3,3]=0
    target_shape=[SZ,SZ,SZ]
    target_shape[im]=int(SZ/2)

    v2 = ni.resample_img(v,target_affine=target_affine,target_shape=target_shape)
    v2.to_filename(f'pstack{i}_96.nii.gz')

    # offsets2 = tuple([slice(xcent-SZ/2,xcent-SZ/2),slice(ycent-SZ/2,ycent+SZ/2),slice(zcent-SZ/4,zcent+SZ/4)])
    # v2 = v.slicer[offsets2]
    # #nb.as_closest_canonical(v2).to_filename(f'stack{i}_64.nii.gz')
    # v2.to_filename(f'stack{i}_64.nii.gz')
