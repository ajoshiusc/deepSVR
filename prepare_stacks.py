import nilearn.image as ni
import numpy as np
import glob
import nibabel as nb

msk = '/deneb_disk/fetal_scan_1_9_2023/morning/nii_files_rot/sample_real_data/fetal_scan_1_9_2023_morning_12_stacks/p28_t2_haste_sag_head_p_mask.nii.gz'

stacks = glob.glob(
    '/deneb_disk/fetal_scan_1_9_2023/morning/nii_files_rot/sample_real_data/fetal_scan_1_9_2023_morning_12_stacks/p*p.nii.gz')

v = ni.load_img(msk)
v2, offsets = ni.crop_img(v, return_offset=True)
v2 = v.slicer[offsets]


for i, s in enumerate(stacks):

    msk2img = ni.resample_to_img(msk, s, interpolation='nearest')
    _, offsets = ni.crop_img(msk2img, return_offset=True)

    v = ni.load_img(s)
    v = v.slicer[offsets]
    v.to_filename(f'stack{i}.nii.gz')
    nb.as_closest_canonical(v).to_filename(f'rstack{i}.nii.gz')


    print(v.shape)


    print(v.shape)

    xcent = int(np.round(v.shape[0]/2))
    ycent = int(np.round(v.shape[1]/2))
    zcent = int(np.round(v.shape[2]/2))

    offsets2 = tuple([slice(xcent-32,xcent+32),slice(ycent-32,ycent+32),slice(zcent-16,zcent+16)])
    v2 = v.slicer[offsets2]
    #nb.as_closest_canonical(v2).to_filename(f'stack{i}_64.nii.gz')
    v2.to_filename(f'stack{i}_64.nii.gz')
