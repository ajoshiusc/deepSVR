import nilearn.image as ni

import glob


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

    print(v.shape)
