
import monai
import torch
import itk
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import os.path
import tempfile

from utils import (
    preview_image, preview_3D_vector_field, preview_3D_deformation,
    jacobian_determinant, plot_against_epoch_numbers
)

monai.config.print_config()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=2938649572)

use_synthetic_data = True
directory = '/home/ajoshi/monai_data' #os.environ.get("MONAI_DATA_DIRECTORY")

root_dir = tempfile.mkdtemp() if directory is None else directory
data_dir = os.path.join(root_dir, "synthetic_data")

save_img = monai.transforms.SaveImage(output_dir=data_dir, output_postfix="img", print_log=False)
save_seg = monai.transforms.SaveImage(output_dir=data_dir, output_postfix="seg", print_log=False)

# Set the amount of synthetic data here
num_img_seg_pairs_to_generate = 50
num_segmentation_classes = 5
num_segs_to_select = 6

for i in range(num_img_seg_pairs_to_generate):
    img, seg = monai.data.synthetic.create_test_image_3d(
        64, 64, 64,  # image size
        num_objs=5,
        rad_max=30, rad_min=4,
        noise_max=0.5,
        num_seg_classes=num_segmentation_classes - 1,  # background is not counted
        channel_dim=0,
        random_state=None
    )
    save_img(img)
    if i < num_segs_to_select:  # pretend that only a few segmentations are available
        save_seg(seg)

image_paths = glob.glob(os.path.join(data_dir, "*/*img.nii.gz"))
segmentation_paths = glob.glob(os.path.join(data_dir, "*/*seg.nii.gz"))

print(segmentation_paths)








device = torch.device("cuda:0")


# Function to extract an image or segmentation ID from its path
def path_to_id(path):
    if use_synthetic_data:
        return os.path.basename(path).split('_')[0]
    else:
        return os.path.basename(path).strip('OAS1_')[:8]


seg_ids = list(map(path_to_id, segmentation_paths))
img_ids = map(path_to_id, image_paths)
data = []
for img_index, img_id in enumerate(img_ids):
    data_item = {'img': image_paths[img_index]}
    if img_id in seg_ids:
        data_item['seg'] = segmentation_paths[seg_ids.index(img_id)]
    data.append(data_item)




data_seg_available = list(filter(lambda d: 'seg' in d.keys(), data))
data_seg_unavailable = list(filter(lambda d: 'seg' not in d.keys(), data))

data_seg_available_train, data_seg_available_valid = \
    monai.data.utils.partition_dataset(data_seg_available, ratios=(8, 2))
# Validation of the segmentation network only makes sense if you have enough segmentation labels.
# E.g. definitely skip validation here if there's just one segmentation label.
#     

