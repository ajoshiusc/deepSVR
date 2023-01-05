import glob
from monai.transforms import LoadImage, LoadImaged, RandAffine, Resized, Resize, EnsureChannelFirst, EnsureChannelFirstd, Randomizable, MapTransform, Transform, Compose

import numpy as np
from copy import deepcopy
from monai.config import KeysCollection
from typing import Optional, Any, Mapping, Hashable
import monai
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism, first
import torch

PRE_ALIGNED_STACKS = False


def stacks2batchvol(stack, dir=0, device='cpu'):

    batch_size = int(64/2)
    slice_vols = torch.zeros((batch_size, 1, 64, 64, 64),device=device)#.to(device)

    for sliceno in range(int(64/2)):

        slice_ind = torch.tensor(range(2*sliceno, 2*(sliceno+1)))

        if dir==0:
            slice_vols[sliceno, :, slice_ind, :, :] = stack[sliceno, :, :]#.to(device)
        elif dir==1:
            slice_vols[sliceno, :, :, slice_ind, :] = stack[:, slice_ind, :]#.to(device)
        elif dir==2:
            slice_vols[sliceno, :, :, :, slice_ind] = stack[:, :, slice_ind]#.to(device)

    return slice_vols


class RandMakeStack(Randomizable, Transform):
    def __init__(self, stack_axis: int = 0) -> None:
        self.stack_axis = stack_axis

    def make_stackx(self, img):

        """ rand_affine = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(2, 2, 2),
                                 rotate_range=(
            np.pi / 16, np.pi / 16, np.pi / 16),
            padding_mode="border") """

        rand_affine_x = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(.2, 1, 1),
                                   rotate_range=(
            np.pi / 16, np.pi / 32, np.pi / 32),
            padding_mode="border")

        # deepcopy(img)  # torch.zeros(data_ds.shape)
        data_new = torch.zeros(img.shape)
        data_new = Resize(spatial_size=[32, 64, 64])(data_new)
        rot_data = img #rand_affine(img)

        for i in range(data_new.shape[1]):
            temp = rand_affine_x(rot_data)
            temp = Resize(spatial_size=[32, 64, 64])(temp)
            data_new[:, i, :, :] = temp[:, i, :, :]

        return data_new

    def make_stacky(self, img):

        """ rand_affine = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(2, 2, 2),
                                 rotate_range=(
            np.pi / 16, np.pi / 16, np.pi / 16),
            padding_mode="border") """

        rand_affine_y = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(1, .2, 1),
                                   rotate_range=(
            np.pi / 32, np.pi / 16, np.pi / 32),
            padding_mode="border")

        # deepcopy(img)  # torch.zeros(data_ds.shape)
        data_new = torch.zeros(img.shape)
        data_new = Resize(spatial_size=[64, 32, 64])(data_new)
        rot_data = img #rand_affine(img)

        for i in range(data_new.shape[2]):
            temp = rand_affine_y(rot_data)
            temp = Resize(spatial_size=[64, 32, 64])(temp)
            data_new[:, :, i, :] = temp[:, :, i, :]

        return data_new

    def make_stackz(self, img):

        """ rand_affine = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(2, 2, 2),
                                 rotate_range=(
            np.pi / 16, np.pi / 16, np.pi / 16),
            padding_mode="border") """

        rand_affine_z = RandAffine(mode=("bilinear"), prob=1.0, translate_range=(1, 1, .2),
                                   rotate_range=(
            np.pi / 32, np.pi / 32, np.pi / 16),
            padding_mode="border")

        # deepcopy(img)  # torch.zeros(data_ds.shape)
        data_new = torch.zeros(img.shape)
        data_new = Resize(spatial_size=[64, 64, 32])(data_new)
        #print('hi')
        rot_data = img #rand_affine(img)

        for i in range(data_new.shape[3]):
            temp = rand_affine_z(rot_data)
            temp = Resize(spatial_size=[64, 64, 32])(temp)
            data_new[:, :, :, i] = temp[:, :, :, i]

        return data_new

    def __call__(self, img: np.ndarray) -> np.ndarray:
        
        if self.stack_axis == 0:
            return self.make_stackx(img)
        elif self.stack_axis == 1:
            return self.make_stacky(img)
        elif self.stack_axis == 2:
            return self.make_stackz(img)
        # return self.add_noise(img)


class RandMakeStackd(Randomizable, MapTransform):
    def __init__(
        self, keys: KeysCollection, stack_axis: int = 0
    ) -> None:
        super(Randomizable, self).__init__(keys)
        self.transform = RandMakeStack(stack_axis)
        self.stack_axis = stack_axis

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandMakeStackd":
        self.transform.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        self.transform.randomize(data)

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Mapping[Hashable, np.ndarray]:
        # self.randomize(data[monai.utils.first(self.keys)])

        d = dict(data)
        for key in self.keys:
            if self.stack_axis == 0:
                d[key] = self.transform.make_stackx(d[key])
            elif self.stack_axis == 1:
                d[key] = self.transform.make_stacky(d[key])
            elif self.stack_axis == 2:
                d[key] = self.transform.make_stackz(d[key])

        return d


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    filename = '/deneb_disk/feta_2022/feta_2.2/sub-080/anat/sub-080_rec-irtk_T2w.nii.gz'

    trans = Compose([LoadImaged(keys="img"), EnsureChannelFirstd(keys="img"), Resized(
        keys='img', spatial_size=[64, 64, 64]), RandMakeStackd(keys="img",stack_axis=0)])

    img = LoadImage(image_only=True)(filename)
    img = EnsureChannelFirst()(img)
    img = Resize(spatial_size=[64, 64, 64])(img)
    outimg = RandMakeStack(stack_axis=0)(img)
    #img = trans(filename)
    check_ds = Dataset(data=[{"img": filename}], transform=trans)
    ds1 = DataLoader(check_ds, batch_size=1, shuffle=True)
    ds = first(ds1)
    ds = ds['img'][0]

    fig2, (ax2, ax3) = plt.subplots(nrows=1, ncols=2)
    ax2.imshow(outimg[0, 8])
    # plt.show()
    ax3.imshow(ds[0, 8])
    plt.tight_layout()
    plt.show()

    print('done')

""" 

check_ds = Dataset(data=training_datadict, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
check_data = first(check_loader)
fixed_image = check_data["image"][0][0]
moving_image = check_data["stack"][0][0]
 """
