import numpy as np
import matplotlib.pyplot as plt
import monai
import torch
from typing import Optional, Tuple
from monai.transforms.utils import rescale_array



def create_synthetic_stacks_3d(
    height: int,
    width: int,
    depth: int,
    num_objs: int = 12,
    rad_max: int = 30,
    rad_min: int = 5,
    noise_max: float = 0.0,
    num_seg_classes: int = 5,
    channel_dim: Optional[int] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a noisy 3D image and segmentation.

    Args:
        height: height of the image. The value should be larger than `2 * rad_max`.
        width: width of the image. The value should be larger than `2 * rad_max`.
        depth: depth of the image. The value should be larger than `2 * rad_max`.
        num_objs: number of circles to generate. Defaults to `12`.
        rad_max: maximum circle radius. Defaults to `30`.
        rad_min: minimum circle radius. Defaults to `5`.
        noise_max: if greater than 0 then noise will be added to the image taken from
            the uniform distribution on range `[0,noise_max)`. Defaults to `0`.
        num_seg_classes: number of classes for segmentations. Defaults to `5`.
        channel_dim: if None, create an image without channel dimension, otherwise create
            an image with channel dimension as first dim or last dim. Defaults to `None`.
        random_state: the random generator to use. Defaults to `np.random`.

    Returns:
        Randomised Numpy array with shape (`width`, `height`, `depth`)

    See also:
        :py:meth:`~create_test_image_2d`
    """

    if rad_max <= rad_min:
        raise ValueError("`rad_min` should be less than `rad_max`.")
    if rad_min < 1:
        raise ValueError("`rad_min` should be no less than 1.")
    min_size = min(width, height, depth)
    if min_size <= 2 * rad_max:
        raise ValueError("the minimal size of the image should be larger than `2 * rad_max`.")

    image = np.zeros((width, height, depth))
    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state  # type: ignore

    for _ in range(num_objs):
        x = rs.randint(rad_max, width - rad_max)
        y = rs.randint(rad_max, height - rad_max)
        z = rs.randint(rad_max, depth - rad_max)
        rad = rs.randint(rad_min, rad_max)
        spy, spx, spz = np.ogrid[-x : width - x, -y : height - y, -z : depth - z]
        circle = (spx * spx + spy * spy + spz * spz) <= rad * rad

        if num_seg_classes > 1:
            image[circle] += np.ceil(rs.random() * num_seg_classes)
        else:
            image[circle] += rs.random() * 0.5 + 0.5

    labels = np.ceil(image).astype(np.int32, copy=False)

    norm = rs.uniform(0, num_seg_classes * noise_max, size=image.shape)
    noisyimage: np.ndarray = rescale_array(np.maximum(image, norm))  # type: ignore

    if channel_dim is not None:
        if not (isinstance(channel_dim, int) and channel_dim in (-1, 0, 3)):
            raise AssertionError("invalid channel dim.")
        noisyimage, labels = (
            (noisyimage[None], labels[None]) if channel_dim == 0 else (noisyimage[..., None], labels[..., None])
        )

    return noisyimage, image, labels


def preview_image(image_array, normalize_by="volume", cmap=None, figsize=(12, 12), threshold=None):
    """
    Display three orthogonal slices of the given 3D image.

    image_array is assumed to be of shape (H,W,D)

    If a number is provided for threshold, then pixels for which the value
    is below the threshold will be shown in red
    """
    if normalize_by == "slice":
        vmin = None
        vmax = None
    elif normalize_by == "volume":
        vmin = 0
        vmax = image_array.max().item()
    else:
        raise(ValueError(
            f"Invalid value '{normalize_by}' given for normalize_by"))

    # half-way slices
    x, y, z = np.array(image_array.shape)//2
    imgs = (image_array[x, :, :], image_array[:, y, :], image_array[:, :, z])

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for ax, im in zip(axs, imgs):
        ax.axis('off')
        ax.imshow(im, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)

        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape+(4,))  # RGBA array
            red[im <= threshold] = [1, 0, 0, 1]
            ax.imshow(red, origin='lower')

    plt.show()


def plot_2D_vector_field(vector_field, downsampling):
    """Plot a 2D vector field given as a tensor of shape (2,H,W).

    The plot origin will be in the lower left.
    Using "x" and "y" for the rightward and upward directions respectively,
      the vector at location (x,y) in the plot image will have
      vector_field[1,y,x] as its x-component and
      vector_field[0,y,x] as its y-component.
    """
    downsample2D = monai.networks.layers.factories.Pool['AVG', 2](
        kernel_size=downsampling)
    vf_downsampled = downsample2D(vector_field.unsqueeze(0))[0]
    plt.quiver(
        vf_downsampled[1, :, :], vf_downsampled[0, :, :],
        angles='xy', scale_units='xy', scale=downsampling,
        headwidth=4.
    )


def preview_3D_vector_field(vector_field, downsampling=None):
    """
    Display three orthogonal slices of the given 3D vector field.

    vector_field should be a tensor of shape (3,H,W,D)

    Vectors are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """

    if downsampling is None:
        # guess a reasonable downsampling value to make a nice plot
        downsampling = max(1, int(max(vector_field.shape[1:])) >> 5)

    x, y, z = np.array(vector_field.shape[1:])//2  # half-way slices
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[1, 2], x, :, :], downsampling)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 2], :, y, :], downsampling)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_vector_field(vector_field[[0, 1], :, :, z], downsampling)
    plt.show()


def plot_2D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot an x-y grid warped by this deformation.

    vector_field should be a tensor of shape (2,H,W)
    """
    _, H, W = vector_field.shape
    grid_img = np.zeros((H,W))
    grid_img[np.arange(0, H, grid_spacing),:]=1
    grid_img[:,np.arange(0, W, grid_spacing)]=1
    grid_img = torch.tensor(grid_img, dtype=vector_field.dtype).unsqueeze(0) # adds channel dimension, now (C,H,W)
    warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")
    grid_img_warped = warp(grid_img.unsqueeze(0), vector_field.unsqueeze(0))[0]
    plt.imshow(grid_img_warped[0], origin='lower', cmap='gist_gray')


def preview_3D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot warped grids along three orthogonal slices.

    vector_field should be a tensor of shape (3,H,W,D)
    kwargs are passed to matplotlib plotting

    Deformations are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """
    x, y, z = np.array(vector_field.shape[1:])//2  # half-way slices
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plot_2D_deformation(vector_field[[1, 2], x, :, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 2], :, y, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plot_2D_deformation(vector_field[[0, 1], :, :, z], grid_spacing, **kwargs)
    plt.show()


def jacobian_determinant(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.

    vf is assumed to be a vector field of shape (3,H,W,D),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.

    Returns a numpy array of shape (H-1,W-1,D-1).
    """

    _, H, W, D = vf.shape

    # Compute discrete spatial derivatives
    def diff_and_trim(array, axis): return np.diff(
        array, axis=axis)[:, :(H-1), :(W-1), :(D-1)]
    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1
    dz[2] += 1

    # Compute determinant at each spatial location
    det = dx[0]*(dy[1]*dz[2]-dz[1]*dy[2]) - dy[0]*(dx[1]*dz[2] -
                                                   dz[1]*dx[2]) + dz[0]*(dx[1]*dy[2]-dy[1]*dx[2])

    return det


def plot_against_epoch_numbers(epoch_and_value_pairs, **kwargs):
    """
    Helper to reduce code duplication when plotting quantities that vary over training epochs

    epoch_and_value_pairs: An array_like consisting of pairs of the form (<epoch number>, <value of thing to plot>)
    kwargs are forwarded to matplotlib.pyplot.plot
    """
    array = np.array(epoch_and_value_pairs)
    plt.plot(array[:, 0], array[:, 1], **kwargs)
    plt.xlabel("epochs")
