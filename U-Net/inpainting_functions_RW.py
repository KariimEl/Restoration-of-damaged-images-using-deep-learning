import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import transforms
from PIL import Image
import torchvision


import inpainting_functions_conversion as ifc
import inpainting_functions_treatment as ift
import inpainting_network as i_n

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.
    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = ifc.pil_to_np(img)

    return img, img_np


def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)

    plt.figure(figsize=(len(images_np) + factor, 12 + factor))

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.show()

    return grid


def save_reshape_img(img_origin_path, img_origin_reshape_path):
    img_origin = mpimg.imread(img_origin_path);
    print('taille image de base : ',img_origin.shape)

    #reshaping
    img_origin_torch = ifc.np_to_torch(img_origin).type(torch.ByteTensor)
    img_origin_reshape_torch = ift.add_border_zero(img_origin_torch)
    img_origin_reshape_np = ifc.torch_to_np(img_origin_reshape_torch)
    #saving new image reshape
    mpimg.imsave(img_origin_reshape_path, img_origin_reshape_np)
    print('taille nouvelle image de base : ',img_origin_reshape_np.shape)
