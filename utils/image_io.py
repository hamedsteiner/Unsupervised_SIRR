'''
Date: Feb,  2020 
Author: Suhong Kim , Hamed RahmaniKhezri

This file is originally from "'Double DIP" (https://github.com/yossigandelsman/DoubleDIP) 
Some modifications are built to define the baselines
'''
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# import skvideo.io

import os
import math

matplotlib.use('agg')

def gaussian_kernel(size, sigma=2., dim=2, channels=3):
    #https://github.com/kechan/FastaiPlayground/blob/master/Quick%20Tour%20of%20Data%20Augmentation.ipynb
	# The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.
    
    kernel_size = 2*size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel


def smooth_image_torch(x, k_size=3): 
	kernel = gaussian_kernel(size=k_size).to(str(x.device))
	kernel_size = 2*k_size + 1 
	padding = (kernel_size - 1) // 2
	x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
	x = F.conv2d(x, kernel, groups=3) 
	return x


def renormalize(img):
    if torch.is_tensor(img):
        min_val = torch.min(img.reshape(-1))
        max_val = torch.max(img.reshape(-1))
        img = (img-min_val)*(1 / (max_val-min_val + 1e-20))
    else: #numpy
        min_val = np.min(img.flatten())
        max_val = np.max(img.flatten())
        new_max = 255 if max_val > 1 else 1
        img = (img-min_val)*(new_max / (max_val-min_val + 1e-20))
    return img

def edge_image(img_np):
    img = np_to_pil(img_np)
    img = img.filter(ImageFilter.FIND_EDGES)
    return pil_to_np(img)    

def blur_image(img_np, r=2): 
    img = np_to_pil(img_np)
    img = img.filter(ImageFilter.GaussianBlur(radius=r))
    return pil_to_np(img)

def compute_psnr(img1_np, img2_np): 
    if img1_np is None or img2_np is None: 
        return -1.0
    else: 
        return peak_signal_noise_ratio(img1_np, img2_np)

def compute_ssim(img1_np, img2_np): 
    if img1_np is None or img2_np is None: 
        return -1.0
    
    assert(img1_np.shape == img2_np.shape)
    if len(img1_np.shape) < 3: 
        return structural_similarity(sim1_np, img2_np)
    else: 
        # matplotlib input img shape (W, H, C)
        if img1_np.shape[0] == 1 or img1_np.shape[0] == 3: 
            img1_np = img1_np.transpose(1,2,0)
        if img2_np.shape[0] == 1 or img2_np.shape[0] == 3: 
            img2_np = img2_np.transpose(1,2,0) 
        return structural_similarity(img1_np, img2_np, multichannel=True) 

def rgb_to_gray(rgb): 
    if len(rgb.shape) == 4: # tensor
        assert(rgb.shape[1]== 3)
        gray = rgb[:, [0], :, :] * 299/1000 + rgb[:, [1], :, :] * 587/1000 +rgb[:, [2], :, :] * 114/1000
    else: # np
        assert(rgb.shape[0]== 3)
        gray = rgb[[0], :, :] * 299/1000 + rgb[[1], :, :] * 587/1000 +rgb[[2], :, :] * 114/1000
    return gray

def crop_image(img, d=32):
    """
    Make dimensions divisible by d

    :param pil img:
    :param d:
    :return:
    """
    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def crop_np_image(img_np, d=32):
    return torch_to_np(crop_torch_image(np_to_torch(img_np), d))


def crop_torch_image(img, d=32):
    """
    Make dimensions divisible by d
    image is [1, 3, W, H] or [3, W, H]
    :param pil img:
    :param d:
    :return:
    """
    new_size = (img.shape[-2] - img.shape[-2] % d,
                img.shape[-1] - img.shape[-1] % d)
    pad = ((img.shape[-2] - new_size[-2]) // 2, (img.shape[-1] - new_size[-1]) // 2)

    if len(img.shape) == 4:
        return img[:, :, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]
    assert len(img.shape) == 3
    return img[:, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]


def get_params(opt_over, net, net_input, downsampler=None):
    """
    Returns parameters that we want to optimize over.
    :param opt_over: comma separated list, e.g. "net,input" or "net"
    :param net: network
    :param net_input: torch.Tensor that stores input `z`
    :param downsampler:
    :return:
    """

    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    """
    Creates a grid from a list of images by concatenating them.
    :param images_np:
    :param nrow:
    :return:
    """
    images_torch = [torch.from_numpy(x).type(torch.FloatTensor) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(name, images_np, interpolation='lanczos', image_mode='rgb', output_path=None, show=True):
    """
    Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW or 1xHxW
        nrow: how many images will be in one row
        interpolation: interpolation used in plt.imshow
    """
    # assert len(images_np) == 2 
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    if image_mode=='lab': 
        for i in range(len(images_np)): 
            im_p = np_to_pil(images_np[i], image_mode)
            im_p = lab_to_rgb(im_p)
            images_np[i] = pil_to_np(im_p)

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, len(images_np))

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.title(name)
    plt.axis('off')
    if output_path != None: 
        if not os.path.exists(output_path): 
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path,"{}.png".format(name)))
    if show: 
        plt.show()
    plt.close()
    

def save_image(name, image_np, image_mode='rgb', output_path="output", show=False):
    p = np_to_pil(image_np, image_mode)
    p = lab_to_rgb(p) if p.mode.lower()=='lab' else  p 
    if show:
        plt.imshow(p)
        plt.show()
    if output_path != None: 
        if not os.path.exists(output_path): 
            os.makedirs(output_path)
        p.save(os.path.join(output_path,"{}.jpg".format(name)))
    plt.close()

def video_to_images(file_name, name):
    video = prepare_video(file_name)
    for i, f in enumerate(video):
        save_image(name + "_{0:03d}".format(i), f)

def images_to_video(images_dir ,name, gray=True):
    num = len(glob.glob(images_dir +"/*.jpg"))
    c = []
    for i in range(num):
        if gray:
            img = prepare_gray_image(images_dir + "/"+  name +"_{}.jpg".format(i))
        else:
            img = prepare_image(images_dir + "/"+name+"_{}.jpg".format(i))
        print(img.shape)
        c.append(img)
    save_video(name, np.array(c))

def save_heatmap(name, image_np):
    cmap = plt.get_cmap('jet')

    rgba_img = cmap(image_np)
    rgb_img = np.delete(rgba_img, 3, 2)
    save_image(name, rgb_img.transpose(2, 0, 1))


def save_graph(name, graph_list, output_path="output", show=True):
    plt.clf()
    plt.plot(graph_list)
    if show: 
        plt.show()
    if output_path != None: 
        if not os.path.exists(output_path): 
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, name + ".png"))
    plt.close()

def create_augmentations(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    aug = [np_image.copy(), np.rot90(np_image, 1, (1, 2)).copy(),
           np.rot90(np_image, 2, (1, 2)).copy(), np.rot90(np_image, 3, (1, 2)).copy()]
    flipped = np_image[:,::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (1, 2)).copy(), np.rot90(flipped, 2, (1, 2)).copy(), np.rot90(flipped, 3, (1, 2)).copy()]
    return aug


def create_video_augmentations(np_video):
    """
        convention: original, left, upside-down, right, rot1, rot2, rot3
        :param np_video:
        :return:
        """
    aug = [np_video.copy(), np.rot90(np_video, 1, (2, 3)).copy(),
           np.rot90(np_video, 2, (2, 3)).copy(), np.rot90(np_video, 3, (2, 3)).copy()]
    flipped = np_video[:, :, ::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (2, 3)).copy(), np.rot90(flipped, 2, (2, 3)).copy(),
            np.rot90(flipped, 3, (2, 3)).copy()]
    return aug


def save_graphs(name, graph_dict, output_path="output/"):
    """

    :param name:
    :param dict graph_dict: a dict from the name of the list to the list itself.
    :return:
    """
    plt.clf()
    fig, ax = plt.subplots()
    for k, v in graph_dict.items():
        ax.plot(v, label=k)
        # ax.semilogy(v, label=k)
    ax.set_xlabel('iterations')
    # ax.set_ylabel(name)
    ax.set_ylabel('MSE-loss')
    # ax.set_ylabel('PSNR')
    plt.legend()
    plt.savefig(output_path + name + ".png")


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

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


# def prepare_image(file_name, imsize=-1):
#     """
#     loads makes it divisible
#     :param file_name:
#     :return: the numpy representation of the image
#     """
#     img_pil = crop_image(get_image(file_name, imsize)[0], d=32)
#     return pil_to_np(img_pil)

def prepare_image(file_name, imsize=-1, image_mode='rgb'):
    """
    loads makes it divisible
    :param file_name:
    :return: the numpy representation of the image
    """
    try: 
        img_pil = crop_image(get_image(file_name, imsize)[0], d=32)
        img_pil = img_pil.convert('L') if image_mode == 'L' else img_pil 
        img_pil = rgb_to_lab(img_pil) if image_mode=='lab' else img_pil 
        return pil_to_np(img_pil)
    except: 
        print("Cannot find or open the file: ", file_name)
        return None 



def prepare_video(file_name, folder="output/"):
    data = skvideo.io.vread(folder + file_name)
    return crop_torch_image(data.transpose(0, 3, 1, 2).astype(np.float32) / 255.)[:35]


def save_video(name, video_np, output_path="output/"):
    outputdata = video_np * 255
    outputdata = outputdata.astype(np.uint8)
    skvideo.io.vwrite(output_path + "{}.mp4".format(name), outputdata.transpose(0, 2, 3, 1))


def prepare_gray_image(file_name):
    img = prepare_image(file_name)
    return np.array([np.mean(img, axis=0)])

def rgb_to_lab(rgb_pil): 
    from PIL import Image, ImageCms
    assert(rgb_pil.mode != 'rgb')
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    lab_pil = ImageCms.applyTransform(rgb_pil, rgb2lab)
    del lab_p, srgb_p, rgb2lab
    return lab_pil

def lab_to_rgb(lab_pil): 
    from PIL import Image, ImageCms
    assert(lab_pil.mode != 'lab')
    lab_p2  = ImageCms.createProfile("LAB")   
    srgb_p2 = ImageCms.createProfile("sRGB")
    lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p2, srgb_p2, "LAB", "RGB")
    rgb_pil = ImageCms.applyTransform(lab_pil, lab2rgb)
    del lab_p2, srgb_p2, lab2rgb
    return rgb_pil

def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def median(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for c in range(shape[0]):
        for w in range(shape[1]):
            for h in range(shape[2]):
                result[c, w, h] = sorted(i[c, w, h] for i in img_np_list)[l//2]
    return result


def average(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for i in img_np_list:
        result += i
    return result / l


def np_to_pil(img_np, img_mode='rgb'):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
        mode = 'L'
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)
        mode = img_mode.upper()

    return Image.fromarray(ar, mode=mode)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]
