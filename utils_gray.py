import os
import numpy as np
import torch

from skimage import io,color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, convolve
from typing import Callable
import os
import cv2
import pandas as pd

from numbers import Number
from typing import Container
from collections import defaultdict


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images
def addjust_shape(image,mask,shape):
    (c, d, h, w) = image.shape
    image_out  = np.full((c,shape[0],shape[1],shape[2]),-9,dtype = np.float32)
    mask_out  = np.full((shape[0],shape[1],shape[2]),0,dtype = np.long)
    if (d<shape[0] or h<shape[1]):
        d1 = int(round((shape[0] - d) / 2.))
        h1 = int(round((shape[1] - h) / 2.))
        w1 = int(round((shape[2] - w) / 2.))
        image_out[:,d1:d1 + d, h1:h1 + h, w1:w1 + w] = image
        mask_out[d1:d1 + d, h1:h1 + h, w1:w1 + w] = mask
    elif (d>shape[0] or h>shape[1]):
        d1 = int(round((d - shape[0]) / 2.))
        h1 = int(round((h - shape[1]) / 2.))
        w1 = int(round((w - shape[2]) / 2.))
        image_out = image[d1:d1 + output_size[0], h1:h1 + output_size[1], w1:w1 + output_size[2]]
        mask_out = mask[d1:d1 + output_size[0], h1:h1 + output_size[1], w1:w1 + output_size[2]]

    return image_out,mask_out

def addjust_shape_img(image,shape):
    (c, d, h, w) = image.shape
    image_out  = np.full((c,shape[0],shape[1],shape[2]),-9,dtype = np.float32)

    if (d<shape[0] or h<shape[1]):
        d1 = int(round((shape[0] - d) / 2.))
        h1 = int(round((shape[1] - h) / 2.))
        w1 = int(round((shape[2] - w) / 2.))
        image_out[:,d1:d1 + d, h1:h1 + h, w1:w1 + w] = image

    elif (d>shape[0] or h>shape[1]):
        d1 = int(round((d - shape[0]) / 2.))
        h1 = int(round((h - shape[1]) / 2.))
        w1 = int(round((w - shape[2]) / 2.))
        image_out = image[d1:d1 + output_size[0], h1:h1 + output_size[1], w1:w1 + output_size[2]]
    return image_out

def addjust_shape_mat(mask,shape):
    (d, h, w) = mask.shape
    mask_out  = np.full((shape[0],shape[1],shape[2]),0,dtype = np.long)
    if (d<shape[0] or h<shape[1]):
        d1 = int(round((shape[0] - d) / 2.))
        h1 = int(round((shape[1] - h) / 2.))
        w1 = int(round((shape[2] - w) / 2.))
        mask_out[d1:d1 + d, h1:h1 + h, w1:w1 + w] = mask
    elif (d>shape[0] or h>shape[1]):
        d1 = int(round((d - shape[0]) / 2.))
        h1 = int(round((h - shape[1]) / 2.))
        w1 = int(round((w - shape[2]) / 2.))
        mask_out = mask[d1:d1 + shape[0], h1:h1 + shape[1], w1:w1 + shape[2]]
    mask_out[mask_out ==3] = 4
    return mask_out


class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """
    def __init__(self, crop=(32, 32), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        # transforming to tensor
        image = F.to_tensor(image)
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask




class Image2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them. As opposed to ImageToImage2D, this
    reads a single image and requires a simple augmentation transform.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as a prediction
           dataset.

    Args:
        
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        transform: augmentation transform. If bool(joint_transform) evaluates to False,
            torchvision.transforms.ToTensor will be used.
    """

    def __init__(self, dataset_path: str, transform: Callable = None):

        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, '')
        self.images_list = os.listdir(self.input_path)

        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]

        image = np.load(os.path.join(self.input_path, image_filename))['train'].astype(np.float32)

        # image = np.transpose(image,(2,0,1))

        image = correct_dims(image)

        image = self.transform(image)

        # image = np.swapaxes(image,2,0)

        return image, image_filename

def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:        
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value/normalize for key, value in self.results.items()}

class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axis_prob=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, image,label):
        assert image.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        assert label.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if image.ndim == 3:
                    image = np.flip(image, axis)
                else:
                    channels = [np.flip(image[c], axis) for c in range(image.shape[0])]
                    image = np.stack(channels, axis=0)
                if label.ndim == 3:
                    label = np.flip(label, axis)
                else:
                    channels = [np.flip(label[c], axis) for c in range(label.shape[0])]
                    label = np.stack(channels, axis=0)

        return image,label

class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='constant', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, image,label):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        k = self.random_state.randint(0, 4)
        angle = k*90

        if image.ndim == 3:
            image = rotate(image, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-9)
        else:
            channels = [rotate(image[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-9) for c
                        in range(image.shape[0])]
            image = np.stack(channels, axis=0)
        if label.ndim == 3:
            label = rotate(label, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=0)
        else:
            channels = [rotate(label[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=0) for c
                        in range(label.shape[0])]
            label = np.stack(channels, axis=0)

        return image,label

class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, image):
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result_image = self.mean + alpha * (image - self.mean)
            return result_image

        return image
class RandomRotFlip:
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, image,label):
        k = np.random.randint(0, 4)
        image = np.stack([np.rot90(x,k) for x in image],axis=0)
        label = np.rot90(label, k)
        axis = np.random.randint(1, 4)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis-1).copy()

        return image, label
def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, image, label):
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return image, label

def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    if not per_channel:
        mn = data_sample.mean()
        if preserve_range:
            minm = data_sample.min()
            maxm = data_sample.max()
        if np.random.random() < 0.5 and contrast_range[0] < 1:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
        data_sample = (data_sample - mn) * factor + mn
        if preserve_range:
            data_sample[data_sample < minm] = minm
            data_sample[data_sample > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample[c] = (data_sample[c] - mn) * factor + mn
            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


class ContrastAugmentationTransform(object):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True,p_per_sample=1.):
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel

    def __call__(self,image,label):
        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                image[b] = augment_contrast(image[b], contrast_range=self.contrast_range,
                                            preserve_range=self.preserve_range, per_channel=self.per_channel)
        return image, label

def augment_brightness_additive(data_sample, mu:float, sigma:float , per_channel:bool=True, p_per_channel:float=1.):
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                data_sample[c] += rnd_nb
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                data_sample[c] += rnd_nb
    return data_sample


class BrightnessTransform(object):
    def __init__(self, mu, sigma, per_channel=True, p_per_sample=1., p_per_channel=1.):
        self.p_per_sample = p_per_sample
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, sample):

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data[b] = augment_brightness_additive(data[b], self.mu, self.sigma, self.per_channel,
                                                      p_per_channel=self.p_per_channel)

        return image, label

class ImageToImage3D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, RandomFlip: Callable = None, RandomRotate: Callable = None,RandomContrast: Callable = None,one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, '')
        self.output_path = os.path.join(dataset_path, '')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        self.RandomFlip = RandomFlip
        self.RandomRotate = RandomRotate
        self.RandomContrast = RandomContrast

            
    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        #print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        #image = cv2.imread(os.path.join(self.input_path, image_filename))
        image_ori = np.load(os.path.join(self.input_path, image_filename))['image'].astype(np.float32)
        mask_ori = np.load(os.path.join(self.input_path, image_filename))['label'].astype(np.int64)
        image = np.full((4,160,224,224),-9,dtype = np.float32)
        mask = np.full((160,224,224),0,dtype = np.long)
        image[:,3:158,...] = image_ori
        mask[3:158,...] = mask_ori
       
        # print(image.shape)
        # read mask image
        #mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)
        
        #mask[mask<=127] = 0
        #mask[mask>127] = 1
        # correct dimensions if needed
        assert len(image.shape)<5, 'Input dims error'
        assert len(mask.shape)<4, 'Input dims error'
        #image, mask = correct_dims(image, mask)
        # print(image.shape)
        
        #mask = np.expand_dims(mask, axis=0)

        if self.RandomFlip:
            image,mask = self.RandomFlip(image,mask)

        if self.RandomRotate:
            image,mask = self.RandomRotate(image,mask)

        if self.RandomContrast:
            image = self.RandomContrast(image)
        #image = np.expand_dims(image, axis=0)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2], mask.shape[3])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print(mask.shape)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)

        return image, mask, image_filename
def class_convert(source):

    mask_WT = source.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 3] = 1

    mask_TC = source.copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 3] = 1


    mask_ET = source.copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 3] = 1

    return np.stack([mask_WT, mask_TC, mask_ET])
    
class ImageToImage3D_Convert(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, RandomFlip: Callable = None, RandomRotate: Callable = None,RandomContrast: Callable = None,one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, '')
        self.output_path = os.path.join(dataset_path, '')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        self.RandomFlip = RandomFlip
        self.RandomRotate = RandomRotate
        self.RandomContrast = RandomContrast
      
    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        #print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        #image = cv2.imread(os.path.join(self.input_path, image_filename))
        image_ori = np.load(os.path.join(self.input_path, image_filename))['image'].astype(np.float32)
        mask_ori = np.load(os.path.join(self.input_path, image_filename))['label'].astype(np.int64)
        image = np.full((160,224,224),-9,dtype = np.float32)
        mask = np.full((160,224,224),0,dtype = np.long)
        image[3:158,...] = image_ori
        mask[3:158,...] = mask_ori
       
        # print(image.shape)
        # read mask image
        #mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)
        
        #mask[mask<=127] = 0
        #mask[mask>127] = 1
        # correct dimensions if needed
        assert len(image.shape)<4, 'Input dims error'
        assert len(mask.shape)<4, 'Input dims error'
        #image, mask = correct_dims(image, mask)
        # print(image.shape)
        
        #mask = np.expand_dims(mask, axis=0)

        if self.RandomFlip:
            image,mask = self.RandomFlip(image,mask)

        if self.RandomRotate:
            image,mask = self.RandomRotate(image,mask)

        
        image = np.expand_dims(image, axis=0)
        mask = class_convert(mask)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2], mask.shape[3])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print(mask.shape)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)

        return image, mask, image_filename
class ImageToImage3D_Pre(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, shape = (160,224,224), RandomFlip: Callable = None, RandomRotate: Callable = None,RandomContrast: Callable = None,one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, '')
        self.output_path = os.path.join(dataset_path, '')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask
        self.imgshape = shape
        self.RandomFlip = RandomFlip
        self.RandomRotate = RandomRotate
        self.RandomContrast = RandomContrast

        

            
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        # print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        #image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = np.load(os.path.join(self.input_path, image_filename))['image'].astype(np.float32)
        image = addjust_shape_img(image,shape = self.imgshape)       
        # print(image.shape)
        # read mask image
        #mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)
        
        #mask[mask<=127] = 0
        #mask[mask>127] = 1
        # correct dimensions if needed
        assert len(image.shape)<5, 'Input dims error'
        #assert len(mask.shape)<4, 'Input dims error'
        #image, mask = correct_dims(image, mask)
        # print(image.shape)
        
        #mask = np.expand_dims(mask, axis=0)

        if self.RandomFlip:
            image,mask = self.RandomFlip(image,mask)

        if self.RandomRotate:
            image,mask = self.RandomRotate(image,mask)

        if self.RandomContrast:
            image = self.RandomContrast(image)
        #image = np.expand_dims(image, axis=0)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2], mask.shape[3])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print(mask.shape)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)

        return image,image_filename

class ImageToImage3D_Valid(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, shape = (160,224,224), RandomFlip: Callable = None, RandomRotate: Callable = None,RandomContrast: Callable = None,one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, '')
        self.output_path = os.path.join(dataset_path, '')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask
        self.imgshape = shape
        self.RandomFlip = RandomFlip
        self.RandomRotate = RandomRotate
        self.RandomContrast = RandomContrast

        

            
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        # print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        #image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = np.load(os.path.join(self.input_path, image_filename))['image'].astype(np.float32)
        mask = np.load(os.path.join(self.input_path, image_filename))['label'].astype(np.int64)
        image, mask = addjust_shape(image,mask,shape = self.imgshape)       
        # print(image.shape)
        # read mask image
        #mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)
        
        #mask[mask<=127] = 0
        #mask[mask>127] = 1
        # correct dimensions if needed
        assert len(image.shape)<5, 'Input dims error'
        assert len(mask.shape)<4, 'Input dims error'
        #image, mask = correct_dims(image, mask)
        # print(image.shape)
        
        #mask = np.expand_dims(mask, axis=0)

        if self.RandomFlip:
            image,mask = self.RandomFlip(image,mask)

        if self.RandomRotate:
            image,mask = self.RandomRotate(image,mask)

        if self.RandomContrast:
            image = self.RandomContrast(image)
        #image = np.expand_dims(image, axis=0)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2], mask.shape[3])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print(mask.shape)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)

        return image, mask, image_filename

class ImageToImage3D_SmallScale(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str,i,Training = True, shape = (160,224,224), kfold = 5,  RandomFlip: Callable = None, RandomRotate: Callable = None,RandomContrast: Callable = None,one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, '')
        self.output_path = os.path.join(dataset_path, '')
        images_list_all = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask
        self.imgshape = shape
        self.RandomFlip = RandomFlip
        self.RandomRotate = RandomRotate
        self.RandomContrast = RandomContrast
        self.Training = Training
        self.i = i
        self.kfold = kfold
        fold_size = len(images_list_all) // kfold  # 每份的个数:数据总条数/折数（组数）
        val_start = i * fold_size
        if i != kfold - 1:
            val_end = (i + 1) * fold_size
            if Training:
                self.images_list = images_list_all[0:val_start] + images_list_all[val_end:]
            else:
                self.images_list = images_list_all[val_start:val_end]
        else:  # 若是最后一折交叉验证
            if Training:
                self.images_list = images_list_all[val_start:]
            else:
                self.images_list = images_list_all[0:val_start]

        

            
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        # print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        #image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = np.load(os.path.join(self.input_path, image_filename))['image'].astype(np.float32)
        mask = np.load(os.path.join(self.input_path, image_filename))['label'].astype(np.int64)
        image, mask = addjust_shape(image,mask,shape = self.imgshape)       
        # print(image.shape)
        # read mask image
        #mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)
        
        #mask[mask<=127] = 0
        #mask[mask>127] = 1
        # correct dimensions if needed
        assert len(image.shape)<5, 'Input dims error'
        assert len(mask.shape)<4, 'Input dims error'
        #image, mask = correct_dims(image, mask)
        # print(image.shape)
        
        #mask = np.expand_dims(mask, axis=0)

        if self.RandomFlip:
            image,mask = self.RandomFlip(image,mask)

        if self.RandomRotate:
            image,mask = self.RandomRotate(image,mask)

        if self.RandomContrast:
            image = self.RandomContrast(image)
        #image = np.expand_dims(image, axis=0)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2], mask.shape[3])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print(mask.shape)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)

        return image, mask, image_filename

