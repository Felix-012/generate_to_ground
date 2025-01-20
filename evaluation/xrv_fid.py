# Code taken and modified from: https://github.com/mseitzer/pytorch-fid

"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
import random

import numpy as np
import pandas as pd
import skimage
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as TF
import torchxrayvision as xrv
from einops import rearrange, repeat
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        """
        Mock version of tqdm.
        :param x: Input, not used.
        :return:
        """
        return x

from log import logger

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading images from file paths with optional transformations.
    """

    def __init__(self, files, transforms=None, channels=3, fid_model="xrv"):
        """
        :param files: A list of file paths for the images.
        :param transforms: Optional transformations to apply to the images.
        :param channels: Number of image channels.
        :param fid_model: Model identifier to select specific preprocessing.
        """

        self.files = files
        self.transforms = transforms
        self.channels = channels
        self.fid_model = fid_model

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.files)

    def __getitem__(self, i):
        """Retrieve an image and apply transformations, returning the transformed image."""
        path = self.files[i]
        img = skimage.io.imread(path)
        if len(img.shape) == 2:
            img = repeat(rearrange(img, "h w -> h w 1"), "h w 1 -> h w c", c=self.channels)
        img = rearrange(img, "h w c -> c h w")
        img = self.transforms(torch.Tensor(img)).clamp(min=0, max=255.)
        if self.fid_model == "xrv":
            img = xrv.datasets.normalize(img.numpy(), 255)  # convert 8-bit image to [-1024, 1024] range
            img = torch.from_numpy(img)
        else:
            img = img / 127.5 - 1.

        if self.channels == 1:
            img = img.mean(dim=0)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1, fid_model="xrv"):
    """
    Calculates the activations of a model for all provided images.

    :param files: List of image file paths.
    :param model: PyTorch model to compute activations.
    :param batch_size: Number of images to process in one batch.
    :param dims: Number of dimensions in the output feature space of the model.
    :param device: Device to perform computations on.
    :param num_workers: Number of worker processes for data loading.
    :param fid_model: Model identifier to select specific preprocessing.
    :return: A numpy array of model activations for all images.
    """

    model.eval()

    if batch_size > len(files):
        print('Warning: batch size is bigger than the data size. Setting batch size to data size')
        batch_size = len(files)

    if fid_model == "xrv":
        logger.info("Resizing and center cropping to 224x224 images")
        transform = torchvision.transforms.Compose([TF.Resize(224), TF.CenterCrop(224)])
        channels = 1
    else:
        logger.info("Resizing and center cropping to 299x299 images")
        transform = torchvision.transforms.Compose([TF.Resize(299), TF.CenterCrop(299)])
        channels = 3

    dataset = ImagePathDataset(files, transforms=transform, channels=channels, fid_model=fid_model)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        if fid_model == "xrv":
            with torch.no_grad():
                pred = model.features2(batch.unsqueeze(dim=1)).detach().cpu()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]
        else:
            with torch.no_grad():
                pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculates the Frechet Distance between two multivariate Gaussians.

    :param mu1: Mean vector of the first distribution.
    :param sigma1: Covariance matrix of the first distribution.
    :param mu2: Mean vector of the second distribution.
    :param sigma2: Covariance matrix of the second distribution.
    :param eps: Small epsilon value to ensure numerical stability.
    :return: The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (f"fid calculation produces singular product, "
               f"adding {eps} to diagonal of cov estimates")
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1, fid_model="xrv"):
    """
    Calculates the mean and covariance of model activations for a given set of image files.

    :param files: List of image file paths.
    :param model: PyTorch model used for activation extraction.
    :param batch_size: Number of images to process in one batch.
    :param dims: Dimensionality of the output feature space of the model.
    :param device: Device to perform computations on.
    :param num_workers: Number of worker processes for data loading.
    :param fid_model: Model identifier to select specific preprocessing.
    :return: Tuple of mean and covariance matrix of activations.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers, fid_model=fid_model)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1, fid_model="xrv"):
    """
    Computes or loads statistics for images at a given path.

    :param path: Path to the image files or to a .npz file containing precomputed statistics.
    :param model: PyTorch model used for activation extraction.
    :param batch_size: Number of images to process in one batch.
    :param dims: Dimensionality of the output feature space of the model.
    :param device: Device to perform computations on.
    :param num_workers: Number of worker processes for data loading.
    :param fid_model: Model identifier to select specific preprocessing.
    :return: Tuple of mean and covariance matrix of activations.
    """
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            files = df["path"].to_list()
            files = [os.path.join(os.path.dirname(path), file) for file in files]
        else:
            path = pathlib.Path(path)
            files = [file for ext in IMAGE_EXTENSIONS
                     for file in path.rglob(f'*.{ext}')]

        random.shuffle(files)
        if len(files) < 5000:
            logger.warning("List of file for path {path} is lower than 5000 - FID may be inconclusive")
        files = files[:5000]

        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers, fid_model)
    return m, s


def calculate_fid_given_paths(paths, batch_size, device, fid_model, model, dims, num_workers=1):
    """
    Calculates the Frechet Inception Distance (FID) between two sets of images.

    :param paths: List of two paths to image sets or precomputed statistics.
    :param batch_size: Number of images to process in one batch.
    :param device: Device to perform computations on.
    :param fid_model: Model identifier to select specific preprocessing.
    :param model: PyTorch model used for activation extraction.
    :param dims: Dimensionality of the output feature space of the model.
    :param num_workers: Number of worker processes for data loading.
    :return: The Frechet Inception Distance between the two image sets.
    """
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError(f"Invalid path: {p}")

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers, fid_model)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers, fid_model)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
