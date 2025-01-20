"""adapted from https://github.com/MischaD/chest-distillation"""

import os.path

import cv2
import torch
import yaml
from einops import rearrange
from ml_collections import ConfigDict


def path_to_tensor(path, normalize=True):
    """
    :param path: Path to an image file.
    :param normalize: If the image should be normalized to 0-255
    :return: Pytorch tensor of the shape (1 c h w)
    """
    if not os.path.isfile(path):
        raise FileExistsError
    img = torch.tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), dtype=torch.float32)
    if normalize:
        img = (img / 127.5) - 1
    img = rearrange(img, "h w c -> 1 c h w")
    return img


def file_to_list(path):
    """
    :param path: Path to a file.
    :return: Lines contained in the file as list.
    """
    with open(path, 'r', encoding="utf-8") as fp:
        lines = fp.readlines()
    return lines


def load_config(filename: str) -> ConfigDict:
    """
    :param filename:  Path to a configuration file.
    :return: ConfigDict created from the given config file path.
    """
    with open(os.path.expandvars(filename), 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return ConfigDict(data, type_safe=False)
