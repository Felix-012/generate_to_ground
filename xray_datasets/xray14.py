"""
File for the bbox subset of chestxray14, used as a validation set for optimizing parameters in this project.
"""

import os

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from xray_datasets.utils import path_to_tensor


class ChestXRay14BboxDataset(Dataset):
    """
    Main wrapper class for the bbox subset of chestxray14.
    """

    def __init__(self, opt, dataset_args=None):
        """
        :param opt: Dictionary containing the attribute images_path (path to the image directory).
        :param dataset_args: Namespace object containing dataset_csv, i.e. the path to the csv file containing the
                             bbox metadata.
        """
        super().__init__()
        self.bbox_csv = pd.read_csv(dataset_args.dataset_csv)
        self.images_path = os.path.expandvars(opt.cx14_images_path)
        self.data = {"img": [], "bbox_img": [], "bboxxywh": [], "label_text": [], "rel_path": []}

    def process_samples(self, vae):
        """
        Loads bbox metadata and precomutes the image latents.
        :param vae: Variational auto encoder used to compute the latents.
        :return:
        """
        vae.requires_grad_(False)
        image_names = self.bbox_csv["Image Index"]
        for image_name in image_names:
            path = os.path.join(self.images_path, image_name)
            image = path_to_tensor(path).to(vae.device)
            image = torch.tensor(cv2.resize(np.array(image.cpu().squeeze()).transpose(1, 2, 0), (512, 512),
                                            interpolation=cv2.INTER_AREA).transpose((2, 0, 1)))
            encoder_posterior = vae.encode(image.to(device=vae.device).unsqueeze(dim=0))
            self.data["img"].append(encoder_posterior)
            self.data["rel_path"].append(image_name)

        for label in self.bbox_csv["Finding Label"]:
            self.data["label_text"].append(label)

        bboxx = self.bbox_csv['Bbox [x']
        bboxy = self.bbox_csv['y']
        bboxw = self.bbox_csv['w']
        bboxh = self.bbox_csv['h]']

        for x, y, w, h in zip(bboxx, bboxy, bboxw, bboxh):
            self.data["bboxxywh"].append(f"{x}-{y}-{w}-{h}")
            bbox_image = create_bbox_mask(x, y, w, h, 1024)
            self.data["bbox_img"].append(cv2.resize(bbox_image, (512, 512), interpolation=cv2.INTER_AREA))

    def __len__(self):
        """
        :return: Length of the data.
        """
        return len(self.data["img"])

    def __getitem__(self, index):
        """
        :param index: Index of the data item.
        :return: Dictionary with the value retrieved via index for each key in the data.
        """
        return {"img": self.data["img"][index],
                "bbox_img": self.data["bbox_img"][index],
                "bboxxywh": self.data["bboxxywh"][index],
                "label_text": self.data["label_text"][index],
                "finding_labels": self.data["label_text"][index],
                "rel_path": self.data["rel_path"][index]}

    def __getitems__(self, indices):
        """
        :param possibly_batched_indices: A list of tuple_iterators or indices.
        :return: A list of dictionaries, each containing the values retrieved via index for each key in the data.
        """
        return [{"img": self.data["img"][index],
                 "bbox_img": self.data["bbox_img"][index],
                 "bboxxywh": self.data["bboxxywh"][index],
                 "label_text": self.data["label_text"][index],
                 "finding_labels": self.data["label_text"][index],
                 "rel_path": self.data["rel_path"][index]}
                for index in indices]


def create_bbox_mask(x, y, w, h, image_size):
    """
    Creates a mask tensor with 1s inside the specified bounding box (bbox)
    and 0s elsewhere.

    :param x: The x-coordinate (column) of the top-left corner of the bbox.
    :param y: The y-coordinate (row) of the top-left corner of the bbox.
    :param w: The width of the bbox.
    :param h : The height of the bbox.
    :param image_size: The size of the square image/tensor.
    :return mask: A tensor with 1s inside the bbox and 0s outside.
    """
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    # Calculate the bottom-right coordinates of the bbox
    x_end = x + w
    y_end = y + h
    # Ensure the bbox coordinates are within the image boundaries
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(image_size, x_end)
    y_end = min(image_size, y_end)
    # Set the region inside the bbox to 1
    mask[int(y_start):int(y_end), int(x_start):int(x_end)] = 1
    return mask
