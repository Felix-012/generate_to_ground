"""adapted from https://github.com/MischaD/chest-distillation"""

import os
import numpy as np
import torch
from einops import repeat
from torch.utils.data import Dataset
from xray_datasets.utils import file_to_list, path_to_tensor
from log import logger
from util_scripts.utils_generic import DatasetSplit, SPLIT_TO_DATASETSPLIT


def add_preliminary_to_sample(entry, path):
    """
    Loads preliminary mask from path and adds it to the data entry dictionary.
    :param entry: Dictionary of data entry.
    :param path: Path to the preliminary mask file.
    :return: Modified data entry.
    """
    entry["preliminary_mask"] = torch.load(path)
    return entry


class FOBADataset(Dataset):
    """
    Baseclass for datasets used in foreground - backgroun distillation, i.e. attention mask generation and evaluation.
    """
    def __init__(self, dataset_args, opt):
        """
        :param dataset_args: Dataset subcategory configuration file
        :param opt: The configuration file.
        """

        self.data = None
        self.opt = opt
        self.base_dir = dataset_args["base_dir"]
        self.load_segmentations = False
        self.split = dataset_args["split"]
        self.h = opt.datasets.H
        self.w = opt.datasets.W
        self.preload_deprecated = dataset_args.get("preload", False)
        self.limit_dataset = dataset_args.get("limit_dataset", None)
        self.shuffle = dataset_args.get("shuffle", False)
        self._data = None
        self._preliminary_masks_path = None
        self._inpainted_images_path = None
        self.latent_attention_mask = opt.latent_attention_masks
        self._transform = lambda x: x

    def set_transform(self, func):
        """
        :param func: torchvision transformation function.
        :return:
        """
        self._transform = func

    @property
    def image_list(self):
        """
        :return: Full path to images from base dir and images.txt.
        """
        return os.path.join(self.base_dir, "images.txt")

    def __len__(self):
        """
        :return: Length of data.
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        :param item: Index of data entry to return.
        :return Data entry at the specified index.
        """
        if hasattr(self, "is_precomputed"):
            try:
                ret = self.data[item]
            except TypeError as exc:
                raise TypeError("NoneType object is non subscriptable - did you call load_precomputed?") from exc
            return ret
        return self._load_images([item])

    def _build_dataset(self):
        """
        Loads the correct image paths from the images.txt file and filters them according to their split (e.g. train).
        Also limits the length of the number of paths, shuffles them and loads their segmentations if the option is set
        in config. Use this function to initialize self.data.
        :return:
        """
        image_paths, splits = [], []
        for entry in file_to_list(os.path.join(self.base_dir, "images.txt")):
            entry = entry.strip().split(" ")
            image_paths.append(entry[0])
            splits.append(entry[1])

        data = [{"rel_path": img_path, "img_path": os.path.join(self.base_dir, img_path)} for img_path in image_paths]

        self._get_split(data, splits)

        if self.load_segmentations:
            for i in range(len(self.data)):
                self.data[i]["seg_path"] = os.path.join(self.base_dir, "segmentations",
                                                        self.data[i]["rel_path"][7:])[:-3] + "png"

        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.data)

        if self.limit_dataset is not None:
            self.data = self.data[self.limit_dataset[0]:min(self.limit_dataset[1], len(self.data))]

    def _get_split(self, data, splits):
        """
        Creates splits of the data.
        :param data: Dictionary of data entries.
        :param splits: List that specifies which split each item belongs to.
        :return:
        """
        if self.split == DatasetSplit.all:
            split_data = data
        else:
            split_data = []
            for dataobj, split in zip(data, splits):
                if SPLIT_TO_DATASETSPLIT[int(split)].value == self.split:
                    split_data.append(dataobj)
        self.data = split_data

    def _load_images(self, index):
        """
        Loads the data entry at the specified index and adds the image from the image path, as well as optional
        segmentation and mask information.
        :param index: Index for self.data. Specifies which data entry should be modified.
        :return: The loaded and modified data entry.
        """

        entry = self.data[index].copy()
        img = path_to_tensor(entry["img_path"])
        entry["img"] = img

        assert img.size()[2] == 256 and img.size()[3] == 256
        x = repeat(img, "1 c h w -> 1 c (h h2) (w w2)", h2=2, w2=2)

        entry["x"] = x
        entry["slice"] = (slice(None), slice(None), slice(None), slice(None))

        if self.load_segmentations:
            seg = (path_to_tensor(entry["seg_path"]) + 1) / 2
            entry["segmentation"] = seg
            y = torch.full((1, 3, self.h, self.w), 0.)
            y[0, :, :seg.size()[2], :seg.size()[3]] = seg
            entry["segmentation_x"] = y

        if self._preliminary_masks_path is not None:
            entry["preliminary_mask"] = torch.load(
                os.path.join(self._preliminary_masks_path, entry["rel_path"] + ".pt"))
            if not self.latent_attention_mask:
                entry["preliminary_mask"] = repeat(entry["preliminary_mask"], "1 1 c h w -> 1 1 c (h h2) (w w2)",
                                                   h2=self.opt.datasets.f, w2=self.opt.datasets.f)

        if self._inpainted_images_path is not None:
            entry["inpainted_image"] = torch.load(os.path.join(self._inpainted_images_path, entry["rel_path"] + ".pt"))

        if entry["inpainted_image"].sum() < 100:
            logger.warn("Redo sampling for missing labels")
            return self._load_images(np.random.randint(len(self)))
        return entry

    def add_preliminary_masks(self, base_path=None, sanity_check=True):
        """
        Adds the path to the preliminary masks to the class. Optionally does a sanity check if masks exist.
        :param base_path: Base path to the preliminary mask directory.
        :param sanity_check: If set to True, a sanity check is performed if mask files exist.
        :return:
        """
        if base_path is not None:
            self._preliminary_masks_path = base_path
        else:
            self._preliminary_masks_path = self.opt.out_dir
        # sanity check
        if sanity_check:
            for i in [0, len(self.data) - 1]:
                attention_mask_file = os.path.join(self._preliminary_masks_path, self.data[i]["rel_path"] + ".pt")
                assert os.path.isfile(attention_mask_file), f"File not Found: {attention_mask_file}"
        logger.info("Sanity check complete! All attention masks computed.")

    def add_inpaintings(self, exp_name):
        """
        :param exp_name: name of experiment == extension to base_dir leading to refined masks
        :return:
        """
        self._inpainted_images_path = os.path.join(self.opt.base_dir, exp_name)
        for i in [0, len(self.data) - 1]:
            path_sample = os.path.join(self._inpainted_images_path, self.data[i]["rel_path"]) + ".pt"
            assert os.path.isfile(path_sample), f"Inpainted Image not Found: {path_sample}"

        logger.info("Sanity check complete! All Inpaintings Computed.")
