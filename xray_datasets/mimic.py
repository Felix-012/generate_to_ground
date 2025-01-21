"""addapted from https://github.com/MischaD/chest-distillation"""
import hashlib
import os
import pickle
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, Compose
from tqdm import tqdm

from xray_datasets.dataset import FOBADataset
from xray_datasets.utils import path_to_tensor
from log import logger
from util_scripts.utils_generic import DatasetSplit


class MimicCXRDataset(FOBADataset):
    """
    Mimic dataset used for training.
    """

    def __init__(self, dataset_args, opt):
        """
        :param dataset_args: Dataset subcategory configuration file
        :param opt: The configuration file.
        """

        super().__init__(dataset_args, opt)
        self.data = None
        self._meta_data = None
        self._csv_file = "mimic_metadata_preprocessed.csv"
        if dataset_args.get("dataset_csv") is not None:
            self._csv_file = os.path.expandvars(dataset_args.get("dataset_csv"))
        self.precomputed_base_dir = os.path.expandvars(dataset_args.get("precomputed_base_dir"))
        self._build_dataset()
        self.opt = opt
        self._precomputed_path = None
        self._save_original_images = dataset_args.get("save_original_images", False)
        self.text_label_key = dataset_args.get("text_label_key", "impression")
        self.chunk_size = None
        self.num_chunks = dataset_args.get("num_chunks")
        self.current_chunk_index = -1
        self.chunk_path = os.path.expandvars(dataset_args.get("chunk_path"))
        self.limit_dataset = dataset_args.get("limit_dataset")
        self.chunk_load_counter = 0
        if self.num_chunks:
            self.chunk_indices = list(range(self.num_chunks))
        random.seed(4200)

    @property
    def precomputed_path(self):
        """
        Uses a hash based on the data for the precomputed directory path.
        :return: The path to the directory where the precomputed latents should be stored.
        """
        if self._precomputed_path is None:
            name = "".join([x["rel_path"] for x in self.data])
            name = hashlib.sha1(name.encode("utf-8")).hexdigest()
            precompute_path = os.path.join(os.path.expandvars(self.precomputed_base_dir), str(name))
            self._precomputed_path = precompute_path
        return self._precomputed_path

    @property
    def is_precomputed(self):
        """
        :return: Returns if the directory with the precomputed latents exists.
        """
        return os.path.isdir(self.precomputed_path)

    def load_precomputed(self, model):
        """
        Loads the precomputed latents from the precomputed path if they exist. If not, they are precomputed first.
        :param model: An autoencoder. Needs an encode() function.
        :return:
        """
        logger.info(f"Using precomputed dataset with name: {self.precomputed_path}")
        if not self.is_precomputed:
            logger.info(f"Precomputed dataset not found - precomputing it on my own: {self.precomputed_path}")
            self.precompute(model)
        with open(os.path.join(self.precomputed_path, "entries.pkl"), "rb") as entry_file:
            entries = pickle.load(entry_file)
        dir_list = os.listdir(self.precomputed_path)
        for file in dir_list:
            if not file.endswith(".pt"):
                continue
            tensor_key = os.path.basename(file.rstrip(".pt"))
            entries[tensor_key] = torch.load(os.path.join(self.precomputed_path, file))

        self.data = []
        if isinstance(entries, dict):
            for i in range(len(entries["dicom_id"])):
                self.data.append({k: entries[k][i] for k in entries.keys()})
        else:
            self.data = entries

    def load_chunk(self, chunk_index):
        """
        Loads the data chunk with the specified index into self.data.
        :param chunk_index: Index of the data chunk.
        :return:
        """
        if chunk_index == self.current_chunk_index:
            return  # No need to load if it's already the current chunk
        filename = f"entries_part{chunk_index}.pkl"
        with open(os.path.join(os.path.expandvars(self.chunk_path), filename), "rb") as f:
            entries = pickle.load(f)
        if isinstance(entries, dict):
            self.data = [{k: entries[k][i] for k in entries.keys()} for i in range(len(entries['rel_path']))]
        else:
            self.data = entries
        self.current_chunk_index = chunk_index
        self.chunk_size = len(self.data)
        logger.info(f"loaded chunk {chunk_index} with size: {self.chunk_size}")

    def compute_latent(self, img, model):
        """
        Preprocessing. Img is already 512x512 tensor 1xCx512x512 --> compute latent using vqvae -
        saves Gaussian parameters
        :param img: Image that should be converted into a latent
        :param model: An autoencoder.
        :return
        """

        img = img.to(model.device)
        encoder_posterior = model.encode(img)
        return encoder_posterior

    def sample_latent(self, encoder_posterior, scale_factor):
        """
        :param encoder_posterior: Diagonal Gaussian distribution that can be sampled from.
        :param scale_factor: Scale factor of the auto encoder by which the latents should be scaled.
        :return: Latent sampled from the distribution.
        """
        z = encoder_posterior.sample()
        return z * scale_factor

    def decode_from_latent(self, encoder_posterior, model):
        """
        Helper function to decode latent space of vqvae
        :param encoder_posterior: Diagonal Gaussian distribution that can be sampled from created by the vae.
        :param model: The vae used to encoder the latents.
        :return: Decoded image.
        """

        n, c, _, _ = encoder_posterior.size()
        assert encoder_posterior.ndim == 4 and n == 1
        old_device = encoder_posterior.device
        encoder_posterior = encoder_posterior.to("cuda")

        if c == 8:
            # params for latent gaussian
            z = self.sample_latent(encoder_posterior, model.scale_factor)
        elif c == 4:
            # sampled values
            z = encoder_posterior
        else:
            raise ValueError(f"Unable to interpret encoder_posterior of shape: {encoder_posterior.size()}")
        img = model.decode_first_stage(z).detach()
        img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
        return img.to(old_device)

    def precompute(self, model):
        """
        Precomputes the latents using the model.
        :param model: A vae.
        :return:
        """
        # load entries
        entries = {}
        if self._save_original_images:
            entries["img_raw"] = []
        if hasattr(self.opt, "control_cond_path"):
            entries["control"] = []
        j = 0
        for i in tqdm(range(len(self)), "Precomputing Dataset"):
            try:
                entry = self._load_images(j)
            except FileExistsError:
                print(f"skipping {self.data[j]['rel_path']} - file does not exist")
                del self.data[j]
                continue
            for k in entry.keys():
                if entries.get(k) is None:
                    assert i == 0
                    entries[k] = []
                entries[k].append(entry[k])

            # preprocess --> 1 x 8 x 64 x 64 diag gaussian latent
            z = self.compute_latent(entry["img"], model)
            if self._save_original_images:
                entries["img_raw"].append(entry["img"])
            if hasattr(self.opt, "control_cond_path") and self.opt.control_cond_path is None:
                if hasattr(self.opt, "control_preprocessing_type"):
                    entries["control"].append(
                        self.preprocess_control(entries["img"][j], self.opt.control_preprocessing_type))
            entries["img"][j] = z
            j += 1
        if hasattr(self.opt, "control_cond_path") and self.opt.control_cond_path is not None:
            if hasattr(self.opt, "control_preprocessing_type"):
                control_preprocessing_type = self.opt.control_preprocessing_type
            else:
                control_preprocessing_type = None
            if not entries["control"]:
                entries = self.load_control_conditioning(entries, self.opt.control_cond_path,
                                                         control_preprocessing_type)

        # save entries
        entry_keys = list(entries.keys())
        data_tensors = {}
        for key in entry_keys:
            if isinstance(entries[key][0], torch.Tensor):
                data_tensors[key] = torch.stack(entries.pop(key))

        path = self.precomputed_path
        logger.info(f"Saving precomputed dataset to: {path}")
        os.makedirs(path)
        with open(os.path.join(path, "entries.pkl"), "wb") as entries_file:
            pickle.dump(entries, entries_file)
        for key in data_tensors:
            torch.save(data_tensors[key], os.path.join(path, f"{key}.pt"))

    @property
    def meta_data_path(self):
        """
        :return: Path to the csv file containing the meta data of the dataset.
        """
        return os.path.join(os.path.expandvars(self.precomputed_base_dir), self._csv_file)

    @property
    def meta_data(self):
        """
        :return: Returns the meta_data contained in the csv_file of the dataset.
        """
        if self._meta_data is None:
            logger.info(f"Loading image list from {os.path.expandvars(self.meta_data_path)}")
            self._meta_data = pd.read_csv(os.path.expandvars(self.meta_data_path), index_col="dicom_id")
            return self._meta_data

        return self._meta_data

    def _build_dataset(self):
        """
        Loads the correct image paths from the metadata csv file and filters them according to their split (e.g. train).
        Also limits the length of the number of paths and shuffles them if the option is set in config.
        Use this function to initialize self.data.
        :return:
        """
        path = "path"
        if "rel_path" in self.meta_data:
            path = "rel_path"
        try:
            filtered_meta_data = self.meta_data.dropna(subset=['path', 'Finding Labels'])
            paths = filtered_meta_data['path'].to_list()
            labels = filtered_meta_data['Finding Labels'].to_list()
            data = [{"rel_path": os.path.join(img_path.replace(".dcm", ".jpg")), "finding_labels": label}
                    for img_path, label in zip(paths, labels)]
        except KeyError:
            filtered_meta_data = self.meta_data.dropna(subset=[path])
            paths = filtered_meta_data[path].to_list()
            data = [{"rel_path": os.path.join(img_path.replace(".dcm", ".jpg"))} for img_path in paths]

        try:
            splits = self.meta_data["split"].astype(int)
            self._get_split(data, splits)
        except KeyError:
            self.data = data

        self.data = data if not self.data else self.data

        if self.shuffle:
            np.random.shuffle(np.array(self.data))

        if self.limit_dataset is not None:
            self.data = self.data[self.limit_dataset[0]:min(self.limit_dataset[1], len(self.data))]

    def load_image(self, img_path):
        """
        Loads image from image path as pytorch tensor and resizes it if it is too large.
        :param img_path: Path to image file.
        :return: Loaded image tensor.
        """
        img = path_to_tensor(img_path)
        # images too large are resized to self.W^2 using center cropping
        if max(img.size()) > self.w:
            transforms = Compose([Resize(self.w), CenterCrop(self.w)])
            img = transforms(img)
        return img

    def _load_images(self, index):
        """
        Loads the data entry at the specified index and adds the image from the image path, as well as the corresponding
        impression and dicom_id.
        :param index: Index for self.data. Specifies which data entry should be modified.
        :return: The loaded and modified data entry.
        """
        entry = self.data[index].copy()
        entry["dicom_id"] = os.path.basename(entry["rel_path"]).rstrip(".jpg")
        img_path = os.path.join(self.base_dir, entry["rel_path"].replace(".dcm", ".jpg"))
        entry["img"] = self.load_image(img_path)
        entry["impression"] = self.meta_data.loc[entry["dicom_id"]]["impression"]
        return entry

    def load_control_conditioning(self, entries, control_cond_path, control_preprocessing_type):
        """
        Loads the control image from the specified path,  applies the specified preprocessing on it and adds it to the
        passed entries.
        :param entries: Dictionary of data entries where the control image should be added.
        :param control_cond_path: Path to the directory containing the conditioning images.
        :param control_preprocessing_type: Specifies the preprocessing type for the conditioning images. Only 'canny'
               is supported as of yet.
        :return: The given entries with the additional preprocessed conditioning images.
        """
        for i in tqdm(range(len(entries)), "Processing control conditioning"):
            control = self.load_image(control_cond_path)
            if control_preprocessing_type:
                control = self.preprocess_control(control, control_preprocessing_type)
            entries[i]["control"] = control
        return entries

    def preprocess_control(self, control, control_preprocessing_type):
        """
        Preprocesses given images according to the passed type.
        :param control: List of images to preprocess.
        :param control_preprocessing_type: The preprocessing type. Only 'canny' is supported as of yet.
        :return: Preprocessed images.
        """
        if control_preprocessing_type != "canny":
            raise NotImplementedError("Only canny preprocessing is implemented for control conditioning")
        if torch.is_tensor(control):
            control = cv2.cvtColor(control.numpy().squeeze().transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
            control = np.round((control + 1) * 255 / 2).astype(np.uint8)
        control = cv2.medianBlur(control, 5)
        control = cv2.Canny(control, np.median(control) * 0.4, np.median(control) * 0.3)
        control = control[:, :, None]
        control = np.concatenate([control, control, control], axis=2)
        control = Image.fromarray(control)
        # control = control.resize((64, 64))
        return control

    def load_next_chunk(self):
        """
        If chunked data is used, this function will specify a random order of chunks and loads them accordingly.
        The order is reshuffled after each cycle.
        :return:
        """
        if self.chunk_load_counter >= len(self.chunk_indices):  # If all chunks have been loaded once
            random.shuffle(self.chunk_indices)  # Reshuffle the list
            self.chunk_load_counter = 0  # Reset the counter
        next_chunk_index = self.chunk_indices[self.chunk_load_counter]
        self.load_chunk(next_chunk_index + 1)
        self.chunk_load_counter += 1

    def __getitem__(self, idx):
        """
        Returns the data entry ad the specified index. Includes some optional processing of the impression, mainly for
        legacy reasons.
        :param idx: Index of the data item.
        :return: Data item at index.
        """
        ret = self.data[idx]
        # Apply your custom logic for the text_label_key
        if self.text_label_key in ret:
            if isinstance(ret[self.text_label_key], float):
                ret["impression"] = ""
            else:
                finding_labels = ret[self.text_label_key].split("|")
                ret["impression"] = " ".join(random.sample(finding_labels, len(finding_labels)))
        return ret


class MimicCXRDatasetMSBBOX(MimicCXRDataset):
    """
    Wrapper class for the MS-CXR dataset. This dataset is primarily used for evaluating phrase grounding, since it
    contains bounding boxes for the diseases and improved impressions.
    """

    def __init__(self, dataset_args, opt):
        """
        :param dataset_args: Dataset subcategory configuration file
        :param opt: The configuration file.
        """
        self._bbox_meta_data = None
        self._csv_name = "mcxr_with_impressions.csv"
        assert dataset_args["split"] == DatasetSplit("mscxr").value
        if dataset_args.get("phrase_grounding", False):
            logger.info("Phrase grounding mode on in MSBBOX Dataset")
            self._csv_name = "mimi_scxr_phrase_grounding_preprocessed.csv"

        super().__init__(dataset_args, opt)

    @property
    def bbox_meta_data(self):
        """
        :return: The dicom_ids of the dataset.
        """
        return pd.read_csv(os.path.join(self.base_dir, self._csv_name), index_col="dicom_id")

    def _build_dataset(self):
        """
        Loads the correct image paths and bbox paths from the metadata csv files and filters them according to their
        split (e.g. train).
        Also limits the length of the number of paths and shuffles them if the option is set in config.
        Use this function to initialize self.data.
        :return:
        """
        data = [{"dicom_id": dicom_id, "rel_path": os.path.join(img_path.replace(".dcm", ".jpg")),
                 "finding_labels": labels}
                for img_path, labels, dicom_id in
                zip(list(self.bbox_meta_data.paths), list(self.bbox_meta_data["category_name"]),
                    self.bbox_meta_data.index)]
        self.data = data
        if self.shuffle:
            np.random.shuffle(np.array(self.data))

        if self.limit_dataset is not None:
            self.data = self.data[self.limit_dataset[0]:min(self.limit_dataset[1], len(self.data))]


    def _load_images(self, index):
        """
        Loads the data entry at the specified index and adds the image from the image path, as well as the corresponding
        bounding box ground truth image + data, the impressions (or label_text) and category name.
        :param index: Index for self.data. Specifies which data entry should be modified.
        :return: The loaded and modified data entry.
        """

        entry = self.data[index].copy()
        entry["img"] = self.load_image(os.path.join(self.base_dir, entry["rel_path"]
                                                    .replace(".dcm", ".jpg")))

        meta_data_entry = self.bbox_meta_data.loc[entry["dicom_id"]]
        if isinstance(meta_data_entry, pd.DataFrame):
            meta_data_entry = meta_data_entry[meta_data_entry["category_name"] == entry["finding_labels"]]
            assert len(meta_data_entry) == 1
            meta_data_entry = meta_data_entry.iloc[0]

        image_width, image_height = meta_data_entry[["image_width", "image_height"]]
        bboxes = meta_data_entry["bboxxywh"].split("|")
        bbox_img = torch.zeros(image_height, image_width, dtype=torch.bool)
        processed_bboxes = []

        for bbox in bboxes:
            bbox = bbox.split("-")
            bbox = tuple(map(lambda y_val: int(y_val), bbox))
            processed_bboxes.append(bbox)
            x, y, w, h = bbox
            bbox_img[y: (y + h), x:(x + w)] = True

        if max(bbox_img.size()) > self.w:
            transforms = Compose([Resize(self.w), CenterCrop(self.w)])
            bbox_img = transforms(bbox_img.unsqueeze(dim=0)).squeeze()

        for i, bbox in enumerate(processed_bboxes):
            bboxes[i] = self.adjust_bbox(bbox, original_size=[image_width, image_height])


        entry["bbox_img"] = bbox_img
        entry["bboxxywh"] = meta_data_entry["bboxxywh"]
        entry["label_text"] = meta_data_entry["label_text"]
        entry["category_name"] = meta_data_entry["category_name"]
        entry["bbox_processed"] = bboxes
        return entry

    def adjust_bbox(self, bbox, original_size, new_size=(512, 512)):
        original_width, original_height = original_size
        new_width, new_height = new_size
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        x, y, w, h = bbox
        x = x * scale_x
        y = y * scale_y
        w = w * scale_x
        h = h * scale_y

        # Adjust to center crop
        left_offset = (new_width - 512) / 2
        top_offset = (new_height - 512) / 2

        x -= left_offset
        y -= top_offset

        return [x, y, w, h]

