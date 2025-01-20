"""adapted from https://github.com/MischaD/chest-distillation"""

import argparse
import os
from enum import Enum

import numpy as np
import torch
import torchvision
import imageio

from PIL import Image, ImageDraw
from einops import rearrange
from matplotlib import pyplot as plt


class DatasetSplit(Enum):
    """
    Enumeration for different dataset splits used to categorize data processing workflows.
    """
    train = "train"
    test = "test"
    val = "val"
    mscxr = "mscxr"
    p19 = "p19"
    all = "all"


def resize_long_edge(img, size_long_edge):
    """
    Resizes an image so that its longest edge is equal to the specified length.
    :param img: Input image tensor.
    :param size_long_edge: Desired length of the longest edge after resizing.
    :return: Resized image tensor.
    """
    # torchvision resizes so shorter edge has length - I want longer edge to have spec. length
    assert img.size()[-3] == 3, "Channel dimension expected at third position"
    img_longer_edge = max(img.size()[-2:])
    img_shorter_edge = min(img.size()[-2:])
    resize_factor = size_long_edge / img_longer_edge

    # resized_img = torchvision.transforms.functional.resize(img_longer_edge/img_shorter_edge)
    resize_to = img_shorter_edge * resize_factor
    resizer = torchvision.transforms.Resize(size=round(resize_to))
    return resizer(img)[..., :size_long_edge, :size_long_edge]


SPLIT_TO_DATASETSPLIT = {0: DatasetSplit("test"), 1: DatasetSplit("train"), 2: DatasetSplit("val"),
                         3: DatasetSplit("p19"), 4: DatasetSplit("mscxr")}


def collate_batch(batch):
    """
    Collates a batch of data into a batched format suitable for model input.
    :param batch: List of data samples.
    :return: Dictionary with batched data.
    """
    # make list of dirs to dirs of lists with batchlen
    batched_data = {}
    for data in batch:
        # label could be img, label, path, etc
        for key, value in data.items():
            if batched_data.get(key) is None:
                batched_data[key] = []
            batched_data.get(key).append(value)

    # cast to torch.tensor
    for key, value in batched_data.items():
        if isinstance(value[0], torch.Tensor):
            if value[0].size()[0] != 1:
                for i in range(len(value)):
                    value[i] = value[i][None, ...]
            # check if concatenatable
            if all(value[0].size() == value[i].size() for i in range(len(value))):
                batched_data[key] = torch.concat(batched_data[key])
    return batched_data


def img_to_viz(img):
    """
    Converts a tensor image to a visualizable format.
    :param img: Tensor image.
    :return: Numpy array suitable for visualization.
    """
    img = rearrange(img, "1 c h w -> h w c")
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = np.array(((img + 1) * 127.5), np.uint8)
    return img


def get_args_parameter_search():
    """
    ets up and parses command-line arguments for finding the best layer combination in evaluation.
    :return: Namespace containing the parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Find best layer combination")
    parser.add_argument("--config", type=str, help="Path to the dataset config file")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="dir to save masks in. Default will be inside log dir and should be used!")
    parser.add_argument("--use_ema", action="store_true", default=False,
                        help="If set, then lora weights are used")
    parser.add_argument("--llm_name", type=str, default="",
                        choices=["radbert", "chexagent", "med-kebert", "clip", "openclip", "cxr_clip"],
                        help="Name of the llm to use")
    parser.add_argument("--path", type=str, default="", help="Path to the repository or local folder of the pipeline")
    parser.add_argument("--custom_path", type=str, default=None,
                        help="Additional custom path for text encoder and tokenizer")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Specifies the number of timesteps used during image generation")
    parser.add_argument("--guidance_scale", type=float, default=16.0,
                        help="Guidance scale used for unconditional guidance during sampling")
    parser.add_argument("--split", type=str, help="Which dataset and corresponding split should be used. "
                                                  "Should be test or validation usually. Structure your config file"
                                                  "accordingly.", default="test")
    parser.add_argument("--num_iterations", type=int, default=None, help="Number of iterations to perform"
                                                                         "before doing early stopping")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the checkpoint.")
    parser.add_argument("--use_attention_mask", action="store_true",
                        help="If set, uses attention mask when encoding the texts.")
    parser.add_argument("--sample_mode", action="store_true", help="If enabled, uses our SamplePipeline.")

    return parser.parse_args()

def get_latest_directory(args):
    """
    :param args: arguments passed to the training script. Should contain the attribute resume_from_checkpoint.
    :return: The latest checkpoint directory.
    """
    if args.resume_from_checkpoint != "latest":
        return os.path.basename(args.resume_from_checkpoint)
    # Get the most recent checkpoint
    dirs = os.listdir(os.path.expandvars(args.output_dir))
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    return dirs[-1] if len(dirs) > 0 else None


def normalize_and_scale_tensor(tensor, scale=True):
    """
    Normalize and scale a float16 tensor to the range 0-255.
    :param tensor: A float16 tensor with arbitrary scale.
    :param scale: If the tensor should be scaled.
    :return: torch.Tensor: A uint8 tensor scaled to 0-255.
    """

    # Ensure the input tensor is float16 for consistent processing
    tensor = tensor.type(torch.float16)

    # Find the minimum and maximum values
    min_val = tensor.min()
    max_val = tensor.max()

    # Normalize the tensor to the range 0-1
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    # Scale to the range 0-255
    if scale:
        scaled_tensor = normalized_tensor * 255
        normalized_tensor = scaled_tensor.to(torch.uint8)

    return normalized_tensor

def vis(tensor):
    plt.imshow(tensor)
    plt.axis('off')
    plt.show()



def tensors_to_gif(tensors, bbox_tensor, output_path, duration=1500):
    """
    Convert a list of image tensors and a bounding box tensor into a GIF.

    Args:
    - tensors (list of numpy arrays): List of image tensors (each tensor is assumed to be a numpy array).
    - bbox_tensor (numpy array): A binary mask tensor with the same height and width as the images.
    - output_path (str): The path where the gif will be saved.
    - duration (int): Duration between frames in milliseconds (default: 100ms).

    Returns:
    - None. Saves the GIF at the specified path.
    """

    frames = []

    for i, tensor in enumerate(tensors):
        # Normalize tensor to 0-255 if it isn't already
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        tensors[i] = (tensor * 255).astype(np.uint8)

    for i, tensor in enumerate(tensors):
        if i < tensors.shape[0] - 1:
            tensors[i] = tensors[i+1] - tensor

    for i, tensor in enumerate(tensors):
        # Convert the image tensor to a PIL image
        img = Image.fromarray(tensors[i])

        # Create a drawing context for the bbox
        draw = ImageDraw.Draw(img)
        for j in range(len(bbox_tensor)):
            x,y,w,h = bbox_tensor[j]
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

        # Append the image to frames
        frames.append(img)

    # Save the frames as a gif
    imageio.mimsave(output_path, frames, duration=duration / 1000)
