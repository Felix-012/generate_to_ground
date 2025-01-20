"""from https://github.com/MischaD/chest-distillation"""

import argparse
import os
import shutil
from pathlib import PurePath, Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import EMAModel, UNet2DConditionModel, ControlNetModel
from matplotlib import pyplot as plt
from torchvision.transforms.v2 import Resize, Compose, CenterCrop

from custom_pipe import FrozenCustomPipe
from log import logger
from util_scripts.utils_generic import get_latest_directory

MIMIC_STRING_TO_ATTENTION = {"Atelectasis": ["atelectasis", "atelectatic"],
                             "Cardiomegaly": ["cardiomegaly", "cardiac", ],  # enlarged, heart
                             "Consolidation": ["consolidation", "consolidations", "consolidative", ],
                             "Edema": ["edema", ],
                             "Lung Opacity": ["opacity", "opacities", "opacification"],
                             "Pleural Effusion": ["pleural", "effusion", "effusions"],
                             "Pneumonia": ["pneumonia", ],
                             "Pneumothorax": ["pneumothorax", "pneumothoraces"],
                             }


def compute_prediction_from_binary_mask(binary_prediction):
    """
    Computes the bounding box coordinates and the center of mass from a binary mask of an image.

    :param binary_prediction: A binary mask as a tensor where pixels belonging to the region of interest are marked.
    :return: A tuple containing the updated binary mask with only the bounding box, center of mass coordinates, and the
    bounding box coordinates (x1, x2, y1, y2).
    """

    binary_prediction = binary_prediction.to(torch.bool).numpy()
    horizontal_indicies = np.where(np.any(binary_prediction, axis=0))[0]
    vertical_indicies = np.where(np.any(binary_prediction, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    prediction = np.zeros_like(binary_prediction)
    prediction[y1:y2, x1:x2] = 1
    center_of_mass = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
    return prediction, center_of_mass, (x1, x2, y1, y2)


def contrast_to_noise_ratio(ground_truth_img, prelim_mask_large):
    """
    Calculates the Contrast to Noise Ratio (CNR) between the region of interest in the predicted and ground truth masks.

    :param ground_truth_img: The ground truth image mask as a tensor.
    :param prelim_mask_large: The preliminary mask of the image as a tensor.
    :return: The contrast to noise ratio value.
    """
    gt_mask = ground_truth_img.flatten()
    pr_mask = prelim_mask_large.flatten()

    roi_values = pr_mask[gt_mask == 1.0]
    not_roi_values = pr_mask[gt_mask != 1.0]

    contrast = roi_values.mean() - not_roi_values.mean()
    noise = torch.sqrt(
        roi_values.var() + not_roi_values.var()
    )
    cnr = contrast / noise
    return cnr


def check_mask_exists(mask_dir, samples):
    """
    Checks if mask files exist in a specified directory for a given set of sample paths.

    :param mask_dir: Directory where masks are supposed to be stored.
    :param samples: A dictionary with key 'rel_path' containing relative paths to the masks.
    :return: Boolean indicating whether all masks exist.
    """

    for i in range(len(samples["rel_path"])):
        path = os.path.join(mask_dir, samples["rel_path"][i] + ".pt")
        if not os.path.exists(path):
            return False
    return True


def samples_to_path(mask_dir, samples, j):
    """
    Constructs a path for storing a mask based on sample metadata.

    :param mask_dir: Directory to store the mask.
    :param samples: Dictionary containing sample metadata.
    :param j: Index of the sample to process.
    :return: The full path where the mask should be saved.
    """

    sample_path = samples["rel_path"][j]
    label = samples["finding_labels"][j]
    impr = samples["impression"][j].replace(" ", "_")[:100]
    path = os.path.join(mask_dir, sample_path + label + impr) + ".pt"
    logger.info(f"StoPath: {path}")
    return path


def get_compute_mssim():
    """
    Configures and parses command-line arguments for computing the MS-SSIM (Multi-Scale Structural Similarity Index) for
    datasets.

    :return: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Compute MS-SSIM of dataset")
    parser.add_argument("--n_sample_sets", type=int, default=100)
    parser.add_argument("--trial_size", type=int, default=4)
    parser.add_argument("--use_mscxrlabels", action="store_true", default=False,
                        help="If set, then we use shortned impressions from mscxr")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="dir to save images in. Default will be inside log dir and should be used!")
    parser.add_argument("--config", type=str,
                        help="path to config file")
    return parser.parse_args()


def word_to_slice(label: list, query_words):
    """
    Finds the indices of label words that contain any of the query words.

    :param label: A list of words from a label.
    :param query_words: A list of query words to search within the label.
    :return: A list of indices in the label where the query words are found.
    """

    locations = []
    query_words = [w.lower() for w in query_words]
    label = [w.lower() for w in label]
    for query_word in query_words:
        for i, word in enumerate(label):
            if query_word.lower() in word.lower():
                locations.append(i)
    return locations


def get_args_compute_bbox():
    """
    Sets up and parses command-line arguments for computing localization scores.
    :return: Namespace containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Compute Localization Scores")
    parser.add_argument("--config", type=str, help="Path to the dataset config file")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="dir to save masks in. Default will be inside log dir and should be used!")
    parser.add_argument("--phrase_grounding_mode", action="store_true", default=False,
                        help="If set, then we use shortened impressions from mscxr")
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="If set, then lora weights are used")
    parser.add_argument("--use_ema", action="store_true", default=False,
                        help="If set, then lora weights are used")
    parser.add_argument("--llm_name", type=str, default="",
                        choices=["radbert", "chexagent", "med-kebert", "clip", "openclip", "cxr_clip", "gloria"],
                        help="Name of the llm to use")
    parser.add_argument("--path", type=str, default="", help="Path to the repository or local folder of the pipeline")
    parser.add_argument("--custom_path", type=str, default=None,
                        help="Additional custom path for text encoder and tokenizer")
    parser.add_argument("--from_scratch", action="store_true",
                        help="If set, deletes the old mask directory if it exists")
    parser.add_argument("--use_attention_mask", action="store_true",
                        help="If set, uses attention mask when encoding the texts.")
    parser.add_argument("--control_path", type=str, default=None,
                        help="Path to the control net you wish to evaluate.")
    parser.add_argument("--lora_weights", type=str, default=None,
                        help="Path to the lora weights / checkpoint. Name of the directory should be the rank used for"
                             "training.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the checkpoint.")
    parser.add_argument("--sample_mode", action="store_true", help="If enabled, uses our SamplePipeline.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Specifies the number of timesteps used during image generation")
    parser.add_argument("--rev_diff_steps", type=int, default=40,
                        help="Specifies the last number of timesteps to consider for attention maps")
    parser.add_argument("--guidance_scale", type=float, default=16.0,
                        help="Guidance scale used for unconditional guidance during sampling")
    parser.add_argument("--medklip_preprocessing", action="store_true",
                        help="if set, preprocesses the impressions using knowledge graphs as used in medklip.")
    parser.add_argument("--split", type=str, help="Which dataset and corresponding split should be used. "
                                                  "Should be test or validation usually. Structure your config file"
                                                  "accordingly.", default="test")
    parser.add_argument("--only_first", action="store_true", help="Can only be used with sample_mode!"
                                                                  "If set, the ground truth image will be only fed into"
                                                                  "the model in the first denoising step.")
    parser.add_argument("--layers", nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                        help="Specify the layer numbers that should be used for inference.")
    parser.add_argument("--subtract_attn_images", action="store_true", default=False,
                        help="If set, subtracts negative attention images from positive attention images."
                             "Warning: Only use if you use unconditional guidance!")
    parser.add_argument("--seed", type=int, default=4200, help="random seed to use for image generation.")
    parser.add_argument("--log_biovil", action="store_true", default=False,
                        help="Specifies if phrase grounding images should should be logged, including biovil")
    parser.add_argument("--log", action="store_true", default=False,
                        help="Specifies if phrase grounding images should be logged")
    parser.add_argument("--bbm", action="store_true", default=False,
                        help="If set, uses the new method in the midl paper for evaluation.")
    parser.add_argument("--disease_filtering", action="store_true", default=False,
                        help="If set, filters tokens by disease instead of content words.")

    return parser.parse_args()


def prepare_evaluation(config, args, mapper, lora_weights, trust_remote_code=False, use_ddim=True, rank=0):
    """
    Prepares the evaluation pipeline with specified configurations and models.
    :param use_ddim: If the DDIM scheduler should be used instead of the DDPM scheduler.
    :param trust_remote_code: True if code from the specified remote repository should be executed.
    :param config: Configuration settings.
    :param args: Command-line arguments specifying various options.
    :param mapper: Mapping of model state keys.
    :param lora_weights: Path to lora model weights.
    :param rank: Device rank, used for model parallelism.
    :return: Tuple of the initialized pipeline and mask directory.
    """
    sample_mode = getattr(args, "sample_mode", None)
    control_path = getattr(args, "control_path", None)
    use_lora = getattr(args, "use_lora", None)
    if control_path is not None:
        pipeline = FrozenCustomPipe(path=args.path, custom_path=args.custom_path, llm_name=args.llm_name,
                                        device=torch.device(rank), trust_remote_code=trust_remote_code, control=True,
                                        use_ddim=use_ddim, sample_mode=sample_mode).pipe
        pipeline.controlnet = ControlNetModel.from_pretrained(control_path)
    else:
        pipeline = FrozenCustomPipe(path=args.path, custom_path=args.custom_path, llm_name=args.llm_name,
                                    trust_remote_code=trust_remote_code, device=torch.device(rank),
                                    use_ddim=use_ddim, sample_mode=sample_mode).pipe

    if hasattr(args, "mask_dir"):
        mask_dir = args.mask_dir
    else:
        mask_dir = os.path.join(config.log_dir, "preliminary_masks")

    if use_lora:
        lora_rank = Path(lora_weights).stem
        if not lora_rank.isdigit():
            raise ValueError("The --lora_weights directory should be a number (i.e. the rank used)")
        mask_dir = list(PurePath(mask_dir).parts)
        mask_dir.insert(-1, lora_rank)
        mask_dir = PurePath("").joinpath(*mask_dir)
        checkpoint = get_latest_directory(argparse.Namespace(resume_from_checkpoint="latest", output_dir=lora_weights))
        lora_weights = os.path.join(lora_weights, checkpoint)
        pipeline.load_lora_weights(lora_weights)
    elif args.use_ema and args.checkpoint is not None:
        ema_unet = EMAModel(pipeline.unet.parameters(), model_cls=UNet2DConditionModel,
                            model_config=pipeline.unet.config)
        ema_unet.to("cuda")
        load_model = EMAModel.from_pretrained(os.path.join(args.checkpoint, "unet_ema"), UNet2DConditionModel)
        ema_unet.load_state_dict(load_model.state_dict())
        del load_model
        ema_unet.copy_to(pipeline.unet.parameters())
    elif args.checkpoint is not None and not config.load_lightning:
        pipeline.unet = UNet2DConditionModel.from_pretrained(os.path.join(args.checkpoint, "unet"))
    elif config.load_lightning:
        loaded_state_dict = torch.load(args.checkpoint)["state_dict"]
        prefix = "model.diffusion_model."
        filtered_dict = {}
        for key in loaded_state_dict:
            if key.startswith(prefix):
                # Remove the prefix and add the remaining part of the key to the new dictionary
                new_key = key[len(prefix):]
                filtered_dict[new_key] = loaded_state_dict[key]

        state_dict = {}
        for old_key, new_key in mapper.items():
            state_dict[new_key] = filtered_dict[old_key]
        pipeline.unet.load_state_dict(state_dict, strict=True)

    if args.checkpoint is not None and os.path.isdir(os.path.join(args.checkpoint, "text_encoder")):
        pipeline.text_encoder = (type(pipeline.text_encoder).
                                 from_pretrained(os.path.join(args.checkpoint, "text_encoder")))

    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    return pipeline, mask_dir


def log_biovil(dataset, sample, ground_truth_img, results, configuration, prelim_mask_large, binary_mask_large,
               args, image_text_inference):
    base_filename = None

    # Normalize the raw image
    img = (sample["img_raw"] + 1) / 2
    img = img.cpu().numpy().transpose((1, 2, 0))

    if args.log_biovil:
        similarity_map = image_text_inference.get_similarity_map_from_raw_data(
            image_path=Path(str(os.path.join(dataset.base_dir, sample["rel_path"]))),
            query_text=sample["impression"],
            interpolation="bilinear",
        )
        non_nan_rows = np.any(~np.isnan(similarity_map), axis=1)
        non_nan_cols = np.any(~np.isnan(similarity_map), axis=0)

        # Use these indices to slice the array and remove the NaN frame
        similarity_map = similarity_map[non_nan_rows][:, non_nan_cols]
        transforms = Compose([Resize(img.shape[0]), CenterCrop(img.shape[0])])
        similarity_map = transforms(torch.tensor(similarity_map).unsqueeze(dim=0)).squeeze()
        similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min())
        biovil_cnr = float(contrast_to_noise_ratio(ground_truth_img, similarity_map))
        base_filename = f"bv_cnr_{biovil_cnr:.4f}_our_cnr{results['cnr'][-1]:.4f}:_{sample['finding_labels']}"
        similarity_map = plt.cm.jet(similarity_map)[:, :, :3]
        # Create biovil overlay image
        biovil_overlay = np.add(0.7 * img, 0.3 * similarity_map) * 255
        biovil_overlay = biovil_overlay.astype(np.uint8)
        biovil_overlay = Image.fromarray(biovil_overlay)
        biovil_draw = ImageDraw.Draw(biovil_overlay)
        for bbox in sample["bbox_processed"]:
            x, y, w, h = bbox
            biovil_draw.rectangle([x, y, x + w, y + h], outline="white", width=5)

    # Convert masks to numpy arrays
    prelim_mask_large = plt.cm.jet(prelim_mask_large)[:, :, :3]
    binary_mask_large = binary_mask_large.cpu().numpy().transpose((1, 2, 0))
    # Create overlay image
    overlay = np.add(0.7 * img, 0.3 * prelim_mask_large) * 255
    overlay = overlay.astype(np.uint8)
    overlay = Image.fromarray(overlay)

    # Draw bounding boxes on the overlays
    draw = ImageDraw.Draw(overlay)

    for bbox in sample["bbox_processed"]:
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline="white", width=5)

    # Prepare directories for saving
    if base_filename is None:
        base_filename = f"our_cnr{results['cnr'][-1]:.4f}:_{sample['finding_labels']}"

    path = os.path.join(str(configuration.log_dir), "localization_examples", str(base_filename))

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    logger.info(f"Logging to {path}")

    # Save the images
    img_save_path = os.path.join(str(path), f"{base_filename}_img.png")
    mask_save_path = os.path.join(str(path), f"{base_filename}_binary_mask_large.png")
    overlay_save_path = os.path.join(str(path), f"overlay.png")
    biovil_overlay_save_path = os.path.join(str(path), f"biovil_overlay.png")

    # Convert and save images
    img_to_save = Image.fromarray((img * 255).astype(np.uint8))
    binary_mask_large_to_save = Image.fromarray((binary_mask_large * 255).astype(np.uint8))

    img_to_save.save(img_save_path)
    binary_mask_large_to_save.save(mask_save_path)
    overlay.save(overlay_save_path)
    if args.log_biovil:
        biovil_overlay.save(biovil_overlay_save_path)
