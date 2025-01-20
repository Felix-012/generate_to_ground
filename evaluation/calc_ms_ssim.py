"""addapted from https://github.com/MischaD/chest-distillation"""

import datetime
import json
import logging
import os
import time

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast

from custom_pipe import FrozenCustomPipe
from xray_datasets import get_dataset
from xray_datasets.utils import load_config
from evaluation.mssim import calc_ms_ssim_for_path_ordered, get_mscxr_synth_dataset
from evaluation.utils_evaluation import get_compute_mssim
from log import formatter as log_formatter
from log import logger


def main(arguments):
    """
    Computes the MS-SIM score of the dataset.
    :param arguments: Arguments passed to get_compute_mssim().
    :return:
    """
    # mean, sd = calc_ms_ssim_for_path(opt.path, n=opt.n_samples, trials=opt.trials)
    if not hasattr(arguments, "img_dir") or arguments.img_dir is None:
        img_dir = os.path.join(arguments.log_dir, "ms_ssim")
    else:
        img_dir = arguments.img_dir

    logger.info(f"Saving Images to {img_dir}")
    config = load_config(arguments.config)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)
        model = FrozenCustomPipe(path=config.component_dir).pipe

        device = torch.device("cuda")
        model = model.to(device)

        if arguments.use_mscxrlabels:
            dataset = get_dataset(config, "test")
            dataset.load_precomputed(model.vae)
            synth_dataset, _ = get_mscxr_synth_dataset(config, dataset)
        else:
            dataset = get_dataset(config, "testp19")
            dataset.load_precomputed(model.vae)
            synth_dataset, _ = get_mscxr_synth_dataset(config, dataset, finding_key="impression",
                                                       label_key="finding_labels")

        actual_batch_size = arguments.trial_size
        batch_size = 1

        seed_everything(int(time.time()))

        batched_dataset = [synth_dataset[i:i + batch_size] for i in range(0, len(synth_dataset), batch_size)]
        prompt_list = set()
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    for sample_num in range(len(batched_dataset)):
                        samples = batched_dataset[sample_num]
                        prompts = [list(x.values())[0] for x in samples]
                        prompt = prompts[0]
                        if prompt in prompt_list:
                            continue
                        prompt_list.add(prompt)

                        prompts = prompts * actual_batch_size
                        output = model(prompts).images
                        output = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0)
                        output = output.cpu()
                        base_count_dir = len(os.listdir(img_dir))
                        dir_name = f"{base_count_dir: 05}"
                        dir_path = os.path.join(img_dir, dir_name)
                        os.makedirs(dir_path, exist_ok=True)

                        for i in range(len(output)):
                            base_count = len(os.listdir(dir_path))
                            sample = 255. * rearrange(output[i].numpy(), 'c h w -> h w c')
                            Image.fromarray(sample.astype(np.uint8)).save(
                                os.path.join(dir_path, f"{base_count: 05}.png"))

                        with open(os.path.join(dir_path, "prompt.txt"), 'w', encoding="utf-8") as file:
                            # Write a string to the file
                            file.write("\n".join(prompts))

                        logger.info(
                            f"Computed trial set {base_count_dir + 1} out of {arguments.n_sample_sets} for prompt "
                            f"{prompt}")
                        if base_count_dir + 1 == arguments.n_sample_sets:
                            break

    if len(os.listdir(img_dir)) < arguments.n_sample_sets:
        logger.warning(
            f"Found fewer samples than specified. Fallback to using fewer samples. Given: {os.listdir(img_dir)}, "
            f" Needed: {arguments.n_sample_sets}")

    mean, sd = calc_ms_ssim_for_path_ordered(img_dir, trial_size=arguments.trial_size)
    with open(os.path.join(img_dir, "ms_ssim_results.json"), "w", encoding="utf-8") as file:
        as_str = f"$.{round(float(mean) * 100): 02d} \\pm.{round(float(sd) * 100): 02d}$"
        json.dump({"mean": float(mean), "sdv": float(mean), "as_string": as_str}, file)


if __name__ == '__main__':
    args = get_compute_mssim()
    log_dir = os.path.join(os.path.abspath("."), "log", "mssim", datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'console.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.debug("=" * 30 + f"Running {os.path.basename(__file__)}" + "=" * 30)
    logger.debug(f"Current file: {__file__}")
    main(args)
