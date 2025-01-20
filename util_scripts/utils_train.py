"""
Contains utility functions used in the training scripts, such as argument parsing.
"""
import argparse
import logging
import os
import shutil
from datetime import timedelta
from pathlib import Path
from random import choice

import diffusers
import wandb
from accelerate import InitProcessGroupKwargs, Accelerator
from accelerate.utils import ProjectConfiguration

from packaging import version

import numpy as np
import torch.nn.functional as F
import torch

from diffusers import StableDiffusionPipeline, get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers, is_xformers_available, check_min_version
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import compute_snr

from peft import get_peft_model_state_dict
from torch.utils.data import DataLoader
from torchvision.transforms import functional

from util_scripts.attention_maps import temporary_cross_attention, register_cross_attention_hook, \
    set_layer_with_name_and_path
from util_scripts.preliminary_masks import preprocess_attention_maps
from xray_datasets import get_dataset
from util_scripts.utils_generic import get_latest_directory, normalize_and_scale_tensor


def tokenize_captions(caption_data, tokenizer, is_train=True):
    """
    :param caption_data: A collection of impressions, labels, captions or the like. Multiple captions for a single image
           should be separated by a '|'.
    :param tokenizer: The desired tokenizer.
    :param is_train: If True, will choose a random caption. Otherwise, it will take the first one.
    :return: The input ids and attention mask returned by the tokenizer.
    """
    captions = []

    for caption in caption_data:
        if isinstance(caption, str):
            if '|' in caption:
                caption = caption.split('|')
                captions.append(choice(caption) if is_train else caption[0])
                continue
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column '{captions}' should contain either strings or lists of strings."
            )
    try:
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    # try openai tokenizer if usual call fails
    except TypeError:
        inputs = tokenizer(captions, truncate=True)

    try:
        return inputs.input_ids.to("cuda"), inputs.attention_mask.to("cuda")
    # try openai tokenizer if ususal call fails
    except AttributeError:
        return inputs.to("cuda"), None


def unwrap_model(model, accelerator):
    """
    Unwraps a model from its accelerator wrapper to access the underlying raw model.
    :param model: The model wrapped by an accelerator.
    :param accelerator: The accelerator object used to wrap the model.
    :return: The unwrapped, original model.
    """

    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def get_parser_arguments_train(parser):
    """
    Parses the arguments for the training scripts.
    :param parser: Parser that should parse the arguments.
    :return: Parsed arguments.
    """
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=5,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading for training. 0 means that the data will be loaded in the "
            "main process."
        ),
    )
    parser.add_argument(
        "--dataloader_num_val_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading for validation. 0 means that the data will be loaded "
            "in the main process."
        ),
    )
    parser.add_argument(
        "--chunk_path",
        type=str,
        default=None,
        help=(
            "Path to chunked train data if used."
        ),
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help=(
            "Number of chunks of the training data"
        ),
    )

    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=10,
        help=(
            "Specifies the interval for validation"
        ),
    )

    parser.add_argument(
        "--generation_validation_epochs",
        type=int,
        default=100,
        help=(
            "Specifies the interval for image generation validation"
        ),
    )

    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or "
             "leave `None`. If left to `None` the default prediction type of the scheduler: "
             "`noise_scheduler.config.prediction_type` is chosen.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/experiments/datasets_config_mscoco.yml",
        help=(
            "Path to the dataset configs"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=100,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--llm_name", type=str, default=None,
                        choices=["radbert", "med-kebert", "clip", "openclip", "cxr_clip", "gloria"],
                        help="Name of the custom text encoder to use.")

    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
             " If not specified controlnet weights are initialized from unet.",
    )

    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )

    parser.add_argument("--ucg_probability", type=float, default=0.0, help="unconditional guidance probability")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of cycles to be performed by the "
                                                                     "learning rate scheduler")
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--tracker_project_name", type=str, default="text2img-finetune")
    parser.add_argument("--use_attention_mask", action="store_true",
                        help="enables cross attention masks to mask padding tokens")
    parser.add_argument("--custom_path", type=str, default=None,
                        help="addtional path to pass to custom pipe to be used for loading the"
                             "tokenizer and text encoder")
    parser.add_argument("--use_ddim", action="store_true",
                        help="use ddim scheduler instead of ddpm scheduler if set")
    parser.add_argument("--force_download", action="store_true",
                        help="if set, forces downloading the model from the repository")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="if set, allows executing remote code from the specified repository")
    parser.add_argument("--medklip_preprocessing", action="store_true",
                        help="if set, preprocesses the impressions using knowledge graphs as used in medklip.")
    parser.add_argument("--nofreeze", action='store_false', dest='freeze',
                        help="Pass --nofreeze to unfreeze the text encoder (only supported for CLIP).")
    parser.add_argument("--save_attention", action="store_true",  help="Specifies wheter attention maps should be saved"
                                                                       "during training.")

    return parser


def parse_args():
    """
    Wrapper call to get_parser_arguments_train(parser). Initializes argument parser and parses arguments for training.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser = get_parser_arguments_train(parser)
    args = parser.parse_args()
    if args.use_attention_mask:
        if args.llm_name in ("openclip", "chexzero"):
            raise ValueError("Can't use argument use_attention_mask with llm_name openclip or chexzero.")
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank not in (-1, args.local_rank):
        args.local_rank = env_local_rank
    return args


def save_checkpoint(args, global_step, accelerator, logger, lora_unet=None):
    """
    Saves model checkpoints periodically, managing checkpoint limits.

    :param args: Configuration arguments with checkpointing details.
    :param global_step: Current training step count.
    :param accelerator: Handles model saving on appropriate devices.
    :param logger: Logs checkpoint management details.
    :param lora_unet: Optional unet to extract the lora weights from.
    """

    if global_step % args.checkpointing_steps == 0:
        if accelerator.is_main_process:
            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
            if args.checkpoints_total_limit is not None:
                checkpoints = os.listdir(args.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} "
                        f"checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)

            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(save_path)
            if lora_unet:
                unwrapped_unet = unwrap_model(lora_unet, accelerator)
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet)
                )

                StableDiffusionPipeline.save_lora_weights(
                    save_directory=save_path,
                    unet_lora_layers=unet_lora_state_dict,
                    safe_serialization=True,
                )

            logger.info(f"Saved state to {save_path}")


def prepare_data_at_epoch_start(args, accelerator, epoch, first_epoch, train_dataset, config, collate_batch,
                                tokenizer, train_dataloader):
    """
    Prepares the data loader for training at the start of each epoch, especially handling datasets divided into chunks.

    :param args: Arguments object containing configuration details like batch size.
    :param accelerator: Accelerator object managing device placement and distributed training.
    :param epoch: Current epoch number.
    :param first_epoch: The first epoch of the training (typically zero).
    :param train_dataset: Dataset object containing the training data.
    :param config: Configuration object with settings related to dataset and dataloading.
    :param collate_batch: Function to collate data batches.
    :param tokenizer: Tokenizer to process text data.
    :param train_dataloader: Dataloader for iterating the data.
    :return: DataLoader object prepared for the current epoch.
    """

    with accelerator.main_process_first(), accelerator.autocast():
        if config.datasets.train.num_chunks > 1 and epoch != first_epoch:
            train_dataset.load_next_chunk()
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=collate_batch,
                batch_size=args.train_batch_size,
                num_workers=config.dataloading.num_workers,
            )
            accelerator.print("Tokenizing training data...")
            for data in train_dataset:
                impression = data['impression'] if 'impression' in data else None
                if impression:
                    data['input_ids'], data['attention_mask'] = tokenize_captions([impression], tokenizer,
                                                                                  is_train=True)
                else:
                    raise KeyError("No impression saved")
    return train_dataloader


def enable_xformers(logger, unet):
    """
    Checks and enables xFormers compatibility if available, specifically for memory efficiency in transformers.

    :param logger: Logger object to output warnings or errors.
    :param unet: Model object that might leverage efficient transformers.
    """
    if is_xformers_available():
        import xformers
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warning(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training,"
                " please update xFormers to at least 0.0.17. See "
                "https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")


def train_step(accelerator, unet, vae, text_encoder, noise_scheduler, batch, args, weight_dtype, train_loss,
               optimizer, lr_scheduler, progress_bar, global_step, logger, lora_layers=None, ema_unet=None):
    """
    Performs a single training step, including forward and backward passes, and updates model parameters.

    :param accelerator: Accelerator object to handle device-agnostic code execution.
    :param unet: UNet model for generating predictions.
    :param vae: Variational Autoencoder involved in generating or processing latent spaces.
    :param text_encoder: Encoder for processing text inputs.
    :param noise_scheduler: Scheduler to manage noise levels in the diffusion process.
    :param batch: The current batch of data.
    :param args: General arguments for training specifics, like batch size and noise adjustments.
    :param weight_dtype: Data type for weights, usually float32 or float16 for mixed precision.
    :param train_loss: Cumulative training loss.
    :param optimizer: Optimizer to update model weights.
    :param lr_scheduler: Learning rate scheduler.
    :param progress_bar: Progress bar object to update training progress.
    :param global_step: Global step count across all epochs.
    :param logger: Logger object for logging information.
    :param lora_layers: Optional layers for Low-Rank Adaptation.
    :param ema_unet: Optional Exponential Moving Average model.
    :return: Updated training loss and global step count after the step.
    """
    with accelerator.accumulate(unet):
        # Convert images to latent space
        latents = torch.cat([latent.latent_dist.sample() for latent in batch["img"]]).to(unet.device)
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=unet.device
            )

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=unet.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=weight_dtype)

        # Get the text embedding for conditioning
        if args.use_attention_mask:
            encoder_hidden_states = \
                text_encoder(batch["input_ids"], return_dict=False, attention_mask=batch["attention_mask"])[0]
        else:
            encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

        # Get the target for loss depending on the prediction type
        if args.prediction_type is not None:
            # set prediction_type of scheduler if defined
            noise_scheduler.register_to_config(prediction_type=args.prediction_type)

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        if args.save_attention:
            with temporary_cross_attention(unet) as (unet_modified, attn_maps, neg_attn_maps):
                unet = unet_modified
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                attention_images, _ = preprocess_attention_maps(attn_maps, on_cpu=True)
        else:
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]


        if args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, args.snr_gamma *
                                            torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
            if noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        train_loss += avg_loss.item() / args.gradient_accumulation_steps

        # Backpropagate
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            if lora_layers is not None:
                params_to_clip = lora_layers
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            else:
                accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
    if accelerator.sync_gradients:
        if args.use_ema and ema_unet is not None:
            ema_unet.step(unet.parameters())
        progress_bar.update(1)
        global_step += 1
        accelerator.log({"train_loss": train_loss}, step=global_step)
        train_loss = 0.0
        if lora_layers:
            save_checkpoint(args, global_step, accelerator, logger, unet)
        else:
            save_checkpoint(args, global_step, accelerator, logger)

    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
    progress_bar.set_postfix(**logs)

    return train_loss, global_step


def load_checkpoint(args, accelerator, num_update_steps_per_epoch, first_epoch, pipeline=None):
    """
    Loads model state from a checkpoint if available, or initializes training state.

    :param args: Training arguments including checkpoint path.
    :param accelerator: Accelerator object to manage device-specific state loading.
    :param num_update_steps_per_epoch: Number of update steps per epoch for proper epoch calculation.
    :param first_epoch: The first epoch to start from after loading the checkpoint.
    :param pipeline: Optional StableDiffusionPipeline to load lora weights.
    :return: Tuple containing the initial global step and updated first_epoch.
    """
    if args.resume_from_checkpoint:
        path = get_latest_directory(args)
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            initial_global_step = 0
        else:
            if pipeline is not None:
                pipeline.load_lora_weights(os.path.join(args.output_dir, path))
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    return initial_global_step, first_epoch


def load_data(accelerator, config, vae, tokenizer, collate_batch, args):
    """
    Loads and prepares the training data into a DataLoader.

    :param accelerator: Accelerator object for efficient parallel processing.
    :param config: Configuration object with dataset specifications.
    :param vae: Variational Autoencoder model for processing or generating data.
    :param tokenizer: Tokenizer for processing text inputs.
    :param collate_batch: Function to collate batched data.
    :param args: Arguments specifying details like batch size.
    :return: DataLoader object along with the possibly modified train_dataset.
    """
    train_dataset = get_dataset(config, "train")
    if config.datasets.train.num_chunks > 1:
        accelerator.print("using chunked data")
        train_dataset.load_next_chunk()
    else:
        accelerator.print("using whole dataset")
        train_dataset.load_precomputed(vae)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_batch,
        batch_size=args.train_batch_size,
        num_workers=config.dataloading.num_workers,
    )
    accelerator.print("Tokenizing training data...")
    for data in train_dataset:
        impression = data['impression'] if 'impression' in data else None
        if impression:
            data['input_ids'], data['attention_mask'] = tokenize_captions([impression], tokenizer,
                                                                          is_train=True)
        else:
            raise KeyError("No impression saved")
    return train_dataloader, train_dataset


def track_validation_images(accelerator, images, attention_images, args):
    """
    Tracks and logs validation images and their attention maps using specified tracking tools.

    :param accelerator: Accelerator object equipped with tracking capabilities.
    :param images: List of image tensors from the validation set.
    :param attention_images: Corresponding attention map tensors for the images.
    :param args: Arguments containing specifics like validation prompts.
    """
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                        for i, image in enumerate(images)
                    ],
                    "attention": [
                        wandb.Image(functional.to_pil_image(
                            normalize_and_scale_tensor(
                                image.squeeze()[:, :, 1:-1].mean(dim=(0, 1, 2)))),
                            caption=f"{i}: {args.validation_prompt}")
                        for i, image in enumerate(attention_images)
                    ]
                }
            )
    torch.cuda.empty_cache()


def print_initial_train_stats(args, accelerator, logger, train_dataloader):
    """
    Prints initial training statistics such as number of examples, epochs, and batch sizes.

    :param args: Arguments containing training configurations.
    :param accelerator: Accelerator object for distributed processing stats.
    :param logger: Logger to output training statistics.
    :param train_dataloader: DataLoader for the training dataset.
    """
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")


def init_accelerate_and_logging(args, logger):
    """
    Initializes the Accelerator and logging configuration necessary for training.

    :param args: Arguments with configurations for output directories and acceleration.
    :param logger: Logger object to record training process.
    :return: Initialized Accelerator and updated logger.
    """
    # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
    check_min_version("0.28.0.dev0")

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir))
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=8000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()
    return accelerator, logger


def prepare_with_accelerate(args, accelerator, optimizer, unet, noise_scheduler, vae, text_encoder):
    """
    Prepares all models and utilities with the accelerator for distributed and efficient computation.

    :param args: Arguments containing settings for models and training.
    :param accelerator: Accelerator object to optimize the training and model distribution.
    :param optimizer: Optimizer for updating model weights.
    :param unet: UNet model for image generation.
    :param noise_scheduler: Noise scheduler for managing diffusion process noise.
    :param vae: Variational Autoencoder model.
    :param text_encoder: Text encoder for processing input text.
    :return: Tuple containing all prepared components (unet, optimizer, lr_scheduler, noise_scheduler, vae,
    text_encoder).
    """
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, noise_scheduler, vae, text_encoder = accelerator.prepare(
        unet, optimizer, lr_scheduler, noise_scheduler, vae, text_encoder,
        device_placement=[True, True, True, True, True, True]
    )

    accelerator.register_for_checkpointing(lr_scheduler)

    return unet, optimizer, lr_scheduler, noise_scheduler, vae, text_encoder
