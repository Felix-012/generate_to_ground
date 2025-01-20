"""adapted from https://github.com/huggingface/diffusers/tree/main/examples/text_to_image"""

import logging
import os
from datetime import timedelta
from pathlib import Path

import accelerate
import diffusers
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    ControlNetModel,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional
from tqdm import tqdm as tqdm_def
from tqdm.auto import tqdm

from custom_pipe import FrozenCustomPipe
from util_scripts.utils_generic import collate_batch
from util_scripts.utils_train import tokenize_captions, \
    parse_args, save_checkpoint, prepare_data_at_epoch_start
from xray_datasets import get_dataset
from xray_datasets.utils import load_config

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    """
    Arranges a list of image objects into a specified grid layout and returns the resulting composite image.

    :param imgs: List of PIL.Image objects to arrange in a grid.
    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    :return: A new PIL.Image object representing the composite image grid.
    """

    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def main(main_args):
    """
    Sets up and executes training for a control net.

    :param main_args: Parsed arguments passed to the script.
    :return:
    """
    logging_dir = Path(main_args.output_dir, main_args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=main_args.output_dir, logging_dir=str(logging_dir))
    kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=6000))]

    accelerator = Accelerator(
        gradient_accumulation_steps=main_args.gradient_accumulation_steps,
        mixed_precision=main_args.mixed_precision,
        log_with=main_args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=kwargs)

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

    # If passed along, set the training seed now.
    if main_args.seed is not None:
        set_seed(main_args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if main_args.output_dir is not None:
            os.makedirs(main_args.output_dir, exist_ok=True)

    pipeline = FrozenCustomPipe(path=main_args.pretrained_model_name_or_path, accelerator=accelerator,
                                llm_name=main_args.llm_name, control=True, use_ddim=main_args.use_ddim,
                                trust_remote_code=args.trust_remote_code, force_download=args.force_download)
    unet = pipeline.pipe.unet
    vae = pipeline.pipe.vae
    text_encoder = pipeline.pipe.text_encoder
    tokenizer = pipeline.pipe.tokenizer
    noise_scheduler = pipeline.pipe.scheduler
    controlnet = pipeline.pipe.controlnet

    accelerator.wait_for_everyone()

    if main_args.controlnet_model_name_or_path:
        with os.scandir(main_args.controlnet_model_name_or_path) as it:
            if any(it):
                logger.info("Loading existing controlnet weights")
                controlnet = ControlNetModel.from_pretrained(main_args.controlnet_model_name_or_path)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        """
        Unwraps a model from its accelerator wrapping for direct operations.

        :param model: Model to be unwrapped.
        :return: Unwrapped model.
        """
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            """
            Saves models to a specified directory, using model-specific subdirectories.

            :param models: List of models to be saved.
            :param weights: Dictionary of model weights to manage.
            :param output_dir: Directory where models will be saved.
            """

            if accelerator.is_main_process:
                for model in models:
                    if isinstance(model, UNet2DConditionModel):
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif isinstance(model, ControlNetModel):
                        model.save_pretrained(os.path.join(output_dir, "controlnet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            """
            Loads model configurations and states from a specified directory into existing model instances.

            :param models: List of models to load states into.
            :param input_dir: Directory from which to load model states.
            """

            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    if main_args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during "
                    "training, please update xFormers to at least 0.0.17. See "
                    "https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if main_args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if main_args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if main_args.scale_lr:
        main_args.learning_rate = (
                main_args.learning_rate * main_args.gradient_accumulation_steps * main_args.train_batch_size *
                accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if main_args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            ) from exc

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=main_args.learning_rate,
        betas=(main_args.adam_beta1, main_args.adam_beta2),
        weight_decay=main_args.adam_weight_decay,
        eps=main_args.adam_epsilon,
    )
    train_dataloader = None
    config = load_config(main_args.config)
    train_dataset = get_dataset(config, "train")
    for i in range(dist.get_world_size()):
        if i == dist.get_rank():
            if config.datasets.train.num_chunks > 1:
                print("using chunked data")
                train_dataset.load_next_chunk()
            else:
                print("using whole dataset")
                train_dataset.load_precomputed(vae)

            if not train_dataset[0].get("control", None):
                for j in tqdm_def(range(len(train_dataset)), desc="Processing control conditioning"):
                    control = train_dataset.load_image(os.path.expandvars(os.path.join(train_dataset.base_dir,
                                                                                       train_dataset[j]["rel_path"]
                                                                                       .replace(".dcm", ".jpg"))))
                    train_dataset[i]["control"] = (train_dataset.
                                                   preprocess_control(control,
                                                                      config.get("control_preprocessing_type:",
                                                                                 "canny")))

            train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_batch,
                                          batch_size=main_args.train_batch_size,
                                          num_workers=config.dataloading.num_workers)

            print("Tokenizing training data...")
            for data in train_dataset:
                impression = data['impression'] if 'impression' in data else None
                if impression:
                    data['input_ids'], data['attention_mask'] = tokenize_captions([impression], tokenizer,
                                                                                  is_train=True)
                else:
                    raise KeyError("No impression saved")
        dist.barrier()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / main_args.gradient_accumulation_steps)
    if main_args.max_train_steps is None:
        main_args.max_train_steps = main_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        main_args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=main_args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=main_args.max_train_steps * accelerator.num_processes,
        num_cycles=main_args.lr_num_cycles,
        power=main_args.lr_power,
    )
    accelerator.register_for_checkpointing(lr_scheduler)
    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / main_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        main_args.max_train_steps = main_args.num_train_epochs * num_update_steps_per_epoch
    # Afterward we recalculate our number of training epochs
    main_args.num_train_epochs = math.ceil(main_args.max_train_steps / num_update_steps_per_epoch)
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(main_args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(main_args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = main_args.train_batch_size * accelerator.num_processes * main_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {main_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {main_args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {main_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {main_args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if main_args.resume_from_checkpoint:
        if main_args.resume_from_checkpoint != "latest":
            path = os.path.basename(main_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(main_args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{main_args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            main_args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(main_args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, main_args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    empty_token_id = tokenizer(
        "", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to("cuda")

    for epoch in range(first_epoch, main_args.num_train_epochs):
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                train_dataloader = prepare_data_at_epoch_start(args=main_args, accelerator=accelerator,
                                                               collate_batch=collate_batch, config=config,
                                                               train_dataset=train_dataset, tokenizer=tokenizer,
                                                               epoch=epoch, first_epoch=first_epoch,
                                                               train_dataloader=train_dataloader)
            dist.barrier()
        for batch in train_dataloader:
            with accelerator.accumulate(controlnet):
                for i in range(len(batch["input_ids"])):
                    if bool(torch.rand(1) < main_args.ucg_probability):
                        batch["input_ids"][i] = empty_token_id

                # Convert images to latent space
                latents = torch.cat([latent.latent_dist.sample() for latent in batch["img"]]).to(unet.device)
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                if main_args.use_attention_mask:
                    encoder_hidden_states = \
                        text_encoder(batch["input_ids"], return_dict=False, attention_mask=batch["attention_mask"])[0]
                else:
                    encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                controlnet_images = [transforms.functional.pil_to_tensor(control) for control in batch["control"]]
                controlnet_images = torch.stack(controlnet_images, dim=0).to(dtype=weight_dtype, device=unet.device)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_images,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents.to(unet.dtype),
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, main_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                save_checkpoint(main_args, global_step, accelerator, logger)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= main_args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(main_args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
