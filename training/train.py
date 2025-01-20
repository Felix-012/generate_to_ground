"""adapted from https://github.com/huggingface/diffusers/tree/main/examples/text_to_image"""

import copy
import os
import math
from contextlib import nullcontext

import accelerate
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import UNet2DConditionModel, EMAModel
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel

from custom_pipe import FrozenCustomPipe
from util_scripts.attention_maps import (set_layer_with_name_and_path, register_cross_attention_hook, \
                                         temporary_cross_attention)
from util_scripts.preliminary_masks import preprocess_attention_maps
from util_scripts.utils_generic import collate_batch
from util_scripts.utils_generic import get_latest_directory
from util_scripts.utils_train import tokenize_captions, unwrap_model, \
    parse_args, prepare_data_at_epoch_start, enable_xformers, \
    train_step, load_data, track_validation_images, print_initial_train_stats, load_checkpoint, \
    init_accelerate_and_logging, prepare_with_accelerate
from xray_datasets.impression_preprocessors import MedKeBERTPreprocessor, MedKLIPPreprocessor
from xray_datasets.utils import load_config


def main():
    """
    Sets up and executes training.
    :return:
    """
    logger = get_logger(__name__, log_level="INFO")
    # Need to set cache dir before importing diffusers.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    from diffusers import StableDiffusionPipeline

    accelerator, logger = init_accelerate_and_logging(args, logger)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler, tokenizer and models.

    pipeline = FrozenCustomPipe(path=args.pretrained_model_name_or_path, accelerator=accelerator,
                                llm_name=args.llm_name, custom_path=args.custom_path, use_ddim=args.use_ddim,
                                trust_remote_code=args.trust_remote_code, force_download=args.force_download,
                                use_freeze=args.freeze)
    unet = pipeline.pipe.unet
    vae = pipeline.pipe.vae
    text_encoder = pipeline.pipe.text_encoder
    tokenizer = pipeline.pipe.tokenizer
    noise_scheduler = pipeline.pipe.scheduler

    # CLIP max length for comparing models, not setting model_max_length can cause problems if repo config does not
    # set it
    tokenizer.model_max_length = 77
    vae.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        ema_unet.to("cuda")

    if args.enable_xformers_memory_efficient_attention:
        enable_xformers(logger=logger, unet=unet)

        # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for model in models:
                    if isinstance(model, UNet2DConditionModel):
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    if not args.freeze and isinstance(model, CLIPTextModel):
                        model.save_pretrained(os.path.join(output_dir, "text_encoder"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif not args.freeze and isinstance(model, CLIPTextModel):
                    load_model = CLIPTextModel.from_pretrained(input_dir, subfolder="text_encoder")
                    model.load_state_dict(load_model.state_dict())
                    del load_model


            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to("cuda")
                ema_unet.copy_to(unet.parameters())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size *
                accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW
    if args.freeze:
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = optimizer_cls(
            list(unet.parameters()) + list(text_encoder.parameters()),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    config = load_config(args.config)
    with accelerator.main_process_first():
        train_dataloader, train_dataset = load_data(accelerator, config, vae, tokenizer, collate_batch, args)
        if args.medklip_preprocessing and args.llm_name == "med-kebert":
            raise ValueError("You cannot combine medklip preprocessing with medkebert preprocessing.")
        if args.llm_name == "med-kebert":
            print("Processing impressions with knowledge graphs...")
            preprocessor = MedKeBERTPreprocessor(config.data_dir, config.datasets.train.dataset_csv)
            train_dataset.data = preprocessor.preprocess_impressions(train_dataset.data)
        if args.medklip_preprocessing:
            print("Processing impressions with knowledge graphs...")
            preprocessor = MedKLIPPreprocessor(os.path.expandvars(config.data_dir))
            train_dataset.data = preprocessor.preprocess_impressions(train_dataset.data)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    unet, optimizer, lr_scheduler, noise_scheduler, vae, text_encoder = prepare_with_accelerate(args, accelerator,
                                                                                                optimizer, unet,
                                                                                                noise_scheduler, vae,
                                                                                                text_encoder)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    # Train!
    print_initial_train_stats(args, accelerator, logger, train_dataloader)
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    global_step, first_epoch = load_checkpoint(args, accelerator, num_update_steps_per_epoch, first_epoch)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    try:
        empty_token_id = tokenizer(
            "", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
    except TypeError:
        empty_token_id = tokenizer("", truncate=True)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if not args.freeze:
            text_encoder.train()
        train_loss = 0.0
        train_dataloader = prepare_data_at_epoch_start(args=args, accelerator=accelerator, collate_batch=collate_batch,
                                                       config=config, train_dataset=train_dataset, tokenizer=tokenizer,
                                                       epoch=epoch, first_epoch=first_epoch,
                                                       train_dataloader=train_dataloader)
        for batch in train_dataloader:
            train_loss, global_step = train_step(
                accelerator, unet, vae, text_encoder, noise_scheduler, batch, args, weight_dtype, train_loss,
                optimizer, lr_scheduler, progress_bar, global_step, logger, ema_unet=ema_unet)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                for i in range(len(batch["input_ids"])):
                    if bool(torch.rand(1) < args.ucg_probability):
                        batch["input_ids"][i] = empty_token_id
                if args.validation_prompt is not None and progress_bar.n % args.generation_validation_epochs == 0:
                    try:
                        get_latest_directory(args)
                    except TypeError:
                        logger.info(
                            f"Skipping validation - checkpoint {args.resume_from_checkpoint} could not be found")
                        continue
                    logger.info(
                        f"Running validation generation... \n Generating {args.num_validation_images} "
                        f"images with prompt: "
                        f" {args.validation_prompt}."
                    )

                    inference_unet = copy.deepcopy(unet)
                    inference_unet.eval()
                    inference_unet.requires_grad_(False)
                    pipeline = StableDiffusionPipeline(
                        vae=accelerator.unwrap_model(vae),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        tokenizer=tokenizer,
                        unet=accelerator.unwrap_model(inference_unet),
                        safety_checker=None,
                        feature_extractor=None,
                        scheduler=noise_scheduler
                    )

                    pipeline.set_progress_bar_config(disable=True)

                    if args.enable_xformers_memory_efficient_attention:
                        pipeline.enable_xformers_memory_efficient_attention()

                    # run inference
                    generator = torch.Generator(device="cuda")
                    if args.seed is not None:
                        generator = generator.manual_seed(args.seed)
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type)

                    images = []
                    attention_images = []
                    with temporary_cross_attention(unet) as (unet_modified, attn_maps, neg_attn_maps):
                        pipeline.unet = accelerator.unwrap_model(unet_modified)
                        for i in range(args.num_validation_images):
                            with autocast_ctx, torch.no_grad():
                                input_ids, attention_masks = tokenize_captions(args.validation_prompt,
                                                                               pipeline.tokenizer, is_train=False)
                                # input_ids_neg, neg_attention_mask = tokenize_captions([""] * len(samples),
                                #  pipeline.tokenizer, is_train=False)
                                if args.use_attention_mask:
                                    encoder_hidden_states = \
                                        pipeline.text_encoder(input_ids, attention_masks, return_dict=False)[0]
                                else:
                                    encoder_hidden_states = pipeline.text_encoder(input_ids, return_dict=False)[0]

                                image = pipeline(prompt_embeds=encoder_hidden_states,
                                                 num_inference_steps=50, guidance_scale=1.0, clip_skip=1,
                                                 generator=generator)[0]
                            images.append(image[0])

                            attention_images.append(
                                preprocess_attention_maps(attn_maps, on_cpu=True)[0].detach().cpu())

                    track_validation_images(accelerator, images, attention_images, args)
                    del pipeline
                    del inference_unet
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

    # Save the layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet, accelerator)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )

        pipeline.unet.save_pretrained(args.output_dir)

        accelerator.end_training()


if __name__ == "__main__":
    main()
