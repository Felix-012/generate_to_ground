"""adapted from https://github.com/huggingface/diffusers/tree/main/examples/text_to_image"""
import os
from contextlib import nullcontext
from pathlib import Path

import math
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm

from custom_pipe import FrozenCustomPipe
from xray_datasets.utils import load_config
from util_scripts.attention_maps import temporary_cross_attention, set_layer_with_name_and_path, \
    register_cross_attention_hook, all_attn_maps
from util_scripts.preliminary_masks import preprocess_attention_maps
from util_scripts.utils_generic import collate_batch, get_latest_directory
from util_scripts.utils_train import tokenize_captions, unwrap_model, \
    parse_args, prepare_data_at_epoch_start, enable_xformers, train_step, \
    load_data, track_validation_images, print_initial_train_stats, load_checkpoint, init_accelerate_and_logging, \
    prepare_with_accelerate

logger = get_logger(__name__, log_level="INFO")


def main():
    """
    Prepares and starts lora training loop.
    :return:
    """

    args = parse_args()
    global logger

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from diffusers import StableDiffusionPipeline
    from diffusers.training_utils import cast_training_params
    from diffusers.utils import convert_state_dict_to_diffusers

    accelerator, logger = init_accelerate_and_logging(args, logger)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    pipeline = FrozenCustomPipe(path=args.pretrained_model_name_or_path, accelerator=accelerator,
                                llm_name=args.llm_name, use_ddim=args.use_ddim, force_download=args.force_download,
                                trust_remote_code=args.trust_remote_code)
    unet = pipeline.pipe.unet
    vae = pipeline.pipe.vae
    text_encoder = pipeline.pipe.text_encoder
    tokenizer = pipeline.pipe.tokenizer
    noise_scheduler = pipeline.pipe.scheduler

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet)
    # to half-precision as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(device="cuda", dtype=weight_dtype)
    vae.to(device="cuda", dtype=weight_dtype)
    text_encoder.to(device="cuda", dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        enable_xformers(logger=logger, unet=unet)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    args.output_dir = os.path.join(os.path.expandvars(args.output_dir), str(args.rank))

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

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    config = load_config(args.config)
    with accelerator.main_process_first():
        train_dataloader, train_dataset = load_data(accelerator, config, vae, tokenizer, collate_batch, args)
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
    # Afterward we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
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
    empty_token_id = tokenizer(
        "", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to("cuda")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        train_dataloader = prepare_data_at_epoch_start(args=args, accelerator=accelerator, collate_batch=collate_batch,
                                                       config=config, train_dataset=train_dataset, tokenizer=tokenizer,
                                                       epoch=epoch, first_epoch=first_epoch,
                                                       train_dataloader=train_dataloader)
        for batch in train_dataloader:
            for i in range(len(batch["input_ids"])):
                if bool(torch.rand(1) < args.ucg_probability):
                    batch["input_ids"][i] = empty_token_id
            train_loss, global_step = train_step(
                accelerator, unet, vae, text_encoder, noise_scheduler, batch, args, weight_dtype, train_loss,
                optimizer, lr_scheduler, progress_bar, global_step, logger, lora_layers=lora_layers)
            if global_step >= args.max_train_steps:
                break
            if accelerator.is_main_process:
                if (args.validation_prompt is not None
                        and progress_bar.n % args.generation_validation_epochs == 0
                        and any(Path(args.output_dir).iterdir())):
                    try:
                        get_latest_directory(args)
                    except TypeError:
                        logger.info(
                            f"Skipping validation - checkpoint {args.resume_from_checkpoint} could not be found")
                        continue
                    logger.info(
                        f"Running validation generation... \n Generating {args.num_validation_images}"
                        f" images with prompt: "
                        f" {args.validation_prompt}."
                    )
                    # create pipeline
                    pipeline = FrozenCustomPipe(path=args.pretrained_model_name_or_path, llm_name="clip",
                                                accelerator=accelerator, use_ddim=args.use_ddim,
                                                trust_remote_code=True).pipe
                    pipeline.load_lora_weights(
                        os.path.join(os.path.expandvars(args.output_dir), get_latest_directory(args)))

                    pipeline = pipeline.to("cuda")
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    generator = torch.Generator(device="cuda")
                    if args.seed is not None:
                        generator = generator.manual_seed(args.seed)
                    images = []
                    attention_images = []
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type)

                    with autocast_ctx, temporary_cross_attention():
                        guidance_scale = 1.0
                        pipeline.unet = set_layer_with_name_and_path(pipeline.unet)
                        pipeline.unet, _ = register_cross_attention_hook(pipeline.unet, guidance_scale > 1)
                        for _ in range(args.num_validation_images):
                            all_attn_maps.clear()
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
                                preprocess_attention_maps(all_attn_maps, on_cpu=True).detach().cpu())

                    track_validation_images(accelerator, images, attention_images, args)
                    del pipeline
            torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet, accelerator)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        accelerator.end_training()


if __name__ == "__main__":
    main()
