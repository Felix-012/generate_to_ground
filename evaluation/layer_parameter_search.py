"""
File for conducting trials regarding which configuration of attention layers is the best.
"""

import datetime
import os

import cv2
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from evaluation.utils_evaluation import MIMIC_STRING_TO_ATTENTION, word_to_slice, contrast_to_noise_ratio
from util_scripts.attention_maps import (all_attn_maps, all_neg_attn_maps, set_layer_with_name_and_path,
                                         register_cross_attention_hook, temporary_cross_attention)
from util_scripts.preliminary_masks import preprocess_attention_maps
from util_scripts.utils_generic import collate_batch, get_args_parameter_search
from util_scripts.utils_train import tokenize_captions
from utils_evaluation import prepare_evaluation
from xray_datasets import get_dataset
from xray_datasets.utils import load_config


class SubsetRandomSampler(Sampler):
    """
    Provides random sampling of a dataset without replacement, with a visual progress indicator on the number of
    iterations left.
    """

    def __init__(self, num_samples, data_source, max_iterations):
        """
        Initializes the SubsetRandomSampler with specified sample and iteration limits.

        :param num_samples: The number of samples to randomly select.
        :param data_source: The dataset from which to sample.
        :param max_iterations: The maximum number of times the sampler can be used.
        """
        super().__init__()
        self.num_samples = num_samples
        self.data_source = data_source
        self.max_iterations = max_iterations
        self.iteration_count = 0

    def __iter__(self):
        """
        Provides an iterator over randomly chosen indices based on num_samples, up to a maximum number of iterations.
        Integrates tqdm for a progress bar.

        :return: An iterator over randomly selected indices from the dataset if iteration limit not exceeded.
        """
        for _ in tqdm(range(self.max_iterations), desc="Sampling Iterations"):
            if self.iteration_count < self.max_iterations:
                indices = torch.randperm(len(self.data_source))[:self.num_samples].tolist()
                self.iteration_count += 1
                yield from indices

    def __len__(self):
        """
        Returns the number of samples that will be drawn.

        :return: The number of samples.
        """
        return self.num_samples


def evaluate_model(trial):
    """
    Computes masks and evaluates their cnr values against the ground truth. Items are sampled randomly from ChestXRay14
    bounding boxes.
    :param trial: Optuna trial
    :return: Added up cnr evaluation of the runs.
    """
    # Suggests which layers to drop or keep (0 means drop, 1 means keep)
    layer_statuses = [trial.suggest_int(f"layer_{i}", 0, 1) for i in range(16)]
    slices = [i for i, x in enumerate(layer_statuses) if x == 1]
    cnr = 0
    cond_key = "label_text"
    # CLIP max length for comparing models, not setting model_max_length can cause problems if repo config does not
    # set it
    model.tokenizer.model_max_length = 77

    sampler = SubsetRandomSampler(configuration.sample.iou_batch_size, dataset, args.num_iterations)

    dataloader = DataLoader(dataset,
                            batch_size=configuration.sample.iou_batch_size,
                            sampler=sampler,
                            num_workers=0,  # opt.num_workers,
                            collate_fn=collate_batch,
                            drop_last=False,
                            shuffle=False
                            )

    os.makedirs(mask_dir, exist_ok=True)

    for samples in dataloader:
        with torch.no_grad():
            samples[cond_key] = [str(x.split("|")[0]) for x in samples[cond_key]]
            samples["impression"] = samples[cond_key]

            input_ids, attention_mask = tokenize_captions(samples["impression"],
                                                          model.tokenizer, is_train=False)
            if args.use_attention_mask:
                encoder_hidden_states = model.text_encoder(input_ids.to(model.device),
                                                           attention_mask=attention_mask.to(model.device),
                                                           return_dict=False)[0]
            else:
                encoder_hidden_states = model.text_encoder(input_ids.to(model.device), return_dict=False)[0]
            ground_truth_images = samples["img"]
            all_attn_maps.clear()
            all_neg_attn_maps.clear()
            with temporary_cross_attention():
                model.unet = set_layer_with_name_and_path(model.unet)
                model.unet, _ = register_cross_attention_hook(model.unet, args.guidance_scale > 1)
                generator = torch.Generator(device="cuda")
                generator.manual_seed(4200)

                _ = model(prompt_embeds=encoder_hidden_states, num_inference_steps=args.num_inference_steps,
                          guidance_scale=args.guidance_scale, clip_skip=1, generator=generator,
                          ground_truth_image=ground_truth_images)[0]

            attention_images = preprocess_attention_maps(all_attn_maps, on_cpu=True)

            for j, attention in enumerate(list(attention_images)):
                tok_attentions = []
                txt_label = samples[cond_key][j]
                # determine tokenization
                txt_label = txt_label.split("|")[0]
                words = txt_label.split(" ")
                if not isinstance(words, list):
                    words = list(words)
                assert isinstance(words[0], str)
                outs = model.tokenizer(words, padding="max_length",
                                       max_length=model.tokenizer.model_max_length,
                                       truncation=True,
                                       return_tensors="pt")["input_ids"]
                token_lens = []
                for out in outs:
                    out = list(filter(lambda x: x != 0, out))
                    token_lens.append(len(out) - 2)

                token_positions = list(np.cumsum(token_lens) + 1)
                token_positions = [1, ] + token_positions
                label = samples[cond_key][j]

                query_words = MIMIC_STRING_TO_ATTENTION.get(label, [])
                locations = word_to_slice(txt_label.split(" "), query_words)
                locations = [location for location in locations
                             if token_positions[location + 1] <= model.tokenizer.model_max_length]
                if len(locations) == 0:
                    # use all
                    tok_attention = attention[:, slices, token_positions[0]:token_positions[-1]]
                    tok_attentions.append(tok_attention.mean(dim=(0, 1, 2)))
                else:
                    for location in locations:
                        tok_attention = attention[:, slices, token_positions[location]:token_positions[location + 1]]
                        tok_attentions.append(tok_attention.mean(dim=(0, 1, 2)))

                preliminary_attention_mask = torch.stack(tok_attentions).mean(dim=0)
                prelim_mask = preliminary_attention_mask - preliminary_attention_mask / (
                        preliminary_attention_mask.max() - preliminary_attention_mask.min())
                prelim_mask_large = torch.tensor(cv2.resize(np.array(prelim_mask.float()), (512, 512),
                                                            interpolation=cv2.INTER_LINEAR))
                ground_truth_img = samples["bbox_img"][j]
                cnr += float(contrast_to_noise_ratio(ground_truth_img, prelim_mask_large))
                torch.cuda.empty_cache()
    return cnr


args = get_args_parameter_search()
configuration = load_config(args.config)

dataset = get_dataset(configuration, args.split)
model, mask_dir = prepare_evaluation(config=configuration, args=args, rank=3, trust_remote_code=True,
                                     lora_weights=None)
device = torch.device(3) if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.set_progress_bar_config(position=0)
dataset.process_samples(model.vae)
study = optuna.create_study(study_name=f"biovil_parameter_search_{datetime.datetime.timestamp(datetime.datetime.now())}"
                            , direction="maximize")
study.optimize(evaluate_model, n_trials=1000)

print("Best trial:")
print(study.best_trial)
study.trials_dataframe().to_csv(f"/vol/ideadata/ce90tate/data/{study.study_name}/.csv")
