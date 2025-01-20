"""adapted from https://github.com/MischaD/chest-distillation"""

import json
import os
import shutil
from pathlib import Path

from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import pandas as pd
import spacy
import torch
import torch.multiprocessing as mp
from einops import repeat
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.vlp import ImageTextInferenceEngine
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_auc_score
from torch import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional
from tqdm import tqdm

from custom_pipe import FrozenCustomPipe
from evaluation.utils_evaluation import check_mask_exists, samples_to_path, contrast_to_noise_ratio, log_biovil, \
    MIMIC_STRING_TO_ATTENTION, word_to_slice
from evaluation.utils_evaluation import compute_prediction_from_binary_mask
from log import logger
from util_scripts.attention_maps import temporary_cross_attention
from util_scripts.foreground_masks import GMMMaskSuggestor
from util_scripts.preliminary_masks import preprocess_attention_maps
from util_scripts.state_dict_mapper import get_component_mapper
from util_scripts.utils_generic import collate_batch, normalize_and_scale_tensor
from util_scripts.utils_train import tokenize_captions
from utils_evaluation import get_args_compute_bbox, prepare_evaluation
from xray_datasets import get_dataset
from xray_datasets.dataset import add_preliminary_to_sample
from xray_datasets.impression_preprocessors import MedKeBERTPreprocessor, MedKLIPPreprocessor
from xray_datasets.utils import load_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def compute_masks(rank, configuration, mapper, world_size, args):
    """
    Computes the attention masks for the loaded model, as specified in https://arxiv.org/abs/2212.14306.

    :param rank: Device rank of the current process.
    :param configuration: Config file used for more persistent settings.
    :param mapper: Maps between pytorch lightning and diffusers state dicts, only needed for reproduction purposes.
    :param world_size: Number of processes.
    :param args: Additional arguments, see get_args_compute_bbox() for more details.
    :return:
    """

    logger.info(f"Current rank: {rank}")
    lora_weights = args.lora_weights
    dataset = get_dataset(configuration, args.split)
    model, mask_dir = prepare_evaluation(config=configuration, args=args, lora_weights=lora_weights, rank=rank,
                                         mapper=mapper, trust_remote_code=True)
    device = torch.device(rank) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.set_progress_bar_config(position=rank)
    if args.split != "validation":
        dataset.load_precomputed(model.vae)
    else:
        dataset.process_samples(model.vae)
    cond_key = "label_text"
    if args.medklip_preprocessing and args.llm_name == "med-kebert":
        raise ValueError("You cannot combine medklip preprocessing with medkebert preprocessing.")
    if args.llm_name == "med-kebert":
        print("Processing impressions with knowledge graphs...")
        preprocessor = MedKeBERTPreprocessor(configuration.data_dir, configuration.datasets.train.dataset_csv)
        dataset.data = preprocessor.preprocess_impressions(dataset.data)
        cond_key = "impression"
    if args.medklip_preprocessing:
        print("Processing impressions with knowledge graphs...")
        preprocessor = MedKLIPPreprocessor(configuration.data_dir)
        dataset.data = preprocessor.preprocess_impressions(dataset.data)
        cond_key = "impression"
    attention_mask = None
    precision_scope = autocast

    data_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    logger.info(f"Relative path to first sample: {dataset[0]['rel_path']}")

    # CLIP max length for comparing models, not setting model_max_length can cause problems if repo config does not
    # set it
    model.tokenizer.model_max_length = 77

    dataloader = DataLoader(dataset,
                            batch_size=configuration.sample.iou_batch_size,
                            shuffle=False,
                            num_workers=0,  # opt.num_workers,
                            collate_fn=collate_batch,
                            drop_last=False,
                            sampler=data_sampler
                            )

    if args.from_scratch:
        try:
            shutil.rmtree(mask_dir)
        except OSError as exc:
            print(exc)

    os.makedirs(mask_dir, exist_ok=True)

    logger.info(f"Mask dir: {mask_dir}")

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_sci_scibert")

    model2 = FrozenCustomPipe(path=args.path, custom_path=args.custom_path, llm_name=args.llm_name,
                              trust_remote_code=True, device=torch.device(rank),
                              use_ddim=True, sample_mode=False).pipe

    for samples in tqdm(dataloader, "generating masks"):
        docs = []
        for impression in samples[cond_key]:
            indices = []
            doc = nlp(impression)
            text_without_punct = ' '.join(token.text for token in doc if not token.is_punct)
            doc = nlp(text_without_punct)
            for i, token in enumerate(doc):
                if token.pos_ in ['ADJ', 'NOUN', 'VERB']:
                    indices.append(i)
            docs.append(indices)

        with torch.no_grad():
            with precision_scope("cuda"):
                if check_mask_exists(mask_dir, samples):
                    logger.info(f"Masks already exists for {samples['rel_path']}")
                    continue
                if args.llm_name != "med-kebert" and not args.medklip_preprocessing:
                    samples[cond_key] = [str(x.split("|")[0]) for x in samples[cond_key]]
                    samples["impression"] = samples[cond_key]
                if args.llm_name == "openclip":
                    import open_clip
                    input_ids = open_clip.tokenize(samples["impression"]).to(model.device)
                else:
                    input_ids, attention_mask = tokenize_captions(samples["impression"],
                                                                  model.tokenizer, is_train=False)

                controlnet_images = None
                if args.control_path is not None:
                    controlnet_images = [functional.pil_to_tensor(control) for control in
                                         samples["control"]]
                    controlnet_images = torch.stack(controlnet_images, dim=0).to(dtype=model.dtype,
                                                                                 device=model.device)
                if args.use_attention_mask:
                    if args.llm_name == "openclip":
                        raise ValueError("Can't use argument use_attention_mask with llm_name openclip")
                    encoder_hidden_states = model.text_encoder(input_ids.to(model.device),
                                                               attention_mask=attention_mask.to(model.device),
                                                               return_dict=False)[0]
                else:
                    encoder_hidden_states = model.text_encoder(input_ids.to(model.device), return_dict=False)[0]

                with temporary_cross_attention(model.unet, args.guidance_scale > 1) as (unet, attn_maps, neg_attn_maps):
                    model.unet = unet
                    generator = torch.Generator(device="cuda")
                    generator.manual_seed(args.seed)
                    if torch.is_tensor(samples["img"]):
                        ground_truth_images = samples["img"].to(model.device)
                    else:
                        ground_truth_images = samples["img"]
                    if args.sample_mode:
                        if controlnet_images is not None:
                            _ = model(prompt_embeds=encoder_hidden_states, num_inference_steps=args.num_inference_steps,
                                      guidance_scale=args.guidance_scale, clip_skip=1, generator=generator,
                                      ground_truth_image=ground_truth_images, only_first=args.only_first,
                                      image=controlnet_images, attention_mask=attention_mask)[0]
                        else:
                            _ = model(prompt_embeds=encoder_hidden_states, num_inference_steps=args.num_inference_steps,
                                      guidance_scale=args.guidance_scale, clip_skip=1, generator=generator,
                                      ground_truth_image=ground_truth_images, only_first=args.only_first)[0]


                    else:
                        if controlnet_images is not None:
                            _ = model(prompt_embeds=encoder_hidden_states, num_inference_steps=args.num_inference_steps,
                                      guidance_scale=args.guidance_scale, clip_skip=1, generator=generator,
                                      image=controlnet_images)[0]
                        else:
                            _ = model(prompt_embeds=encoder_hidden_states, num_inference_steps=args.num_inference_steps,
                                      guidance_scale=args.guidance_scale, clip_skip=1, generator=generator)[0]

                    attention_images, _ = preprocess_attention_maps(attn_maps, on_cpu=True)
                    negative_attention_images, _ = preprocess_attention_maps(neg_attn_maps, on_cpu=True)
                if args.subtract_attn_images:
                    attention_images = attention_images - negative_attention_images

                model2.unet = model.unet
                with temporary_cross_attention(model2.unet, args.guidance_scale > 1) as (unet, attn_maps, neg_attn_maps):
                    model2.unet = unet
                    _ = model2(prompt_embeds=encoder_hidden_states, num_inference_steps=args.num_inference_steps,
                              guidance_scale=args.guidance_scale, clip_skip=1, generator=generator)[0]
                    attention_images2, _ = preprocess_attention_maps(attn_maps, on_cpu=True)
                    negative_attention_images2, _ = preprocess_attention_maps(neg_attn_maps, on_cpu=True)


                assert len(attention_images) == len(docs)

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
                    if args.llm_name in ("clip", "openclip"):
                        for out in outs:
                            out = list(filter(lambda x: x != 49407, out))
                            token_lens.append(len(out) - 1)
                    else:
                        for out in outs:
                            out = list(filter(lambda x: x != 0, out))
                            token_lens.append(len(out) - 2)

                    token_positions = list(np.cumsum(token_lens) + 1)
                    token_positions = [1, ] + token_positions
                    if not args.split == "validation":
                        label = samples["finding_labels"][j]
                    else:
                        label = samples[cond_key][j]

                    if args.disease_filtering:
                        query_words = MIMIC_STRING_TO_ATTENTION.get(label, [])
                        locations = word_to_slice(txt_label.split(" "), query_words)
                        locations = [location for location in locations
                                 if token_positions[location + 1] <= model.tokenizer.model_max_length]
                        if len(locations) == 0:
                            # use all
                            tok_attention = attention[-args.rev_diff_steps:, :, token_positions[0]:token_positions[-1]]
                            tok_attentions.append(tok_attention.mean(dim=(0, 1, 2)))
                        else:
                            for location in locations:
                                tok_attention = attention[-args.rev_diff_steps:, :,
                                                token_positions[location]:token_positions[location + 1]]
                                tok_attentions.append(tok_attention.mean(dim=(0, 1, 2)))
                    else:
                        token_indices = [token_positions[index] for index in docs[j]]
                        tok_attention = attention[-args.rev_diff_steps:, :, token_indices]

                    
                    if args.bbm:
                        merged_biases = attention_images[j][:, :, [0]] @ attention_images2[j][:, :, token_indices]
                        merged_biases = normalize_and_scale_tensor(merged_biases.mean(dim=(0,1,2)))
                        image_bias = normalize_and_scale_tensor(attention_images[j][:, :, [0]].mean(dim=(0, 1, 2))).numpy().astype(np.uint8)
                        text_bias = normalize_and_scale_tensor(attention_images2[j][:, :, token_indices].mean(dim=(0, 1, 2))).numpy().astype(np.uint8)
                        s = ssim(image_bias,text_bias)
                        if s < 0:
                            s = 0
                        merged_biases = normalize_and_scale_tensor(merged_biases, False)
                        activation_map = normalize_and_scale_tensor(tok_attention.mean(dim=(0,1,2)), False)
                        tok_attentions.append(2*(1-s)*s*((merged_biases+activation_map+(merged_biases*activation_map))/2) + ((1-s)**2)*activation_map + (s**2)*(merged_biases))
                    else:
                        tok_attentions.append(normalize_and_scale_tensor(tok_attention.mean(dim=(0,1,2)), False))

                    preliminary_attention_mask = torch.stack(tok_attentions).mean(dim=0)
                    if preliminary_attention_mask.size() != torch.Size([64, 64]):
                        raise ValueError(f"Wrong attention mask dimension for {path}")
                    if torch.isnan(preliminary_attention_mask).any():
                        print(f"NaN encountered while computing attention masks for {path}.")
                        print("Token attention lengths:")
                        for tok_attention in tok_attentions:
                            print(len(tok_attention))
                    if Path(mask_dir).stem != "files":
                        os.path.join(mask_dir, "files")
                    path = samples_to_path(mask_dir, samples, j)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    logger.info(f"(rank({rank}): Saving attention mask to {path}")
                    torch.save(preliminary_attention_mask.to("cpu"), path)



def compute_iou_score(configuration, mapper, args):
    """
    Computes the metrics such as iou, cnr, top1, or aucroc for the generates attention masks and stores them in csv
    files.

    :param mapper: Maps between pytorch lightning and diffusers state dicts, only needed for reproduction purposes.
    :param configuration: Config file for some persistent settings.
    :param args:  Additional arguments, see get_args_compute_bbox() for more details.
    :return:
    """
    if args.log_biovil:
        text_inference = get_bert_inference(BertEncoderType.CXR_BERT)
        image_inference = get_image_inference(ImageModelType.BIOVIL)

        image_text_inference = ImageTextInferenceEngine(
            image_inference_engine=image_inference,
            text_inference_engine=text_inference,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_text_inference.to(device)

    lora_weights = args.lora_weights
    dataset = get_dataset(configuration, args.split)

    pipeline, mask_dir = prepare_evaluation(config=configuration, args=args, lora_weights=lora_weights, mapper=mapper,
                                            trust_remote_code=True)
    if args.split != "validation":
        dataset.load_precomputed(pipeline.vae)
    else:
        dataset.process_samples(pipeline.vae)

    dataloader = DataLoader(dataset,
                            batch_size=configuration.sample.iou_batch_size,
                            shuffle=False,
                            num_workers=0,  # opt.num_workers,
                            collate_fn=collate_batch,
                            drop_last=False,
                            )
    if hasattr(dataset, "add_preliminary_masks"):
        dataset.add_preliminary_masks(mask_dir, sanity_check=False)
    mask_suggestor = GMMMaskSuggestor(configuration)
    results = {"rel_path": [], "finding_labels": [], "iou": [], "miou": [], "bboxiou": [], "bboxmiou": [],
               "distance": [], "top1": [], "aucroc": [], "cnr": []}

    if args.medklip_preprocessing and args.llm_name == "med-kebert":
        raise ValueError("You cannot combine medklip preprocessing with medkebert preprocessing.")
    if args.llm_name == "med-kebert":
        print("Processing impressions with knowledge graphs...")
        preprocessor = MedKeBERTPreprocessor(configuration.data_dir, configuration.datasets.train.dataset_csv)
        dataset.data = preprocessor.preprocess_impressions(dataset.data)
    if args.medklip_preprocessing:
        print("Processing impressions with knowledge graphs...")
        preprocessor = MedKLIPPreprocessor(configuration.data_dir)
        dataset.data = preprocessor.preprocess_impressions(dataset.data)

    for samples in tqdm(dataloader, "computing metrics"):

        if args.llm_name != "med-kebert" and not args.medklip_preprocessing:
            samples["label_text"] = [str(x.split("|")[0]) for x in samples["label_text"]]
            samples["impression"] = samples["label_text"]

        for i in range(len(samples["img"])):
            sample = {k: v[i] for k, v in samples.items()}
            try:
                add_preliminary_to_sample(sample, samples_to_path(mask_dir, samples, i))
            except FileNotFoundError:
                print(f"{samples_to_path(mask_dir, samples, i)} not found - skipping sample")
                continue

            bboxes = sample["bboxxywh"].split("|")
            for j in range(len(bboxes)):
                bbox = [int(float(box)) for box in bboxes[j].split("-")]
                bboxes[j] = bbox
            if isinstance(sample["bbox_img"], np.ndarray):
                ground_truth_img = torch.tensor(sample["bbox_img"]).float()
            else:
                ground_truth_img = sample["bbox_img"].float()

            if torch.isnan(sample["preliminary_mask"]).any():
                logger.warning(f"NaN in prediction: {sample['rel_path']} -- {samples_to_path(mask_dir, samples, i)}")
                continue

            binary_mask = repeat(mask_suggestor(sample, key="preliminary_mask"), "h w -> 3 h w")
            if not bool(binary_mask.any()):
                print("No bounding box could be extracted from the attention map - "
                      "setting results for this sample to 0")
                results["rel_path"].append(sample["rel_path"])
                results["finding_labels"].append(sample["finding_labels"])
                results["cnr"].append(float(0))
                results["iou"].append(float(0))
                results["miou"].append(float(0))
                results["bboxiou"].append(float(0))
                results["bboxmiou"].append(float(0))
                results["top1"].append(float(0))
                results["aucroc"].append(float(0))
                results["distance"].append(float(0))
                continue

            prelim_binary_mask = np.array(binary_mask.float()).transpose((1, 2, 0))
            binary_mask_large = torch.tensor(cv2.resize(prelim_binary_mask, (512, 512),
                                                        interpolation=cv2.INTER_LINEAR).round().transpose((2, 0, 1)))

            prelim_mask = (sample["preliminary_mask"] - sample["preliminary_mask"].min()) / (
                    sample["preliminary_mask"].max() - sample["preliminary_mask"].min())
            prelim_mask_large = torch.tensor(cv2.resize(np.array(prelim_mask.float()), (512, 512),
                                                        interpolation=cv2.INTER_LINEAR))

            results["rel_path"].append(sample["rel_path"])
            results["finding_labels"].append(sample["finding_labels"])
            results["cnr"].append(float(contrast_to_noise_ratio(ground_truth_img, prelim_mask_large)))
            prediction, center_of_mass_prediction, _ = compute_prediction_from_binary_mask(binary_mask_large[0])
            iou = torch.tensor(jaccard_score(ground_truth_img.flatten(), binary_mask_large[0].flatten()))
            iou_rev = torch.tensor(jaccard_score(1 - ground_truth_img.flatten(), 1 - binary_mask_large[0].flatten()))
            results["iou"].append(float(iou))
            results["miou"].append(float((iou + iou_rev) / 2))

            bboxiou = torch.tensor(jaccard_score(ground_truth_img.flatten(), prediction.flatten()))
            bboxiou_rev = torch.tensor(jaccard_score(1 - ground_truth_img.flatten(), 1 - prediction.flatten()))
            results["bboxiou"].append(float(bboxiou))
            results["bboxmiou"].append(float((bboxiou + bboxiou_rev) / 2))

            if len(bboxes) > 1:
                results["distance"].append(np.nan)
            else:
                _, center_of_mass, _ = compute_prediction_from_binary_mask(ground_truth_img)
                distance = np.sqrt((center_of_mass[0] - center_of_mass_prediction[0]) ** 2 +
                                   (center_of_mass[1] - center_of_mass_prediction[1]) ** 2
                                   )
                results["distance"].append(float(distance))

            argmax_idx = np.unravel_index(prelim_mask_large.argmax(), prelim_mask_large.size())
            mode_is_outlier = ground_truth_img[argmax_idx]
            results["top1"].append(float(mode_is_outlier))

            auc = roc_auc_score(ground_truth_img.flatten(), prelim_mask_large.flatten())
            results["aucroc"].append(auc)

            if (args.log or args.log_biovil) and sample.get("img_raw") is not None:
                logger.info(f"Logging example bboxes and attention maps to {configuration.log_dir}")
                log_biovil(dataset, sample, ground_truth_img, results, configuration, prelim_mask_large,
                           binary_mask_large, args, image_text_inference)

    df = pd.DataFrame(results)
    logger.info(f"Saving file with results to {mask_dir}")
    df.to_csv(os.path.join(mask_dir, f"pgm_{args.phrase_grounding_mode}_bbox_results.csv"))
    if 'frequency' not in df.columns:
        df['frequency'] = df.groupby('finding_labels')['finding_labels'].transform('count')
        df['frequency'] = df['frequency'].astype(int)

    # Compute the weighted average for each numeric column except 'frequency'
    weighted_averages = {}
    numeric_columns = [col for col in df.columns if df[col].dtype.kind in 'bifc' and col != 'frequency']

    for column in numeric_columns:
        weighted_sum = (df[column] * df['frequency']).sum()
        total_weight = df['frequency'].sum()
        weighted_averages[column] = weighted_sum / total_weight

    # Convert weighted averages to a DataFrame
    weighted_avg_df = pd.DataFrame(weighted_averages, index=[0])

    # Add total frequency as a simple sum, not as a weighted average
    weighted_avg_df['frequency'] = df['frequency'].count()

    # Concatenate mean_results with weighted_avg_df
    mean_results = df.groupby('finding_labels').mean(numeric_only=True)
    mean_results = pd.concat([mean_results, weighted_avg_df.rename(index={0: 'Weighted_Average'})])
    mean_results.to_csv(os.path.join(mask_dir, f"pgm_{args.phrase_grounding_mode}_bbox_results_means.csv"))
    logger.info(df.groupby("finding_labels").mean(numeric_only=True))

    with open(os.path.join(mask_dir, f"pgm_{args.phrase_grounding_mode}_bbox_results.json"), "w",
              encoding="utf-8") as file:
        json_results = {"all": dict(df.mean(numeric_only=True))}
        for x in mean_results.index:
            json_results[x] = dict(mean_results.loc[x])

        json.dump(json_results, file, indent=4)


if __name__ == '__main__':
    arguments = get_args_compute_bbox()
    config = load_config(arguments.config)
    mapper_obj = None
    if config.load_lightning:
        mapper_obj = get_component_mapper()
    device_count = torch.cuda.device_count()
    mp.spawn(
        compute_masks,
        args=(config, mapper_obj, device_count, arguments),
        nprocs=device_count
    )
    compute_iou_score(config, mapper_obj, arguments)