"""
Wrapper for StableDiffusionPipelines which provides various utilities for creating a modified pipeline.
"""

import open_clip
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, \
    StableDiffusionInpaintPipeline, DDIMScheduler, StableDiffusionControlNetPipeline, ControlNetModel, PNDMScheduler
from requests.exceptions import HTTPError
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, CLIPTextModel, CLIPTokenizer

from sample_pipe import SamplePipeline, ControlSamplePipeline
from util_scripts.attention_maps import (
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
)
from wrappers import OpenClipWrapper


def _freeze(model):
    """
    Freezes the given models parameters.
    :param model: Pytorch module.
    :return:
    """
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False


def preprocess_checkpoint(checkpoint, prefixes, target=None, remove=None):
    """
    Adapts huggingface transformer models saved with lightning to normal huggingface transformer models.
    :param checkpoint: Checkpoint of transformer model saved with pytorch lightning.
    :param prefixes: List of prefixes that need to be filtered.
    :param target: Part of the key that needs to be present (e.g. text_encoder if you want to load the text_encoder).
    :param remove: Keys that contain this string will be removed.
    :return: Adapted state_dict to be loaded with model.load_state_dict(state_dict).
    """
    filtered_dict = {}
    try:
        items = checkpoint["model"].items()
    except KeyError:
        items = checkpoint["state_dict"].items()
    for key, value in items:
        if target is None or target in key:
            prefix_counter = sum(1 for prefix in prefixes if prefix in key)
            parts = key.split('.')
            new_key = '.'.join(parts[prefix_counter + 1:])  # Remove the prefixes
            if remove is None or remove not in key:
                filtered_dict[new_key] = value
    return filtered_dict


def load_text_encoder(component_name, path, torch_dtype, llm_name, force_download, trust_remote_code, variant=None):
    """
    Loads the text encoder from a huggingface repository or local folder.
    :param component_name: Name of the subfolder (usually text_encoder).
    :param path: Path to the repository.
    :param torch_dtype: Datatype of the weights to load.
    :param llm_name: Name of the language model. Check get_parser_arguments_train in utils_train.py for options.
    :param force_download: if True, forces downloading the model from the repository.
    :param trust_remote_code: if True, allows executing remote code from the specified repository.
    :param variant: Variant of the weights (e.g. ema or non-ema weights)
    :return: Loaded text model.
    """
    if llm_name == "chexagent":
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype,
                                                    variant=variant, force_download=force_download,
                                                    trust_remote_code=trust_remote_code).language_model
    if llm_name in ("cxr_clip", "gloria"):
        # this model is the base for cxr_clip and gloria
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        weights = torch.load(path)
        # adapt checkpoint to model and load it
        model.load_state_dict(preprocess_checkpoint(weights, ['text_encoder.', 'model.'],
                                                    "text_encoder.", "position_ids"))
        return model
    if llm_name == "clip":
        return CLIPTextModel.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype,
                                             variant=variant, force_download=force_download,
                                             trust_remote_code=trust_remote_code)
    if llm_name == "openclip":
        return OpenClipWrapper(open_clip.create_model_from_pretrained(path)[0])
    try:
        return AutoModel.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype, variant=variant,
                                         trust_remote_code=True, force_download=force_download)
    except (OSError, HTTPError):
        return AutoModel.from_pretrained(path, torch_dtype=torch_dtype, variant=variant, trust_remote_code=True,
                                         force_download=force_download)


def load_vae(component_name, path, torch_dtype, force_download, trust_remote_code, variant=None):
    """
    Loads the autoencoder from a huggingface repository or local folder.
    :param component_name: Name of the subfolder (usually vae).
    :param path: Path to the repository.
    :param torch_dtype: Datatype of the weights to load.
    :param force_download: if True, forces downloading the model from the repository.
    :param trust_remote_code: if True, allows executing remote code from the specified repository.
    :param variant: Variant of the weights (e.g. ema or non-ema weights)
    :return: Loaded autoencoder.
    """
    return AutoencoderKL.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype,
                                         variant=variant, trust_remote_code=trust_remote_code,
                                         force_download=force_download)


def load_tokenizer(component_name, path, torch_dtype, llm_name, force_download, trust_remote_code, variant=None):
    """
    Loads the tokenizer from a huggingface repository or local folder.
    :param component_name: Name of the subfolder (usually tokenizer).
    :param path: Path to the repository.
    :param torch_dtype: Datatype of the weights to load.
    :param llm_name: Name of the underlying language model. Check get_parser_arguments_train in utils_train.py
           for options.
    :param force_download: if True, forces downloading the model from the repository.
    :param trust_remote_code: if True, allows executing remote code from the specified repository.
    :param variant: Variant of the weights (e.g. ema or non-ema weights)
    :return: Loaded tokenizer.
    """
    if llm_name == "openclip":
        return open_clip.get_tokenizer(path).tokenizer
    if llm_name == "clip":
        return CLIPTokenizer.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype,
                                             trust_remote_code=trust_remote_code, variant=variant)
    if llm_name in ("cxr_clip", "gloria"):
        return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    try:
        return AutoTokenizer.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype,
                                             variant=variant, force_download=force_download,
                                             trust_remote_code=trust_remote_code)
    except (OSError, HTTPError):
        return AutoTokenizer.from_pretrained(path, torch_dtype=torch_dtype, variant=variant,
                                             force_download=force_download, trust_remote_code=trust_remote_code)


def load_unet(component_name, path, torch_dtype, force_download, trust_remote_code, variant=None):
    """
    Loads the unet from a huggingface repository or local folder.
    :param component_name: Name of the subfolder (usually unet).
    :param path: Path to the repository.
    :param torch_dtype: Datatype of the weights to load.
    :param force_download: if True, forces downloading the model from the repository.
    :param trust_remote_code: if True, allows executing remote code from the specified repository.
    :param variant: Variant of the weights (e.g. ema or non-ema weights)
    :return: Loaded unet.
    """
    return UNet2DConditionModel.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype,
                                                variant=variant, trust_remote_code=trust_remote_code,
                                                force_download=force_download)


def check_pipeline_arguments(inpaint, control, sample_mode):
    """
    :param inpaint: Sets that the inpaint pipeline should be used.
    :param control: Sets that the control pipeline should be used.
    :param sample_mode: Sets that the sample pipeline should be used.
    :return:
    """
    if any([control, sample_mode]) and inpaint:
        raise ValueError("You cannot combine inpaint with control or sample_mode")


class FrozenCustomPipe:
    """
    Wrapper class for custom StableDiffusionPipelines.
    """

    def __init__(self, path, use_freeze=True, variant=None, llm_name="", torch_dtype=torch.float32, device="cuda",
                 save_attention=False, inpaint=False, control=False, accelerator=None, custom_path=None, use_ddim=False,
                 force_download=False, trust_remote_code=False, sample_mode=False):
        """
        :param use_freeze: set to True if the text encoder should be frozen
        :param path: Path to a repository (local or online).
        :param variant: Specifies the variant if multiple ones exist in a repository.
        :param llm_name: Name of the custom text encoder, if you wish to overwrite the standard text encoder in a
               repository. Choices can be seen in parse_args() in utils_train.py.
        :param torch_dtype: Data type of the weights.
        :param device: Device of the pipeline.
        :param save_attention: Enables saving of the cross attention maps. Do not enable during training, but use
               temporary_cross_attention() for that.
        :param inpaint: Use StableDiffusionInpaintPipeline instead of StableDiffusionPipeline
        :param accelerator: Instance of the accelerator used. Only needs to be set if you only want one progress message
               per process.
        :param custom_path: additional custom repository path for tokenizer and text encoder
        :param force_download: if True, forces downloading the model from the repository.
        :param trust_remote_code: if True, allows executing remote code from the specified repository.
        :param sample_mode: if True, the custom pipeline SamplePipeline is used, which feeds the noisy ground-truth
                            images to the pipeline while sampling.
        """
        check_pipeline_arguments(inpaint, control, sample_mode)
        if use_ddim:
            self.sampler_class = DDIMScheduler
        else:
            self.sampler_class = PNDMScheduler

        if accelerator:
            accelerator.print(f"using {self.sampler_class}")
        else:
            print(f"using {self.sampler_class}")

        self.device = device
        component_loader = {
            "text_encoder": load_text_encoder,
            "tokenizer": load_tokenizer,
            "unet": load_unet,
            "vae": load_vae,
            "scheduler": self.load_scheduler,
        }
        component_mapper = {}

        for component_name in component_loader:
            if accelerator:
                accelerator.print(f"Loading {component_name}...")
            else:
                print(f"Loading {component_name}...")
            if component_name in ("tokenizer", "text_encoder"):
                if custom_path is not None:
                    component = component_loader.get(component_name)(component_name, custom_path, torch_dtype, llm_name,
                                                                     force_download, trust_remote_code, variant)
                else:
                    component = component_loader.get(component_name)(component_name, path, torch_dtype, llm_name,
                                                                     force_download, trust_remote_code, variant)
            else:
                component = component_loader.get(component_name)(component_name, path, torch_dtype, force_download,
                                                                 trust_remote_code, variant)
            component_mapper[component_name] = component

        if use_freeze:
            _freeze(component_mapper["text_encoder"])

        if accelerator:
            accelerator.print("Building custom pipeline...")
        else:
            print("Building custom pipeline...")

        if hasattr(component_mapper["tokenizer"], "pad_token") and component_mapper["tokenizer"].pad_token is None:
            component_mapper["tokenizer"].pad_token = component_mapper["tokenizer"].eos_token

        if inpaint:
            pipe = StableDiffusionInpaintPipeline(unet=component_mapper["unet"],
                                                  text_encoder=component_mapper["text_encoder"],
                                                  tokenizer=component_mapper["tokenizer"], vae=component_mapper["vae"],
                                                  scheduler=component_mapper["scheduler"], safety_checker=None,
                                                  feature_extractor=None,
                                                  requires_safety_checker=False)
        elif control and not sample_mode:
            pipe = StableDiffusionControlNetPipeline(unet=component_mapper["unet"],
                                                     text_encoder=component_mapper["text_encoder"],
                                                     tokenizer=component_mapper["tokenizer"],
                                                     vae=component_mapper["vae"],
                                                     scheduler=component_mapper["scheduler"], safety_checker=None,
                                                     feature_extractor=None,
                                                     controlnet=ControlNetModel.from_unet(component_mapper["unet"]),
                                                     requires_safety_checker=False)
        elif control:
            pipe = ControlSamplePipeline(unet=component_mapper["unet"],
                                         text_encoder=component_mapper["text_encoder"],
                                         tokenizer=component_mapper["tokenizer"],
                                         vae=component_mapper["vae"],
                                         scheduler=component_mapper["scheduler"], safety_checker=None,
                                         feature_extractor=None,
                                         controlnet=ControlNetModel.from_unet(component_mapper["unet"]),
                                         requires_safety_checker=False)
        elif sample_mode:
            pipe = SamplePipeline(unet=component_mapper["unet"], text_encoder=component_mapper["text_encoder"],
                                  tokenizer=component_mapper["tokenizer"], vae=component_mapper["vae"],
                                  scheduler=component_mapper["scheduler"], safety_checker=None,
                                  feature_extractor=None, requires_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline(unet=component_mapper["unet"], text_encoder=component_mapper["text_encoder"],
                                           tokenizer=component_mapper["tokenizer"], vae=component_mapper["vae"],
                                           scheduler=component_mapper["scheduler"], safety_checker=None,
                                           feature_extractor=None, requires_safety_checker=False)

        self.pipe = pipe.to(device)

        if save_attention:
            self.init_attn_save()

    def load_scheduler(self, component_name, path, torch_dtype, force_download, trust_remote_code, variant=None):
        """
        Loads the noise scheduler from a huggingface repository or local folder.
        :param trust_remote_code: if True, allows executing remote code from the specified repository.
        :param component_name: Name of the subfolder (usually scheduler).
        :param path: Path to the repository.
        :param torch_dtype: Datatype of the weights to load.
        :param force_download: if True, forces downloading the model from the repository.
        :param variant: Variant of the weights (e.g. ema or non-ema weights)
        :return: Loaded noise scheduler.
        """
        return self.sampler_class.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype,
                                                  variant=variant, trust_remote_code=trust_remote_code,
                                                  force_download=force_download)

    def init_attn_save(self):
        """
        Enables saving cross attention maps in a Unet2DConditional unet.
        Will be called when save_attention is set to True in FrozenCustomPipe. Should not be used during training, since
        it will most likely lead to memory errors. Use temporary_cross_attention() for training instead.
        :return:
        """
        cross_attn_init()
        self.pipe.unet = set_layer_with_name_and_path(self.pipe.unet)
        self.pipe.unet, _ = register_cross_attention_hook(self.pipe.unet)
