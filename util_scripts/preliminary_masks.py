"""adapted from https://github.com/MischaD/chest-distillation"""

from functools import partial

import torch
from einops import rearrange, repeat


def all_mean(inp):
    """
    Computes the mean across all dimensions of the input tensor.
    :param inp: Input tensor.
    :return: Tensor representing the mean value across all dimensions, keeping dimensions.
    """
    return inp.mean(dim=(0, 1, 2), keepdim=True)


def diffusion_steps_mean(x, steps):
    """
    Computes the mean of the specified number of steps from the end of the input tensor.
    :param x: Input tensor where the second dimension is expected to be singleton.
    :param steps: Number of last steps to include in the mean calculation.
    :return: Mean of the selected steps as a tensor, keeping dimensions.
    """
    assert x.size()[2] == 1
    return x[-steps:, :, 0].mean(dim=(0, 1), keepdim=True)


def relevant_token_step_mean(x, tok_idx, steps):
    """
    Computes the mean of specific token indices over a given number of steps from the input tensor.
    :param x: Input tensor.
    :param tok_idx: Token index to focus on.
    :param steps: Number of last steps to include in the mean calculation.
    :return: Mean of the selected token and steps as a tensor, keeping dimensions.
    """
    return x[-steps:, :, tok_idx:(tok_idx + 1)].mean(dim=(0, 1), keepdim=True)


def all_token_mean(x, steps, max_token=None):
    """
    Computes the mean over all tokens up to a maximum token index over the specified last steps.
    :param x: Input tensor.
    :param steps: Number of last steps to include in the mean calculation.
    :param max_token: Maximum token index to consider.
    :return: Mean of the selected tokens and steps as a tensor, keeping dimensions.
    """
    return x[-steps:, :, :max_token].mean(dim=(0, 1), keepdim=True)


def multi_relevant_token_step_mean(x, tok_idx, steps):
    """
    Computes the mean across multiple token indices for a specified number of last steps.
    :param x: Input tensor.
    :param tok_idx: List of token indices to focus on.
    :param steps: Number of last steps to include in the mean calculation.
    :return: Mean of the selected tokens and steps as a tensor, keeping dimensions.
    """
    res = None
    for tok_id in tok_idx:
        if res is None:
            res = x[-steps:, :, tok_id:(tok_id + 1)].mean(dim=(0, 1), keepdim=True)
        else:
            res += x[-steps:, :, tok_id:(tok_id + 1)].mean(dim=(0, 1), keepdim=True)

    res = res.mean(dim=(0, 1), keepdim=True)
    return res


class AttentionExtractor:
    """
    A class to apply a specified function to extract attention maps from input tensors.
    """

    def __init__(self, *args, function=None, **kwargs):
        """
        Initializes the AttentionExtractor with a reduction function.
        :param function: Function or name of the method within this class to use as the reduction function.
        """
        if isinstance(function, str):
            self.reduction_function = getattr(self, function)
        else:
            self.reduction_function = function

        if args or kwargs:
            self.reduction_function = partial(self.reduction_function, *args, **kwargs)

    def __call__(self, inp, *args, **kwargs):
        """
        Applies the reduction function to the input tensor.
        :param inp: Input tensor, expected to be 5-dimensional.
        :return: Output tensor, after applying the reduction function.
        """
        assert inp.ndim == 5
        out = self.reduction_function(inp, *args, **kwargs)
        assert out.ndim == 5
        return out


def print_attention_info(attention):
    """
    Prints detailed information about attention maps.
    :param attention: List of attention maps.
    """
    print(f"Num Forward passes: {len(attention)}, Depth:{len(attention[0])}")
    for i in range(len(attention[0])):
        print(f"Layer: {i} - {attention[0][i].size()}")


def normalize_attention_map_size(attention_maps, on_cpu=False):
    """
    Normalizes the size of attention maps and optionally moves them to CPU.
    :param attention_maps: Dictionary of attention maps.
    :param on_cpu: Boolean indicating whether to move maps to CPU.
    :return: Normalized attention maps.
    """
    if on_cpu:
        for layer_key, layer_list in attention_maps.items():
            for iteration in range(len(layer_list)):
                attention_maps[layer_key][iteration] = attention_maps[layer_key][iteration].to("cpu").detach()
    for key, layer in attention_maps.items():  # trough layers / diffusion steps
        for iteration in range(len(layer)):
            attention_map = attention_maps[key][iteration]  # B x num_resblocks x numrevdiff x H x W
            if attention_map.size()[-1] != 64:
                upsampling_factor = 64 // attention_map.size()[-1]
                attention_map = repeat(attention_map, 'b tok h w -> b tok (h h2) (w w2)', h2=upsampling_factor,
                                       w2=upsampling_factor)
            attention_maps[key][iteration] = attention_map

    attention_maps = torch.cat([torch.stack(lst).unsqueeze(0) for lst in list(attention_maps.values())], dim=0)
    attention_maps = rearrange(attention_maps, "layer depth b tok h w -> b depth layer  tok h w")
    return attention_maps


def get_latent_slice(batch, opt):
    """
    Generates slices for data sampling based on batch information and options.
    :param batch: Batch data containing slices.
    :param opt: Options with sampling frequency details.
    :return: Tuple of slices.
    """
    ds_slice = []
    for slice_ in batch["slice"]:
        if slice_.start is None:
            ds_slice.append(slice(None, None, None))
        else:
            ds_slice.append(slice(slice_.start // opt.f, slice_.stop // opt.f, None))
    return tuple(ds_slice)


def preprocess_attention_maps(attention_masks, on_cpu=None):
    """
    Preprocesses attention masks by normalizing their size and optionally moving to CPU.
    Self-attention and cross-attention maps are treated separately.
    :param attention_masks: Attention masks to preprocess.
    :param on_cpu: Optional; whether to move masks to CPU.
    :return: Preprocessed attention masks.
    """
    self_attention = {}
    self_attn_layers = [key for key in attention_masks if key.endswith('attn1')]
    for key in self_attn_layers:
        self_attention[key] = attention_masks.pop(key)
    if attention_masks and self_attention:
        return (normalize_attention_map_size(attention_masks, on_cpu),
                normalize_attention_map_size(self_attention, on_cpu))
    elif self_attention:
        return None, normalize_attention_map_size(self_attention, on_cpu)
    elif attention_masks:
        return normalize_attention_map_size(attention_masks, on_cpu), None
    else:
        return None, None

