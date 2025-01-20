"""contains parts from https://github.com/wooyeolBaek/attention-map"""

from contextlib import contextmanager

import torch
import torch.utils.checkpoint
import numpy as np
from diffusers.models.attention_processor import (Attention, AttnProcessor, AttnProcessor2_0, XFormersAttnProcessor)
from diffusers.utils import logging, is_xformers_available

if is_xformers_available():
    import xformers
    import xformers.ops

logger = logging.get_logger(__name__)


def cross_attn_init():
    """
    Overwrites the call function of the attention processor classes.
    :return: The original call functions of the attention processors.
    """
    call = AttnProcessor.__call__
    call2 = AttnProcessor2_0.__call__
    if is_xformers_available():
        AttnProcessor.__call__ = xformers_attn_call
        AttnProcessor2_0.__call__ = xformers_attn_call
    else:
        AttnProcessor.__call__ = attn_call
        AttnProcessor2_0.__call__ = attn_call
    return call, call2


def disable_cross_attn(call1, call2):
    """
    Restores the original attention processor call functions. Intended to be ussed with cross_attn_init().
    :param call1: Saved call function of AttnProcessor.
    :param call2: Saved call function of AttnProcessor2_0.
    :return:
    """
    AttnProcessor.__call__ = call1
    AttnProcessor2_0.__call__ = call2


def hook_fn(name, attn_maps, neg_attn_maps):
    def forward_hook(module, _, __):
        if not hasattr(module.processor, "attn_map"):
            return

        # Initialize storage for this module if not present
        if name not in attn_maps:
            attn_maps[name] = []
        attn_maps[name].append(module.processor.attn_map.clone())

        if hasattr(module.processor, "neg_attn_map"):
            if name not in neg_attn_maps:
                neg_attn_maps[name] = []
            neg_attn_maps[name].append(module.processor.neg_attn_map.clone())

        # Clear maps to avoid memory accumulation
        del module.processor.attn_map
        if hasattr(module.processor, "neg_attn_map"):
            del module.processor.neg_attn_map

    return forward_hook

@contextmanager
def temporary_cross_attention(unet, cfg=False, self_attn=False, use_xformers=False):
    """
    Context manager to temporarily replace the attention processor calls and collect attention maps.
    """
    attn_maps = {}
    neg_attn_maps = {}

    # Save original methods
    original_attn_call = AttnProcessor.__call__
    original_attn_call2_0 = AttnProcessor2_0.__call__

    # Replace with custom methods
    if is_xformers_available() and use_xformers:
        AttnProcessor.__call__ = xformers_attn_call
        AttnProcessor2_0.__call__ = xformers_attn_call
    else:
        if use_xformers:
            logger.warning("xformers is not available, using torch-based variant")
        AttnProcessor.__call__ = attn_call
        AttnProcessor2_0.__call__ = attn_call

    # Register cross-attention hooks with clear state
    def reset_attention_processors(unet):
        unet = set_layer_with_name_and_path(unet, target_name="attn" if self_attn else "attn2")
        return unet

    unet = reset_attention_processors(unet)
    unet, hook = register_cross_attention_hook(unet, cfg=cfg, attn_maps=attn_maps, neg_attn_maps=neg_attn_maps, self_attn=self_attn)

    try:
        yield unet, attn_maps, neg_attn_maps
    finally:
        # Restore original methods and remove all hooks
        AttnProcessor.__call__ = original_attn_call
        AttnProcessor2_0.__call__ = original_attn_call2_0
        for h in hook:  # Iterate through the list of hooks
            h.remove()  # Remove each hook
        attn_maps.clear()
        neg_attn_maps.clear()


def register_cross_attention_hook(unet, cfg=False, attn_maps=None, neg_attn_maps=None, self_attn=False):
    """
    Registers the hook function to all cross-attention layers of the model.
    """
    hooks = []

    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn'):
            # Continue only if it's the target layer (self or cross-attention)
            if name.endswith('attn2') or (self_attn and name.endswith('attn1')):
                if isinstance(module.processor, (AttnProcessor, AttnProcessor2_0)):
                    module.processor.store_attn_map = True
                    if cfg:
                        module.processor.cfg = True
                    hook = module.register_forward_hook(hook_fn(name, attn_maps, neg_attn_maps))
                    hooks.append(hook)

    return unet, hooks


def set_layer_with_name_and_path(model, target_name="attn2", current_path=""):
    """
    Recursively overwrites the attention processor of the model.
    """
    for name, layer in model.named_children():
        new_path = current_path + '.' + name if current_path else name
        if name.endswith(target_name):
            layer.processor = AttnProcessor2_0()

        set_layer_with_name_and_path(layer, target_name, new_path)

    return model



def attn_call(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
    """
    Processes input through an attention mechanism with optional normalization, resizing, and output processing.

    :param self: The class instance the method is part of, typically containing configuration or state.
    :param attn: An Attention instance, which provides various attention-related operations.
    :param hidden_states: The main tensor input for attention calculations.
    :param encoder_hidden_states: Optional tensor input from an encoder for cross-attention.
    :param attention_mask: Optional mask to apply on attention scores to exclude certain tokens.
    :param temb: Optional tensor representing conditional embeddings or external input.
    :param height: Optional height for reshaping inputs in case of 2D spatial data.
    :param width: Optional width for reshaping inputs in case of 2D spatial data.
    :return: The tensor after attention has been applied, potentially with residuals and normalization.
    """
    residual = hidden_states

    channel = None

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 3:
        _, image_dim, _ = hidden_states.shape
        height = width = int(np.sqrt(image_dim))


    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    # query = attn.to_q(hidden_states, scale=scale)
    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    # key = attn.to_k(encoder_hidden_states, scale=scale)
    key = attn.to_k(encoder_hidden_states)
    # value = attn.to_v(encoder_hidden_states, scale=scale)
    value = attn.to_v(encoder_hidden_states)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    ####################################################################################################
    if hasattr(self, "store_attn_map"):
        from einops import rearrange
        self.attn_map = (rearrange(attention_probs, '(b nh) (h w) d -> b nh d h w', h=height, b=batch_size)
                         .mean(dim=1)).cpu().detach()
        if hasattr(self, "cfg"):
            self.neg_attn_map = self.attn_map[:batch_size // 2]
            self.attn_map = self.attn_map[batch_size // 2:]
    ####################################################################################################
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    # hidden_states = attn.to_out[0](hidden_states, scale=scale)
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def xformers_attn_call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs):


    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 3:
        _, image_dim, _ = hidden_states.shape
        height = width = np.sqrt(image_dim)

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, key_tokens, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)

    if attention_mask is not None:
        # expand our mask's singleton query_tokens dimension:
        #   [batch*heads,            1, key_tokens] ->
        #   [batch*heads, query_tokens, key_tokens]
        # so that it can be added as a bias onto the attention scores that xformers computes:
        #   [batch*heads, query_tokens, key_tokens]
        # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
        _, query_tokens, _ = hidden_states.shape
        attention_mask = attention_mask.expand(-1, query_tokens, -1)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    query = attn.head_to_batch_dim(query).contiguous()
    key = attn.head_to_batch_dim(key).contiguous()
    value = attn.head_to_batch_dim(value).contiguous()

    ####################################################################################################

    if hasattr(self, "store_attn_map"):
        attention_probs = attn.get_attention_scores(query, key, attention_mask).cpu().detach()
        from einops import rearrange
        self.attn_map = (rearrange(attention_probs, '(b nh) (h w) d -> b nh d h w', h=height.astype(np.uint8), b=batch_size)
                         .mean(dim=1))
        if hasattr(self, "cfg"):
            self.neg_attn_map = self.attn_map[:batch_size // 2]
            self.attn_map = self.attn_map[batch_size // 2:]
    ####################################################################################################

    hidden_states = xformers.ops.memory_efficient_attention(
        query, key, value, attn_bias=attention_mask, op=None, scale=attn.scale
    )
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states

