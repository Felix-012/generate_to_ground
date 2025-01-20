"""
Code modified from DETR tranformer:
https://github.com/facebookresearch/detr
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerDecoder(nn.Module):
    """
    Implements a transformer decoder which can process sequences to return
    the final output and intermediate representations.
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        """
        Initialize the transformer decoder module.

        :param decoder_layer: the type of decoder layer to use
        :param num_layers: the number of layers in the decoder
        :param norm: the normalization layer (optional)
        :param return_intermediate: if set to True, returns all intermediate outputs
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        Propagate the inputs through the decoder and return output.

        :param tgt: target sequence
        :param memory: memory input from the encoder
        :param tgt_mask: target sequence mask
        :param memory_mask: memory input mask
        :param tgt_key_padding_mask: padding mask for target keys
        :param memory_key_padding_mask: padding mask for memory keys
        :param pos: positional encoding for memory sequence
        :param query_pos: positional encoding for target sequence
        :return: output and attention weights
        """
        output = tgt
        intermediate = []
        atten_layers = []
        for n, layer in enumerate(self.layers):

            residual = True
            output, ws = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos, residual=residual)
            atten_layers.append(ws)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output, atten_layers


class TransformerDecoderLayer(nn.Module):
    """
    Represents a single layer in the transformer decoder architecture.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        Initialize a layer of the transformer decoder.

        :param d_model: dimension of the model
        :param nhead: number of attention heads
        :param dim_feedforward: dimension of the feed-forward network model
        :param dropout: dropout rate
        :param activation: activation function to use
        :param normalize_before: if set to True, normalization is applied before the layer operations
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        Optionally adds position embeddings to the input tensor.

        :param tensor: the input tensor to which position embeddings may be added
        :param pos: the position embeddings to add, if not None
        :return: tensor with position embeddings added if pos is not None, otherwise the original tensor
        """

        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        Forward pass for the post layer normalization configuration.

        :param tgt: target sequence
        :param memory: memory input from the encoder
        :param memory_mask: mask for memory sequence
        :param memory_key_padding_mask: padding mask for memory sequence
        :param pos: positional encoding for memory sequence
        :param query_pos: positional encoding for target sequence
        :return: updated target sequence and attention weights
        """
        tgt = self.norm1(tgt)
        tgt2, ws = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)

        # attn_weights [B,NUM_Q,T]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, ws

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """
       Forward pass for the pre layer normalization configuration.

       :param tgt: target sequence
       :param memory: memory input from the encoder
       :param tgt_mask: mask for target sequence
       :param memory_mask: mask for memory sequence
       :param tgt_key_padding_mask: padding mask for target sequence
       :param memory_key_padding_mask: padding mask for memory sequence
       :param pos: positional encoding for memory sequence
       :param query_pos: positional encoding for target sequence
       :return: updated target sequence and attention weights
       """
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, ws = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                                 key=self.with_pos_embed(memory, pos),
                                                 value=memory, attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn_weights

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        The forward pass that decides between pre and post layer normalization based on the configuration.

        :param tgt: target sequence
        :param memory: memory input from the encoder
        :param tgt_mask: mask for target sequence
        :param memory_mask: mask for memory sequence
        :param tgt_key_padding_mask: padding mask for target sequence
        :param memory_key_padding_mask: padding mask for memory sequence
        :param pos: positional encoding for memory sequence
        :param query_pos: positional encoding for target sequence
        :return: updated target sequence and attention weights depending on the normalization strategy
        """
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


def _get_clones(module, n):
    """
    Create a list containing clones of the specified module.

    :param module: module to be cloned
    :param n: number of clones
    :return: a ModuleList of clones
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation):
    """
   Returns the activation function based on the string identifier.

   :param activation: string name of the activation function
   :return: callable activation function
   """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
