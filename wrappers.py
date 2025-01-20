"""
File for wrappers for modules that are not directly compatible with a diffusers pipeline.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from transformer import TransformerDecoder, TransformerDecoderLayer


class OpenClipWrapper(torch.nn.Module):
    """
    Barebone wrapper for OpenClip models. Used to reproduce specific configurations and experiments as described in
    https://arxiv.org/abs/2303.17908. This class enables manipulation and specific custom configurations of the
    underlying OpenClip model.
    """

    def __init__(self, model):
        """
        Initializes the OpenClipWrapper with a provided OpenClip model.
        :param model: The OpenClip model to be wrapped.
        """
        super().__init__()
        self.model = model
        self.dtype = torch.float32
        self.device = None
        self.config = None

    def __call__(self, *args, **kwargs):
        """
        Processes input text through the wrapped OpenClip model's transformer, applying embeddings and positional
        encodings.
        :param args: Positional arguments where the first is expected to be the text input.
        :param kwargs: Keyword arguments for the transformer forward pass.
        :return: List containing the transformed output tensor.
        """
        text = args[0]
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask[:])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return [x]

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        """
        Advances the input tensor `x` through the transformer layers of the model, respecting the attention mask.
        :param x: Input tensor to be processed.
        :param attn_mask: Optional attention mask to apply during the transformer forward pass.
        :return: Tensor after being processed by the transformer blocks.
        """
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:  # - 1 since the penultimate layer is expected
                break
            x = r(x, attn_mask=attn_mask)
        return x

    def to(self, device=None, dtype=None):
        """
        Moves the model to a specified device and dtype.
        :param device: Target device for the model.
        :param dtype: Data type to convert all floating point parameters and buffers.
        :return: self, the instance of the OpenClipWrapper.
        """
        self.model = self.model.to(device)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self


class ClipWrapper(torch.nn.Module):
    """
    Barebone wrapper for the CLIP architecture. Used to make our pipeline compatibkle with
    https://github.com/openai/CLIP/, in order to load the checkpoints provided by chexzero:
    https://github.com/rajpurkarlab/CheXzero.
    """

    def __init__(self, model):
        """
        Initializes the ClipWrapper with a provided CLIP model.
        :param model: The CLIP model to be wrapped.
        """
        super().__init__()
        self.model = model
        self.dtype = torch.float32
        self.device = None
        self.config = None

    def __call__(self, *args, **kwargs):
        """
        Processes input text through the wrapped CLIP model's transformer, applying embeddings and positional
        encodings.
        :param args: Positional arguments where the first is expected to be the text input.
        :param kwargs: Keyword arguments for the transformer forward pass.
        :return: List containing the transformed output tensor.
        """
        text = args[0]
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x.type(self.model.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return [x]

    def to(self, device=None, dtype=None):
        """
        Moves the model to a specified device and dtype.
        :param device: Target device for the model.
        :param dtype: Data type to convert all floating point parameters and buffers.
        :return: self, the instance of the OpenClipWrapper.
        """
        self.model = self.model.to(device)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self


# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
class MedKLIPWrapper(nn.Module):
    """
    Wrapper for MedKLIP https://github.com/MediaBrain-SJTU/MedKLIP/
    """

    def __init__(self, text_encoder, config, ana_book, disease_book, mode='train'):
        """
        Initialize the MedKLIPWrapper.

        :param config: dict, configuration parameters
        :param ana_book: dict, anatomical book with input_ids and attention_mask
        :param disease_book: dict, disease book with input_ids and attention_mask
        :param mode: str, mode of operation ('train' or 'eval')
        """
        super().__init__()

        self.mode = mode
        self.d_model = 768
        # book embedding
        with torch.no_grad():
            bert_model = self._get_bert_basemodel(text_encoder, freeze_layers=None).to(
                ana_book['input_ids'].device)
            self.ana_book = bert_model(input_ids=ana_book['input_ids'],
                                       attention_mask=ana_book['attention_mask'])  # (**encoded_inputs)
            self.ana_book = self.ana_book.last_hidden_state[:, 0, :]
            self.disease_book = bert_model(input_ids=disease_book['input_ids'],
                                           attention_mask=disease_book['attention_mask'])  # (**encoded_inputs)
            self.disease_book = self.disease_book.last_hidden_state[:, 0, :]
        self.disease_embedding_layer = nn.Linear(768, 256)
        self.cl_fc = nn.Linear(256, 768)

        self.disease_name = [
            'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
            'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process',
            'abnormality', 'enlarge', 'tip', 'low',
            'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead',
            'disease', 'calcification', 'prominence',
            'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative',
            'pacemaker', 'thicken', 'marking', 'scar',
            'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd',
            'infiltrate', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware',
            'dilation', 'chf', 'redistribution', 'aspiration',
            'tail_abnorm_obs', 'excluded_obs'
        ]

        self.excluded_disease = [
            'pneumonia',
            'infiltrate',
            'mass',
            'nodule',
            'emphysema',
            'fibrosis',
            'thicken',
            'hernia'
        ]

        self.keep_class_dim = [self.disease_name.index(i) for i in self.disease_name if i not in self.excluded_disease]

        ###################################
        ''' Query Decoder'''
        ###################################

        self.H = config['H']
        decoder_layer = TransformerDecoderLayer(self.d_model, config['H'], 1024,
                                                0.1, 'relu', normalize_before=True)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, config['N'], decoder_norm,
                                          return_intermediate=False)

        # Learnable Queries
        # self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(config['dropout'])

        self.apply(self._init_weights)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        """
        Get the BERT base model.

        :param bert_model_name: str, name of the BERT model
        :param freeze_layers: list, list of layers to freeze
        :return: nn.Module, BERT model
        """
        try:
            model = AutoModel.from_pretrained(bert_model_name)  # , return_dict=True)
            print("text feature extractor:", bert_model_name)
        except ValueError as e:
            raise ValueError("Invalid model name. Check the config file and pass a BERT model from transformers "
                             "library") from e

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def forward(self):
        """
        :return: The embedded query.
        """
        B = self.batch_size
        query_embed = self.disease_embedding_layer(self.disease_book)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)  # Repeat the embeddings across the batch dimension

        return query_embed

    @staticmethod
    def _init_weights(module):
        """
        Initialize weights like BERT - N(0.0, 0.02), bias = 0.

        :param module: nn.Module, module to initialize
        :return: None
        """

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
