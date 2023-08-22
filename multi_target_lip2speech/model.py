import sys,logging
import contextlib
from argparse import Namespace

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model

DBG=True if len(sys.argv) == 1 else False

if DBG:
    pass
else:
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertAsrConfig, AVHubertSeq2SeqConfig, Linear, Embedding
    from pathlib import Path
    sys.path.insert(0, Path(__file__).resolve().parent.parent)
    from espnet.nets.pytorch_backend.transformer.encoder import Encoder
    sys.path.pop(0)

logger = logging.getLogger(__name__)

@dataclass
class MultiTargetEncoderModelConfig(AVHubertSeq2SeqConfig):
    use_conformer: bool = field(default=False)
    conformer_layers: int = field(default=12)
    conformer_embed_dim: int = field(default=512)
    conformer_ffn_embed_dim: int = field(default=2048)
    conformer_attention_heads: int = field(default=8)
    conformer_dropout: float = field(default=0.1)
    conformer_attention_dropout: float = field(default=0.1)
    conformer_layer_norm_first: bool = field(default=True)

@register_model("multi_target", dataclass=MultiTargetEncoderModelConfig)
class MultiTargetEncoderModel(FairseqEncoderModel):
    def __init__(self, conformer, tgt_dict, cfg):
        super().__init__(conformer)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        cfg.decoder_embed_dim = len(tgt_dict)

        conformer = None
        if cfg.use_conformer:
            conformer = Conformer(cfg)

        return cls(conformer, tgt_dict, cfg)


    def forward(self, **kwargs):
        if self.cfg.use_conformer:
            output = self.encoder(**kwargs)

        output['encoder_out'] = output['encoder_out'].transpose(0,1).contiguous()

        return output

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class Conformer(FairseqEncoder):
    def __init__(self, cfg, tgt_dict=None):
        super().__init__(None)

        self.encoder = Encoder(
            idim=-1,
            attention_dim=cfg.conformer_embed_dim, # adim
            attention_heads=cfg.conformer_attention_heads, # aheads
            linear_units=cfg.conformer_ffn_embed_dim, # eunits
            num_blocks=cfg.conformer_layers, #elayers
            dropout_rate=cfg.conformer_dropout, # dropout_rate
            positional_dropout_rate=cfg.conformer_dropout, # dropout_rate
            attention_dropout_rate=cfg.conformer_attention_dropout, # transformer_attn_dropout_rate
            input_layer="conv3d", # transformer_input_layer
            normalize_before=cfg.conformer_layer_norm_first,
            macaron_style=1, # macaron_style
            encoder_attn_layer_type="rel_mha", # transformer_encoder_attn_layer_type
            use_cnn_module=1, # use_cnn_module
            zero_triu=False, # zero_triu
            cnn_module_kernel=31, # cnn_module_kernel
            relu_type="swish", # relu_type
            a_upsample_ratio=1, # a_upsample_ratio,
        )
        # self.encoder.frontend = None

        d = cfg.conformer_embed_dim

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.num_updates = 0

        if 512 != d:
            self.proj_in = Linear(cfg.w2v_args.model.encoder_embed_dim, d)
        else:
            self.proj_in = None

        if tgt_dict is not None:
            # self.proj_out = Linear(d, len(tgt_dict))
            self.proj_out = MLP(
                d, [d, d, len(tgt_dict)], cfg.final_dropout, nn.GELU
            )
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj_out = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj_out = None

        self.mel_conv = nn.Sequential(
            nn.Conv1d(in_channels=d+256,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.final_dropout),
            nn.GELU(),
            nn.Conv1d(in_channels=d,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.final_dropout),
            nn.GELU(),
            nn.Conv1d(in_channels=d,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.final_dropout),
            nn.GELU(),
        )
        self.mel_proj = Linear(d, 160)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, spk_emb=None, tbc=True, **kwargs):

        x = source["video"]

        x = self.encoder.frontend(x.squeeze(1))

        x = x.repeat_interleave(2, dim=1)
        padding_mask = padding_mask.repeat_interleave(2, dim=1)

        if self.proj_in:
            x = self.proj_in(x)

        x, masks = self.encoder.forward_after_frontend(
            x,
            masks = ~padding_mask.unsqueeze(-2),
        )

        padding_mask = ~masks.squeeze(-2)

        if spk_emb is not None:
            assert spk_emb.size(-1) == 256
            spk_x = torch.cat([spk_emb.unsqueeze(1).repeat(1,x.size(1),1), x], dim=-1)
        else:
            spk_x = x

        encoder_out_mel = self.mel_proj(self.mel_conv(spk_x.transpose(1,2)).transpose(1,2))

        B, T, D = encoder_out_mel.shape
        encoder_out_mel = encoder_out_mel.reshape(B, T, D//2, 2).transpose(-1,-2).reshape(B, T*2, D//2)

        if tbc:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj_out:
            x = self.proj_out(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
            "encoder_out_mel": encoder_out_mel,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
    

import math
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims,
        dropout: float = 0.1,
        nonlinearity = nn.ReLU,
        normalization = None, #nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
    ):
        super(MLP, self).__init__()
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        x = self.last_layer(x)
        return x