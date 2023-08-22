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
    from .model import MultiTargetEncoderModelConfig, MLP

logger = logging.getLogger(__name__)

@register_model("multi_target_avhubert", dataclass=MultiTargetEncoderModelConfig)
class MultiTargetAVHubertEncoderModel(FairseqEncoderModel):
    def __init__(self, encoder, tgt_dict, cfg, conformer):
        super().__init__(encoder)
        self.conformer = conformer
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )
        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        cfg.decoder_embed_dim = len(tgt_dict)

        conformer = None
        if cfg.use_conformer:
            conformer = Conformer(cfg)

        return cls(encoder, tgt_dict, cfg, conformer)


    def forward(self, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
        if self.cfg.use_conformer:
            output = self.conformer(
                source=output['encoder_out'].repeat_interleave(2, dim=0),
                padding_mask=output['encoder_padding_mask'].repeat_interleave(2, dim=1),
                spk_emb=kwargs['spk_emb']
            )
        
        ### T, B, V
        output['encoder_out'] = output['encoder_out'].transpose(0,1).contiguous()

        return output

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

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class Conformer(FairseqEncoder):
    def __init__(self, cfg: AVHubertAsrConfig, tgt_dict=None):
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
        self.encoder.frontend = None

        d = cfg.conformer_embed_dim

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.num_updates = 0

        if getattr(cfg.w2v_args.model, "encoder_embed_dim", d) != d:
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

        x = source

        if tbc:
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

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