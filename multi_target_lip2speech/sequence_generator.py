# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from avhubert.sequence_generator import SequenceGenerator
class MultiTargetSequenceGenerator(SequenceGenerator):
    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input. input keys: " + str(net_input.keys()))

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        if src_tokens['audio'] is not None:
            bsz, src_len = src_tokens['audio'].size()[:2]
            src_device = src_tokens['audio'].device
        else:
            bsz, src_len = net_input['padding_mask'].size()
            src_device = src_tokens['video'].device

        ###
        sample['target_lengths'] = src_lengths * 2
        max_len = sample['target_lengths'].max().item()
        sample['target'] = sample['target'][:, :max_len]
        if sample['target'].size(-1) < max_len:
            sample['target'] = F.pad(sample['target'], (0, max_len-sample['target'].size(-1)), "constant", self.pad)
        for i, target_length in enumerate(sample['target_lengths']):
            sample['target'][i][target_length:] = self.pad
            pos_eos = (sample['target'][i]==self.eos).nonzero()
            if pos_eos.nelement() > 0 and pos_eos[0].item() > 0:
                pos_eos = pos_eos[0].item()
                sample['target'][i][pos_eos:] = sample['target'][i][pos_eos-1]
        ###

        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        if hasattr(self.model.single_model, 'conformer'):
            encoder_outs = [
                model.conformer(encoder_outs[i]['encoder_out'].repeat_interleave(2, dim=0),
                                encoder_outs[i]['encoder_padding_mask'].repeat_interleave(2, dim=1),
                                net_input['spk_emb'])
                for i, model in enumerate(self.model.models)
            ]

        encoder_out = encoder_outs[0]['encoder_out']
        encoder_out_mel = encoder_outs[0]['encoder_out_mel']

        if encoder_out_mel is not None:
            mels = encoder_out_mel.cpu().numpy()
            mels = [mel[:target_length*2] for mel, target_length in zip(mels, sample['target_lengths'])]
            sample['mels'] = mels

        encoder_out[..., self.tgt_dict.bos()] = -math.inf
        encoder_out[..., self.pad] = -math.inf
        encoder_out[..., self.eos] = -math.inf
        encoder_out[..., self.unk] = -math.inf

        output = encoder_out.argmax(dim=-1).transpose(0,1)
        finalized = [[{"tokens": tokens}] for tokens in output]
        return finalized, sample
