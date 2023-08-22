# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os, glob
import sys
from typing import List

from fairseq import search
from dataclasses import dataclass, field
from fairseq.data import Dictionary
from fairseq.tasks import register_task

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from dataset import MultiTargetDataset
    from ..avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask, LabelEncoder
else:
    from .dataset import MultiTargetDataset
    from avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask, LabelEncoder

logger = logging.getLogger(__name__)

class LabelEncoderUnit(LabelEncoder):
    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=True, add_if_not_exist=False,
        ).long()

    def decode(self, tok, symbols_ignore=None):
        return self.dictionary.string(tok, extra_symbols_to_ignore=symbols_ignore)


@dataclass
class Lip2SpeechConfig(AVHubertPretrainingConfig):
    time_mask: bool = field(default=False)
    random_erase: bool = field(default=False)

@register_task("lip2speech", dataclass=Lip2SpeechConfig)
class Lip2SpeechTask(AVHubertPretrainingTask):
    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dictionaries = [self.target_dictionary] if self.fine_tuning else self.dictionaries
        pad_list = [dictionary.pad() for dictionary in dictionaries]
        eos_list = [dictionary.eos() for dictionary in dictionaries]

        if not self.cfg.is_s2s:
            procs = [LabelEncoderUnit(dictionary) for dictionary in dictionaries]
        else:
            logger.info(f"Using tokenizer")
            bpe_tokenizer = self.s2s_tokenizer
            procs = [LabelEncoderS2SToken(dictionary, bpe_tokenizer) for dictionary in dictionaries]
        paths = [
            f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels
        ]
        image_aug = self.cfg.image_aug if split == 'train' else False
        noise_fn, noise_snr = f"{self.cfg.noise_wav}/{split}.tsv" if self.cfg.noise_wav is not None else None, eval(self.cfg.noise_snr)
        noise_num = self.cfg.noise_num # 

        self.datasets[split] = MultiTargetDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_trim_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=False,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            stack_order_audio=self.cfg.stack_order_audio,
            skip_verify=self.cfg.skip_verify,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            image_aug=image_aug,
            modalities=self.cfg.modalities,
            is_s2s=self.cfg.is_s2s,
            noise_fn=noise_fn,
            noise_prob=self.cfg.noise_prob,
            noise_snr=noise_snr,
            noise_num=noise_num,
            time_mask=self.cfg.time_mask,
            random_erase=self.cfg.random_erase,
        )

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        if seq_gen_cls is None:
            from .sequence_generator import MultiTargetSequenceGenerator
            seq_gen_cls = MultiTargetSequenceGenerator
        return super().build_generator(models, args, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None)