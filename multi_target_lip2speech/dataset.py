import itertools
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from scipy.io import wavfile

DBG=True if len(sys.argv) == 1 else False

if DBG:
    import utils_aug as custom_utils
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
        stream=sys.stdout,
    )
    from ..avhubert.hubert_dataset import load_audio_visual, load_label, load_label_offset, verify_label_lengths, AVHubertDataset
else:
    from . import utils_aug as custom_utils
    from avhubert.hubert_dataset import load_audio_visual, load_label, load_label_offset, verify_label_lengths, AVHubertDataset

logger = logging.getLogger(__name__)


class MultiTargetDataset(AVHubertDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            label_paths: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            pad_list: List[str],
            eos_list: List[str],
            label_processors: Optional[List[Any]] = None,
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            stack_order_audio: int=1,
            skip_verify: bool=False,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            is_s2s=False,
            noise_fn=None,
            noise_prob=0,
            noise_snr=0,
            noise_num=1,
            time_mask: bool = False,
            random_erase: bool = False,
    ):
        # self.label_rates = (
        #     [label_rates for _ in range(len(label_paths))]
        #     if isinstance(label_rates, int)
        #     else label_rates
        # )
        self.label_rates = [-1 for _ in range(len(label_paths))]
        self.modalities = set(modalities)
        self.audio_root, self.names, inds, tot, self.sizes = load_audio_visual(manifest_path, max_keep_sample_size, min_keep_sample_size, frame_rate=sample_rate, label_paths=label_paths, label_rates=self.label_rates)
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.store_labels = store_labels
        self.is_s2s = is_s2s
        self.noise_wav, self.noise_prob, self.noise_snr, self.noise_num = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else [], noise_prob, noise_snr, noise_num

        # assert self.single_target == (self.label_rates[0] == -1), f"single target should be equivalent to sequence label (label_rate==-1)"
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert (
            label_processors is None
            or len(label_processors) == self.num_labels
        )
        if not skip_verify:
            for label_path, label_rate in zip(label_paths, self.label_rates):
                verify_label_lengths(self.sizes, self.sample_rate, label_path, label_rate, inds, tot)
        else:
            logger.info(f"Skip label alignment verifying")

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std) ]
                + ([custom_utils.RandomErase(0.5)] if random_erase else [])
                + ([custom_utils.TimeMask()] if time_mask else []) )
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        logger.info(f"image transform: {self.transform}")

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")
        logger.info(
            f"Noise wav: {noise_fn}->{len(self.noise_wav)} wav, Prob: {self.noise_prob}, SNR: {self.noise_snr}, Number of mixture: {self.noise_num}"
        )

    def load_additional_feature(self, mix_name):
        video_fn, audio_fn = mix_name

        mel_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/mel/')[:-4]+'.npy'
        if os.path.exists(mel_fn):
            mel = np.load(mel_fn)
        else:
            raise FileNotFoundError(f"{mel_fn} does not exist")

        spk_emb_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/spk_emb/')[:-4]+'.npy'
        if os.path.exists(spk_emb_fn):
            spk_emb = np.load(spk_emb_fn)
        else:
            raise FileNotFoundError(f"{spk_emb_fn} does not exist")

        return mel, spk_emb

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        mel, spk_emb = self.load_additional_feature(self.names[index])

        mel = torch.from_numpy(mel.astype(np.float32))
        spk_emb = torch.from_numpy(spk_emb.astype(np.float32))

        sample["mel"] = mel
        sample["spk_emb"] = spk_emb

        return sample

    def collater(self, samples):
        batch = super().collater(samples)

        max_mel_len = max(len(s["mel"]) for s in samples)
        batch["mel"] = torch.stack([torch.nn.functional.pad(s["mel"], [0, 0, 0, max_mel_len - len(s["mel"])]) for s in samples])

        batch["net_input"]["spk_emb"] = torch.stack([s["spk_emb"] for s in samples])

        return batch