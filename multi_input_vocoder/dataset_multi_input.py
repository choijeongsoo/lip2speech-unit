import json
import os
import random
from pathlib import Path

# import amfm_decompy.basic_tools as basic
# import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import soundfile as sf
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "speech-resynthesis"))
from dataset import MAX_WAV_VALUE, load_audio, mel_spectrogram, CodeDataset
sys.path.pop(0)

def parse_manifest(manifest_path, max_keep=None, min_keep=None):
    n_long, n_short, n_unaligned = 0, 0, 0
    audio_files, sizes, mels, codes = [], [], [], []

    code_path = os.path.splitext(manifest_path)[0]+".unt"
    print(f'manifest_path: {manifest_path}')
    print(f'code_path: {code_path}')

    with open(manifest_path) as f:

        with open(code_path) as f_c:
            root = f.readline().strip()
            for ind, (line, line_code) in enumerate(zip(f, f_c)):
                items = line.strip().split("\t")
                code = line_code.strip().split("|")[-1]

                sz = int(items[-2])

                try:
                    diff = len(code.split()) - sz * 2
                    assert -2 <= diff <= 2, "code length != video length * 2"
                except:
                    import pdb
                    pdb.set_trace()

                if min_keep is not None and sz < min_keep:
                    n_short += 1
                elif max_keep is not None and sz > max_keep:
                    n_long += 1
                else:
                    audio_path = os.path.join(root, items[2])
                    audio_files.append(audio_path)
                    sizes.append(sz)
                    mels.append(audio_path.replace('/audio/', '/mel/')[:-4]+'.npy')
                    codes.append(code)

    print(
        f"example_audio_file: {audio_files[0]}, "
        f"example_code: {codes[0]}, "
        f"example_mel: {mels[0]}"
    )

    print(
        f"max_keep={max_keep}, min_keep={min_keep}, "
        f"loaded {len(audio_files)}, skipped {n_short} short and {n_long} long and {n_unaligned} unaligned, "
        f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
    )

    return audio_files, mels, codes

def get_dataset_filelist(h):
    training_files, training_mels, training_codes = parse_manifest(h.input_training_file, h.get("max_keep", None), h.get("min_keep", None))
    validation_files, validation_mels, validation_codes = parse_manifest(h.input_validation_file, h.get("max_keep", None), h.get("min_keep", None))

    return (training_files, training_mels, training_codes), (validation_files, validation_mels, validation_codes)

def load_code_dict(path):
    with open(path, 'r') as f:
        codes = [line.rstrip().rsplit(" ", 1)[0] for line in f]
    code_dict = {c: i for i, c in enumerate(codes)}

    assert(set(code_dict.values()) == set(range(len(code_dict))))

    return code_dict

def code_to_sequence(code, code_dict, collapse_code):
    if collapse_code:
        prev_c = None
        sequence = []
        for c in code:
            if c in code_dict and c != prev_c:
                sequence.append(code_dict[c])
                prev_c = c
    else:
        sequence = [code_dict[c] for c in code if c in code_dict]
        if len(sequence) < 0.95 * len(code):
            print('WARNING : over 5%% codes are OOV')

    return sequence


import torchvision
class MelCodeDataset(CodeDataset):
    def __init__(self, training_files, segment_size, code_hop_size, mel_hop_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, multispkr=False, pad=None, code_dict_path=None,
                 use_blur=False, blur_kernel_size=None, blur_sigma=None, use_noise=False, noise_factor=None):
        self.audio_files, self.mel_files, self.codes = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.mel_hop_size = mel_hop_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.multispkr = multispkr
        self.pad = pad
        if self.multispkr:
            self.speaker_emb_files = [f.replace('/audio/', '/spk_emb/')[:-4]+'.npy' for f in self.audio_files]

        self.code_dict = load_code_dict(code_dict_path)
        self.collapse_code = False

        self.use_blur = use_blur
        if self.use_blur:
            self.blur = torchvision.transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
        self.use_noise = use_noise
        if self.use_noise:
            self.noise_factor = noise_factor

    def process_code(self, inp_str):
        inp_toks = inp_str.split()
        return code_to_sequence(inp_toks, self.code_dict, self.collapse_code)

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if sampling_rate != self.sampling_rate:
                # raise ValueError("{} SR doesn't match target {} SR".format(
                #     sampling_rate, self.sampling_rate))
                import resampy
                audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

            if self.pad:
                padding = self.pad - (audio.shape[-1] % self.pad)
                audio = np.pad(audio, (0, padding), "constant", constant_values=0)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        # Trim audio ending
        code = np.array(self.process_code(self.codes[index]))
        code_length = min(audio.shape[0] // self.code_hop_size, code.shape[0])
        code = code[:code_length]

        mel = np.load(self.mel_files[index])
        mel_length = min(audio.shape[0] // self.mel_hop_size, mel.shape[0])
        mel = mel[:mel_length]

        cut_length = min(mel_length * self.mel_hop_size, code_length * self.code_hop_size)

        mel = mel[:cut_length // self.mel_hop_size]
        code = code[:cut_length // self.code_hop_size]
        audio = audio[:cut_length]
        assert audio.shape[0] // self.code_hop_size == code.shape[0], "Code audio mismatch"
        assert audio.shape[0] // self.mel_hop_size == mel.shape[0], "Mel audio mismatch"

        mel = mel.transpose(1,0)
        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            code = np.hstack([code, code])
            mel = np.hstack([mel, mel])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if len(mel.shape) < 3:
            mel = mel[None, :]

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        audio, code, mel = self._sample_interval([audio, code, mel])

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        feats = {"code": code.squeeze(), "mel": mel.squeeze()}

        if self.multispkr:
            feats['spkr'] = self._get_speaker_emb(index)

        return feats, audio.squeeze(0), str(filename), mel_loss.squeeze()

    def _get_speaker_emb(self, idx):
        return np.load(self.speaker_emb_files[idx])