# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import argparse
import glob
import json
import os
import random
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import librosa
import numpy as np
import torch
from scipy.io.wavfile import write

from dataset_multi_input import MelCodeDataset, parse_manifest
from models_multi_input import MelCodeGenerator

sys.path.insert(0, str(Path(__file__).parent.parent / "speech-resynthesis"))
from dataset import MAX_WAV_VALUE, mel_spectrogram
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, \
    save_checkpoint, build_env, AttrDict
sys.path.pop(0)


h = None
device = None


def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def generate(h, generator, code):
    start = time.time()
    y_g_hat = generator(**code)
    if type(y_g_hat) is tuple:
        y_g_hat = y_g_hat[0]
    rtf = (time.time() - start) / (y_g_hat.shape[-1] / h.sampling_rate)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio, rtf


def init_worker(queue, arguments):
    import logging
    logging.getLogger().handlers = []

    global generator
    global dataset
    global idx
    global device
    global a
    global h

    a = arguments
    idx = queue.get()
    device = 0 #idx

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = MelCodeGenerator(h).to(device)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g['generator'])

    if a.code_file is not None:
        file_list = glob.glob(f"{a.code_file}/*/*/*.txt")
        def parse_code(c):
            c = [int(v) for v in c.split(" ")]
            return torch.LongTensor(c).numpy()
        dataset = []
        for code_path in file_list:
            mel_path = code_path.replace("/pred_unit/", "/pred_mel/")[:-4]+".npy"
            spk_emb_path = code_path.replace(f"{a.code_file}/", "../datasets/lrs3/spk_emb/")[:-4]+".npy"

            code_str = open(code_path).readline().strip()
            code = parse_code(code_str)
            mel = np.load(mel_path).transpose(1,0)
            spk_emb = np.load(spk_emb_path)

            dataset.append([{"code": code, "mel": mel, "spkr": spk_emb}, None, code_path, None])
    else:
        file_list = parse_manifest(a.input_code_file)
        dataset = MelCodeDataset(file_list, -1, h.code_hop_size, h.mel_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                              h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device,
                              multispkr=h.get('multispkr', None),
                              pad=a.pad,
                              code_dict_path=h.code_dict_path)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()

    # fix seed
    seed = 52 + idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def inference(item_index):
    code, gt_audio, filename, _ = dataset[item_index]
    code = {k: torch.from_numpy(v).to(device).unsqueeze(0) for k, v in code.items()}

    fname_out_name = os.path.join("pred_wav", *(filename.split('/')[-3:]))[:-4]

    new_code = dict(code)

    audio, rtf = generate(h, generator, new_code)
    output_file = os.path.join(a.output_dir, fname_out_name + '.wav')
    audio = librosa.util.normalize(audio.astype(np.float32))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    write(output_file, h.sampling_rate, audio)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', default=None)
    parser.add_argument('--input_code_file', default='./datasets/LJSpeech/cpc100/test.txt')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--pad', default=None, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-n', type=int, default=10)
    a = parser.parse_args()

    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ids = list(range(8))
    manager = Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    if not os.path.isfile(cp_g) or not os.path.exists(cp_g):
        print(f"Didn't find checkpoints for {cp_g}")
        return

    if a.code_file is not None:
        file_list = glob.glob(f"{a.code_file}/*/*/*.txt")
        def parse_code(c):
            c = [int(v) for v in c.split(" ")]
            return torch.LongTensor(c).numpy()
        dataset = []
        for code_path in file_list:
            mel_path = code_path.replace("/pred_unit/", "/pred_mel/")[:-4]+".npy"
            spk_emb_path = code_path.replace(f"{a.code_file}/", "../datasets/lrs3/spk_emb/")[:-4]+".npy"

            code_str = open(code_path).readline().strip()
            code = parse_code(code_str)
            mel = np.load(mel_path).transpose(1,0)
            spk_emb = np.load(spk_emb_path)

            dataset.append([{"code": code, "mel": mel, "spkr": spk_emb}, None, code_path, None])
    else:
        file_list = parse_manifest(a.input_code_file)
        dataset = MelCodeDataset(file_list, -1, h.code_hop_size, h.mel_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                              h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device,
                              multispkr=h.get('multispkr', None),
                              pad=a.pad,
                              code_dict_path=h.code_dict_path)

    if a.debug:
        ids = list(range(1))
        import queue
        idQueue = queue.Queue()
        for i in ids:
            idQueue.put(i)
        init_worker(idQueue, a)

        for i in range(0, len(dataset)):
            inference(i)
            bar = progbar(i, len(dataset))
            message = f'{bar} {i}/{len(dataset)} '
            stream(message)
            if a.n != -1 and i > a.n:
                break
    else:
        idx = list(range(len(dataset)))
        random.shuffle(idx)
        with Pool(8, init_worker, (idQueue, a)) as pool:
            for i, _ in enumerate(pool.imap(inference, idx), 1):
                bar = progbar(i, len(idx))
                message = f'{bar} {i}/{len(idx)} '
                stream(message)
                if a.n != -1 and i > a.n:
                    break


if __name__ == '__main__':
    main()
