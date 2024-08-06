# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain
import logging
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace

import numpy as np
import torch
from fairseq import checkpoint_utils, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    GenerationConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, List, Optional, Tuple, Union
from omegaconf import OmegaConf

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"

@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})
    w2v_path: Optional[str] = field(default=None, metadata={"help": "path to avhubert model"})

@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )

def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for recognition!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos, generator.pad}

def _main(cfg, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("hybrid.speech_recognize")
    if output_file is not sys.stdout:  # also print to stdout
        logger.addHandler(logging.StreamHandler(sys.stdout))

    utils.import_user_module(cfg.common)
    if cfg.override.w2v_path is not None:
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path], arg_overrides={"w2v_path": cfg.override.w2v_path})
    else:
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path])
    models = [model.eval().cuda() for model in models]

    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)

    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()

    # Set dictionary
    dictionary = task.target_dictionary

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav
    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    if cfg.generation.match_source_len:
        logger.warning(
            "The option match_source_len is not applicable to speech recognition. Ignoring it."
        )
    gen_timer = StopwatchMeter()
    extra_gen_cls_kwargs = {
        "lm_model": lms[0],
        "lm_weight": cfg.generation.lm_weight,
    }
    cfg.generation.score_reference = False  #
    save_attention_plot = cfg.generation.print_alignment is not None
    cfg.generation.print_alignment = None  #
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    def decode_fn(x):
        symbols_ignore = get_symbols_to_strip_from_output(generator)
        symbols_ignore.add(dictionary.pad())
        if hasattr(task.datasets[cfg.dataset.gen_subset].label_processors[0], 'decode'):
            return task.datasets[cfg.dataset.gen_subset].label_processors[0].decode(x, symbols_ignore)
        chars = dictionary.string(x, extra_symbols_to_ignore=symbols_ignore)
        words = " ".join("".join(chars.split()).replace('|', ' ').split())
        return words

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    result_dict = {'utt_id': [], 'ref': [], 'hypo': []}
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        hypos, sample = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i in range(len(sample["id"])):
            ref_for_decode = sample['target'][i].int().cpu()
            ref_for_decode = ref_for_decode[:sample['target_lengths'][i]]

            hypo_for_decode = hypos[i][0]['tokens'].int().cpu()
            hypo_for_decode = hypo_for_decode[:sample['target_lengths'][i]]

            assert len(ref_for_decode) == len(hypo_for_decode)

            result_dict['utt_id'].append(sample['utt_id'][i])
            ref_sent = decode_fn(ref_for_decode)
            result_dict['ref'].append(ref_sent)
            best_hypo = hypo_for_decode
            hypo_str = decode_fn(best_hypo)
            result_dict['hypo'].append(hypo_str)
            logger.info(f"\nREF:{ref_sent}\nHYP:{hypo_str}\n")

            mel_save_path = os.path.join(cfg.common_eval.results_path, 'pred_mel', sample['utt_id'][i]+'.npy')
            os.makedirs(os.path.dirname(mel_save_path), exist_ok=True)
            np.save(mel_save_path, sample['mels'][i])

            unit_save_path = os.path.join(cfg.common_eval.results_path, 'pred_unit', sample['utt_id'][i]+'.txt')
            os.makedirs(os.path.dirname(unit_save_path), exist_ok=True)
            with open(unit_save_path, "w") as f:
                f.write(hypo_str)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += sample["nsentences"] if "nsentences" in sample else sample["id"].numel()

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Recognized {:,} utterances ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))

    yaml_str = OmegaConf.to_yaml(cfg.generation)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)
    n_err, n_total = 0, 0
    n_equal = 0
    assert len(result_dict['hypo']) == len(result_dict['ref'])
    for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
        hypo, ref = hypo.strip().split(), ref.strip().split()
        n_err += editdistance.eval(hypo, ref)
        assert len(hypo) == len(ref)
        n_equal += sum([h==f for h,f in zip(hypo, ref)])
        n_total += len(ref)
    wer = 100 * n_err / n_total
    accuracy = 100 * n_equal / n_total
    wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
    with open(wer_fn, "w") as fo:
        fo.write(f"WER: {wer}\n")
        fo.write(f"Accuracy: {accuracy}\n")
        fo.write(f"err / num_ref_words = {n_err} / {n_total}\n\n")
        fo.write(f"{yaml_str}")
    logger.info(f"WER: {wer}%")
    logger.info(f"Accuracy: {accuracy}%")
    return


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))
    return


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
