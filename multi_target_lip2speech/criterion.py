# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, LabelSmoothedCrossEntropyCriterionConfig

@dataclass
class MultiTargetCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    mel_weight: float = field(default=1., metadata={"help": "weight for mel loss"})

@register_criterion("multi_target", dataclass=MultiTargetCriterionConfig)
class LabelSmoothedCrossEntropyCriterionLengthMatch(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        mel_weight,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

        self.criterion_l1 = torch.nn.L1Loss(reduction='none')
        self.criterion_sc = SpectralConvergenceLoss()

        self.mel_weight = mel_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        if net_output["encoder_out_mel"] is not None:
            pred, targ = net_output["encoder_out_mel"], sample["mel"]
            targ_mask = ~sample['net_input']['padding_mask'].repeat_interleave(4, dim=1)

            crop_len = min(targ_mask.sum(1).max().item(), pred.size(1), targ.size(1))

            pred = pred[:,:crop_len].contiguous()
            targ = targ[:,:crop_len].contiguous()
            targ_mask = targ_mask[:,:crop_len].contiguous()

            pred_list, targ_list = [], []
            for p, t, m in zip(pred, targ, targ_mask):
                pred_list.append(p[m])
                targ_list.append(t[m])

            if self.sentence_avg:
                mel_loss = ((self.criterion_l1(pred, targ).mean(-1) * targ_mask).sum(1) / targ_mask.sum(1)).sum()
            else:
                mel_loss = (self.criterion_l1(pred, targ).mean(-1) * targ_mask).sum()

            sc_loss = self.criterion_sc(pred_list, targ_list, self.sentence_avg)

            mel_loss += sc_loss

            loss += mel_loss * self.mel_weight

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "mel_loss": utils.item(mel_loss.data) if net_output["encoder_out_mel"] is not None else None,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        lprobs = lprobs[:, :min(lprobs.size(1), target.size(1))]
        target = target[:, :min(lprobs.size(1), target.size(1))]
        target = target.to(dtype=torch.int64)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.reshape(-1, lprobs.size(-1)), target.reshape(-1)

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if logging_outputs[0]["mel_loss"] is not None:
            ctc_loss_sum = sum(log.get("mel_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "mel_loss", ctc_loss_sum / sample_size, sample_size, round=5
            )

class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag_list, y_mag_list, sentence_avg):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        loss = 0.
        for x_mag, y_mag in zip(x_mag_list, y_mag_list):
            loss_one_sample = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            loss += loss_one_sample if sentence_avg else loss_one_sample * len(y_mag)
        return loss