# Intelligible Lip-to-Speech Synthesis with Speech Units

Official PyTorch implementation for the following paper:
> **Intelligible Lip-to-Speech Synthesis with Speech Units**<br>
> [Jeongsoo Choi](https://choijeongsoo.github.io), [Minsu Kim](https://sites.google.com/view/ms-dot-k), [Yong Man Ro](https://www.ivllab.kaist.ac.kr/people/professor)<br>
> Interspeech 2023<br>
> \[[Paper](https://arxiv.org/abs/2305.19603)\] \[[Project](https://choijeongsoo.github.io/lip2speech-unit)\]

<div align="center"><img width="80%" src="imgs/fig1.png?raw=true"/></div>

## Installation
```
conda create -y -n lip2speech python=3.10
conda activate lip2speech
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

git clone https://github.com/choijeongsoo/lip2speech-unit.git
cd lip2speech-unit
git clone https://github.com/facebookresearch/fairseq.git
cd faireq
git checkout afc77bd
pip install -e ./
cd ..

```

## Data Preparation
#### Video and Audio
* reference: https://github.com/facebookresearch/av_hubert/tree/main/avhubert/preparation
- `${ROOT}/datasets/${DATASET}/audio` for processed audio files
- `${ROOT}/datasets/${DATASET}/video` for processed video files
- `${ROOT}/datasets/${DATASET}/label/*.tsv` for training manifests

#### Speech Units
* reference: https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit
- HuBERT Base + KM200

#### Speaker Embedding
* reference: https://github.com/CorentinJ/Real-Time-Voice-Cloning

#### Mel-spectrogram
* reference: https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/unit2speech/tacotron2
- config

    ```
    filter_length: 640
    hop_length: 160
    win_length: 640
    n_mel_channels: 80
    sampling_rate: 16000
    mel_fmin: 0.0
    mel_fmax: 8000.0
    ```

We provide sample data in 'datasets/lrs3' directory.
    
## Model Checkpoints

<details open>

<summary>Lip Reading Sentences 3 (LRS3)</summary>

<p> </p>

|     1st stage   |    2nd stage    | STOI | ESTOI | PESQ | WER(%) |
|:---------------:|:---------------:|:----:|:----:|:----:|:----:|
| [Multi-target Lip2Speech](https://drive.google.com/file/d/1sFtoczuEmQaQXszCadCnNn6Itiohn5bN/view?usp=sharing) | [Multi-input Vocoder](https://drive.google.com/file/d/1WdbOFwUy-0eGvK2vT691ZsbqRAN9_Tgw/view?usp=sharing) | 0.552 | 0.354 | 1.31 | 50.4 |
| [Multi-target Lip2Speech](https://drive.google.com/file/d/1sFtoczuEmQaQXszCadCnNn6Itiohn5bN/view?usp=sharing) | [Multi-input Vocoder<br/>+ augmentation](https://drive.google.com/file/d/13zimLyyXluQ2RuXbBk2b3S9LnBnfLptj/view?usp=sharing) | 0.543 | 0.351 | 1.28 | 50.2 |
| [Multi-target Lip2Speech<br/>+ AV-HuBERT](https://drive.google.com/file/d/1oS80l6zpIfMTVKwvaHUSOC9ByjzGibSp/view?usp=sharing) | [Multi-input Vocoder<br/>+ augmentation](https://drive.google.com/file/d/13zimLyyXluQ2RuXbBk2b3S9LnBnfLptj/view?usp=sharing) | 0.578 | 0.393 | 1.31 | 29.8 |

</details>

<details open>

<summary>Lip Reading Sentences 2 (LRS2)</summary>

<p> </p>

|     1st stage   |    2nd stage    | STOI | ESTOI | PESQ | WER(%) |
|:---------------:|:---------------:|:----:|:----:|:----:|:----:|
| [Multi-target Lip2Speech](https://drive.google.com/file/d/1aTv0e-TjD9AsVeijomCw_zAZxzE8Lhv-/view?usp=sharing) | [Multi-input Vocoder](https://drive.google.com/file/d/1tzI-LdOauWr6VC3zMHuL-HZcQu_QTCqX/view?usp=sharing) | | | | |
| [Multi-target Lip2Speech](https://drive.google.com/file/d/1aTv0e-TjD9AsVeijomCw_zAZxzE8Lhv-/view?usp=sharing) | [Multi-input Vocoder<br/>+ augmentation](https://drive.google.com/file/d/1WEZM0ICZdnafaC8ASwzIKMp_6fgUlYrs/view?usp=sharing) | 0.565 | 0.395 | 1.32 | 44.8 |
| [Multi-target Lip2Speech<br/>+ AV-HuBERT](https://drive.google.com/file/d/1meL4ZrSLgFEe0xh1yvejQxXunE88dvDf/view?usp=sharing) | [Multi-input Vocoder<br/>+ augmentation](https://drive.google.com/file/d/1WEZM0ICZdnafaC8ASwzIKMp_6fgUlYrs/view?usp=sharing) | 0.585 | 0.412 | 1.34 | 35.7 |

     
</details>

## Training
```
scripts/${DATASET}/train.sh
```
in 'multi_target_lip2speech' and 'multi_input_vocoder' directory

## Inference
```
scripts/${DATASET}/inference.sh
```
in 'multi_target_lip2speech' and 'multi_input_vocoder' directory

## Acknowledgement

This repository is built using [Fairseq](https://github.com/pytorch/fairseq), [AV-HuBERT](https://https://github.com/facebookresearch/av_hubert), [ESPnet](https://github.com/espnet/espnet), [speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis). We appreciate the open source of the projects.

## Citation

If our work is useful for your research, please cite the following paper:
```bibtex
@article{choi2023intelligible,
      title={Intelligible Lip-to-Speech Synthesis with Speech Units},
      author={Jeongsoo Choi and Minsu Kim and Yong Man Ro},
      journal={arXiv preprint arXiv:2305.19603},
      year={2023},
}
```
