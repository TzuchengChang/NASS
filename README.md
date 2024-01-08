# Noise-Aware Speech Separation (NASS)

**NOTE: This paper has been accepted by ICASSP 2024!**

This repository provides the examples of Sepformer (NASS) on Libri2Mix based on [SpeechBrain](https://github.com/speechbrain/speechbrain).

## Install with GitHub

Once you have created your Python environment (Python 3.7+) you can simply type:

```shell
git clone https://github.com/TzuchengChang/NASS
cd NASS
pip install -r requirements.txt
pip install --editable .
pip install mir-eval
pip install pyloudnorm
```

## Introduction

| ![Image 1](/blob/main/resources/figure1.png) |
|:-----------------------------------:|

|                                                                                                                                                                                                                                                         ![Image 1](/blob/main/resources/figure.png)                                                                                                                                                                                                                                                          |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Fig1. The overall pipeline of NASS. $x_n$ and $\hat n$ denote the noisy input and predicted noise. $\hat{s}_1$ and $\hat{s}_2$ are separated speech while $s_1$ and $s_2$ are the ground-truth. $h_{\hat s_1}$, $h_{\hat s_2}$ and $h_{\hat n}$ in dashed box are predicted representations, while $h_{s_1}$ and $h_{s_2}$ in solid box are the ground-truth. ``P" denotes the mutual information between separated and ground-truth speech is maximized while ``N" denotes the mutual information between separated speech and noise is minimized. |

|                                                                                                                                                                      <img src="/blob/main/resources/figure2.png" alt="Image 1" style="zoom: 25%;" />                                                                                                                                                                       |                                                        <img src="/blob/main/resources/figure3.png" alt="Image 2" style="zoom: 200%;" />                                                         |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Fig2.  The illustration of patch-wise contrastive learning. For the $i$-th sampling of $K$ times, one query example $r^i_q$, positive example $r^i_p$ and $M$ negative examples ${r_n^{i,j}}$ ($j \in [1,M]$) are sampled from predicted speech representation $h_{\hat s_a}$, ground-truth speech representation $h_{s_a}$ and predicted noise representation $h_{\hat n}$, respectively, "CS" denotes cosine similarity. | Fig3. Spectrum results on Libri2mix with Sepformer. Subplot (a) denotes the mixture; (b), (c) are baseline results; (d), (e), (f) are NASS results. Note that (d) is the noise output. |

In this paper, we propose a noise-aware SS (NASS) method, which aims to improve the speech quality for separated signals under noisy conditions. Specifically, NASS views background noise as an additional output and predicts it along with other speakers in a mask-based manner. To effectively denoise, we introduce patch-wise contrastive learning (PCL) between noise and speaker representations from the decoder input and encoder output. PCL loss aims to minimize the mutual information between predicted noise and other speakers at multiple-patch level to suppress the noise information in separated signals. Experimental results show that NASS achieves 1 to 2dB SI-SNRi or SDRi over DPRNN and Sepformer on WHAM! and LibriMix noisy datasets, with less than 0.1M parameter increase.

## NASS Example #####

We also provide a true example from Ted Cruz with -2dB WHAM! noise mixed. 

Results are from Sepformer(NASS) trained on Libri2Mix. 

|                           Mixture                            |                          Speaker 1                           |                          Speaker 2                           |                            Noise                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [Download](https://github.com/TzuchengChang/NASS/raw/main/resources/item1_mix.wav) | [Download](https://github.com/TzuchengChang/NASS/raw/main/resources/item1_source1hat.wav) | [Download](https://github.com/TzuchengChang/NASS/raw/main/resources/item1_source2hat.wav) | [Download](https://github.com/TzuchengChang/NASS/raw/main/resources/item1_source3hat.wav) |

## Run NASS Method #####

Step1: Prepare datasets. 
Please refer to [LibriMix repository](https://github.com/JorisCos/LibriMix).

Step2: Modify configurations.
Configuration files are saved in `NASS/recipes/LibriMix/separation/hparams/`

Step3: Run NASS method.

```shell
cd NASS/recipes/LibriMix/separation/
python train.py hparams/sepformer-libri2mix.yaml --data_folder /yourpath/Libri2Mix/
```

We also provide a yaml for custom data, and make sure your custom folder structure is like Libri2Mix.

```shell
python train.py hparams/sepformer-libri2mix-custom.yaml
 --data_folder /yourpath/custom/
```

## Pretrained Model #####

We provide a pretrained model on [github releases](https://github.com/TzuchengChang/NASS/releases/tag/Pretrained_Model).

To use it, download "results.zip" and unzip it to `NASS/recipes/LibriMix/separation/`

Then run NASS method.

## Cite Our Paper #####

Please cite our [paper](https://arxiv.org/abs/2305.10761) and star our repository.

```
@misc{zhang2023noiseaware,
      title={Noise-Aware Speech Separation with Contrastive Learning}, 
      author={Zizheng Zhang and Chen Chen and Hsin-Hung Chen and Xiang Liu and Yuchen Hu and Eng Siong Chng},
      year={2023},
      eprint={2305.10761},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
