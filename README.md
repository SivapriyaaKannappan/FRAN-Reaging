# FRAN: Face Re-Aging Network (Using U-Net in PyTorch)


- [Description](#description)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Weights & Biases](#weights--biases)
- [Pretrained model](#pretrained-model)
- [Data](#data)

## Quick start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the data 
Input Samples are generated from StyleGAN2, which is fed into SAM Re-aging (https://github.com/yuval-alaluf/SAM), which inturn produces the output at different age categories ranging from 10 to 85.
So, totally we have 2000 identities of 1024 X 1024 resolution for each category among 16 different age categories range from 10 to 85.

5. Run training:
```console
> python train.py 
usage: train.py [--epochs E] [--batch-size B] [ --learning-rate LR]

Train FRAN on input aged/de-aged images and output aged/de-aged images

optional arguments:
--epochs E, -e E 		Number of epochs
--batch-size B, -b B		Batch Size
--learning-rate LR -l LR	Learning Rate

```

## Description


## Usage
**Note : Use Python 3.6 or newer**

### Training




### Prediction



## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.

## Pretrained model

## Data




## Cite
```console
Production-Ready Face Re-Aging for Visual Effects
https://studios.disneyresearch.com/app/uploads/2022/10/Production-Ready-Face-Re-Aging-for-Visual-Effects.pdf<br>
Github code is not released for this paper.<br>
https://studios.disneyresearch.com/app/uploads/2022/10/Production-Ready-Face-Re-Aging-for-Visual-Effects_Supplemental.pdf

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
```
