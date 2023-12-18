# FRAN: Face Re-Aging Network (Using U-Net in PyTorch)

- [Quick Start](#quick-start)
- [Description](#description)
- [Usage](#usage)
  - [Dataset & Training](#dataset--training)
  - [Hyperparameter Values](#hyperparameter-values)
  - [Predictions](#predictions)
  - [Weights & Biases](#weights--biases)
  - [Pretrained model](#pretrained-model)
- [Cite](#cite)

## Quick Start
1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)
2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Download the data 
Input Samples are generated from StyleGAN2, which is fed into SAM Re-aging (https://github.com/yuval-alaluf/SAM), which inturn produces the output at different age categories ranging from 10 to 85.
So, totally we have 2000 identities of 1024 X 1024 resolution for each age category (among 16 different age categories) ranging from 10 to 85.

5. Run training:
```console
> python train.py
```
Train FRAN on input aged/de-aged images and target aged/de-aged images by finetuning the paramaeters such as Generator's & Discriminator's learning rate, step-size & gamma value in the scheduler, no of identities in the batch, no of sample per identity, batch size.

5. Run testing:
   ```console
   > python test.py
   ```

## Description
[FRAN](https://studios.disneyresearch.com/app/uploads/2022/10/Production-Ready-Face-Re-Aging-for-Visual-Effects.pdf) (Face Re-Aging Network) is the fully automatic and production ready method for re-aging face images. 

## Usage
### Dataset & Training
The training dataset is generated from StyleGAN2, which is fed into [SAM] (https://github.com/yuval-alaluf/SAM), which inturn produces the output at different age categories (among 16 different age categories) ranging from 10 to 85. So, totally we generated 2000 identities of 1024 X 1024 resolution for each age category ranging from 10 to 85. Here U-Net is used for this image-to-image translation task.<br>
The RGB image along with the input and target age is fed as a 5 channel input to the Generator model (U-Net). When the generator is trained, the discriminator is fixed and when the discriminator is trained, the generator is fixed. Combined losses such as L1 loss, LPIPS loss and Adversarial loss are used. The adversarial loss for the generator is calcualted using the BCEwithLogitsLoss of the predicted image and the GT label as True, so that the generator's loss is minimized and the generator could generate better images.<br>
The PatchGAN discriminator is fed with 4 channel images such as predicted image+ correct target age, target synthetic image+ correct target age, target synthetic image+ incorrect target age, from which adversarial loss is calculated with their corresponding GT labels as False, True and False in order to minimize the discriminator's loss.
For a given image along with the input age and target age, the program will output a re-aged face image based on the target age.
### Hyperparameter Values 
* Image Size - 256 X 256
* Training Epochs - 100
* Generator's Learning Rate - 1e-04
* Discriminator Learning Rate - 1e-05
* Scheduler
  * Step-size - 10
  * Gamma - 0.9
* Mini-batch size - 8
  * No of identities per mini-batch - 4
  * No of samples per identity - 2
* In the Sampler, for every epoch, the image identities are permuted to shuffle their order
### Weights & Biases
The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves are logged to the platform.
When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.

### Predictions
Output in the form of Input/Target/Predictions <br />
70->25 ![1823_iage_70_tage_25_modelout](https://github.com/SivapriyaaKannappan/FRAN-Reaging/assets/14826726/1c8895cb-0920-45fb-8f98-9f0d5d47a4f3) <br />

55->80 ![1832_iage_55_tage_80_modelout](https://github.com/SivapriyaaKannappan/FRAN-Reaging/assets/14826726/30c27636-d656-4643-8737-2b1315c042e4) <br />

85->40 ![1839_iage_85_tage_40_modelout](https://github.com/SivapriyaaKannappan/FRAN-Reaging/assets/14826726/9069869a-8385-4d12-b0df-1331203bca4a) <br />

60->20 ![1911_iage_60_tage_20_modelout](https://github.com/SivapriyaaKannappan/FRAN-Reaging/assets/14826726/d2b85df8-4cbc-4a44-9e5d-ed21f71a3c87) <br />

75->35 ![1813_iage_75_tage_35_modelout](https://github.com/SivapriyaaKannappan/FRAN-Reaging/assets/14826726/d2f7651f-3647-4e57-8bb2-8870d418b5bf) <br />

70->30 ![1955_iage_70_tage_30_modelout](https://github.com/SivapriyaaKannappan/FRAN-Reaging/assets/14826726/f0b71685-cefd-4287-b7d7-d4b67311d262) <br />

### Pretrained model


## Cite
If you use this code, please consider citing the [FRAN paper](https://studios.disneyresearch.com/app/uploads/2022/10/Production-Ready-Face-Re-Aging-for-Visual-Effects.pdf) on which this work is based.
```console
@article{zoss2022production,
  title={Production-Ready Face Re-Aging for Visual Effects},
  author={Zoss, Gaspard and Chandran, Prashanth and Sifakis, Eftychios and Gross, Markus and Gotardo, Paulo and Bradley, Derek},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={6},
  pages={1--12},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```
