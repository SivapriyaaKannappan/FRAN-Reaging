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

### Hyperparameter Values 
* Generator's Learning Rate - 1e-04
* Discriminator Learning Rate - 1e-05
* Scheduler
  * Step-size - 10
  * Gamma - 0.9
* Mini-batch size
  * No of identities per mini-batch - 4
  * No of samples per identity - 2
* In the Sampler, for every epoch, the image identities are permuted to shuffle their order
### Weights & Biases
The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves are logged to the platform.
When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.

### Predictions
For a given image along with the input age and target age, the program will output a re-aged face image based on the target age.

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
