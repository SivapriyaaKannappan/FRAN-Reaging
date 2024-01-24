# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:14:18 2023

@author: Sivapriyaa

Production-Ready Face Re-Aging for Visual Effects (FRAN - Face-Re-Aging Network) 
https://studios.disneyresearch.com/app/uploads/2022/10/Production-Ready-Face-Re-Aging-for-Visual-Effects.pdf
Github code is not released for this paper

Dataset - Generated 2000 identities from StyleGAN which is then fed into SAM to generate reaged images at different ages (Style-based Age Manipulation)
Generator - U-Net model
Losses - L1 Loss, Learned Perceptual Image Patch Similarity (LPIPS) and Adversarial Loss
        LPIPS- takes the predicted image and the target image as input and computes perceptual dissimilarity
Discriminator - PatchGAN
"""

#Divided the RGB image by 255 and the input & target age by 100 in order to normalize
#the image range to 0 and 1 before feeding to the model
# Similarly after the training the predicted image will be between 0 and 1 

#Since the training images are from SAM, the test images are aligned similar to SAM using dlib.

import argparse
import os
import sys
import torch
import torch.nn as nn
from utils.data_loading import AgeDataset, AgeBatchSampler
from unet import UNet
from tqdm import tqdm
import numpy as np
from utils.loss import computeGAN_GLoss, computeGAN_DLoss
from time import gmtime, strftime
from torch.utils.data import DataLoader
from patch_gan_discriminator import PatchGANDiscriminator
from PIL import Image
import random
from torchvision.utils import save_image
   
def test_model(model,discriminator,test_input,output,batch_size,l1_weight: float = 0.3, lpips_weight: float =0.3, adv_weight: float = 0.4):   
    # 1. Create dataset
    test_dataset=AgeDataset(test_input, 512, ['colorjitter', 'rotation', 'resize']) # Load test images as 512x512
    all_ages = [int(age) for age in test_dataset.age_ids]
    # 2. Create samplers
    test_sampler=AgeBatchSampler(test_dataset.n_age_ids,n_img_ids=test_dataset.n_img_ids, n_classes_per_batch=1, n_samples_per_class=batch_size)
    # 3. Create data loader
    test_dataloader=DataLoader(dataset=test_dataset,batch_sampler=test_sampler,shuffle=False)
    model.eval()
    discriminator.eval()
    test_gloss=0.0
    test_dloss=0.0
    # count=0
    with torch.no_grad():
        for batch_idx1, ((imgA, imgA_id, ageA, imgB, imgB_id, ageB)) in enumerate(test_dataloader):
            # img1=img1.permute(0,3,1,2) # Reorder dimensions to B, C, H, W 
            # img1=img1.to(device)
            # disp_img1=img1/255.0  #Normalize the image range to 0 and 1
            imgA, imgB = imgA.to(device), imgB.to(device)
            ageA_matrix=[]
            for idx in range(len(ageA)):
                ageA_matrix.append(torch.Tensor(imgA.shape[2],imgA.shape[3]).fill_(ageA[idx]/100.0))
            ageA_matrix=torch.stack((ageA_matrix),0)
            ageA_matrix=torch.unsqueeze(ageA_matrix, dim=1)
            ageB_matrix=[]
            for idx in range(len(ageB)):
                ageB_matrix.append(torch.Tensor(imgB.shape[2],imgB.shape[3]).fill_(ageB[idx]/100.0))
            ageB_matrix=torch.stack((ageB_matrix),0)
            ageB_matrix=torch.unsqueeze(ageB_matrix, dim=1)            
            
            ageA_matrix, ageB_matrix = ageA_matrix.to(device), ageB_matrix.to(device)
            input_img = torch.cat((imgA, ageA_matrix, ageB_matrix), 1) # i/p img+i/p age+target age
            
            # Forward propagation
            pred_img = model(input_img) # returns RGB aging delta
            final_pred_img = torch.add(pred_img, imgA) # Add the aging delta to the normalized input image
            
            # Prepare the real image and the predicted image variations (4 channel image) to the discriminator
            concat_pred_img = torch.cat((final_pred_img, ageB_matrix), 1) # Predicted img+target age (4 channel)
            concat_real_img1=torch.cat((imgB, ageB_matrix), 1) # Synthetic Target image +target age (4 channel)
            
            # Find the incorrect target age matrix
            incorrect_age=[]
            for k, ta in enumerate(ageB):
                incorrect_age += random.sample([aid for i, aid in enumerate(all_ages) if aid != ageB[k]], 1)
            incorrect_age_matrix = []
            for idx in range(len(incorrect_age)):
                  incorrect_age_matrix.append(torch.Tensor(imgB.shape[2],imgB.shape[3]).fill_(incorrect_age[idx]/100.0))
            incorrect_age_matrix = torch.stack((incorrect_age_matrix), 0)
            incorrect_age_matrix = torch.unsqueeze(incorrect_age_matrix, dim=1)    
            incorrect_age_matrix = incorrect_age_matrix.to(device)
            
            concat_real_img2=torch.cat((imgB, incorrect_age_matrix), 1) # Synthetic Target image +incorrect target age (4 channel)
                      
            # Compute Generator loss
            disc_pred_img = discriminator(concat_pred_img) # returns 30X30 logits map
            gloss=computeGAN_GLoss(final_pred_img, disc_pred_img, imgB, l1_weight, lpips_weight, adv_weight)
            test_gloss+=gloss
           
            # Compute Discriminator loss
            disc_real_img1 = discriminator(concat_real_img1) # returns 30X30 logits map
            disc_real_img2 = discriminator(concat_real_img2) # returns 30X30 logits map
            dloss=computeGAN_DLoss(disc_pred_img, disc_real_img1, disc_real_img2, ageB, test_dataset.age_ids)
            test_dloss+=dloss
            
            # Save the output
            # for i in range(pred_img1.size(0)):
            #     slice_tensor=pred_img1[i, :, :, :]
            #     # save_image(slice_tensor,os.path.join(output,f'{img_id1[0]}_{i}_{target_age1[i]}_out.png'))
            #     save_image(slice_tensor,os.path.join(output,f'{img_id1[0]}_{target_age1[i]}_out.png'))
            # final_pred_img = (final_pred_img + 1) / 2.0
            slice_tensor=torch.clamp(final_pred_img, 0, 1)
            # concat_out_img=torch.cat(((imgA + 1)/2.0, (imgB + 1)/2.0, slice_tensor),0) # input, target, predicted
            concat_out_img=torch.cat((imgA, imgB, slice_tensor), 0) # input, target, predicted
            save_image(concat_out_img,os.path.join("./results/",f'{imgA_id[0]}_iage_{ageA[0].item()}_tage_{ageB[0].item()}_modelout.png'))
            
    total_test_loss=test_gloss+test_dloss
    average_test_loss =total_test_loss/len(test_dataloader)
    print("********************")
    print(f' Test Loss: {average_test_loss.item():.4f}\n')
    print("********************")
    
def get_args():
    parser=argparse.ArgumentParser(description='Test FRAN via UNet with input aged/de-aged images and output aged/de-aged images')
    parser.add_argument('--model', '-m',metavar="FILE", default="checkpoints/UNet_Fri_05Jan2024_135259_epoch50.pth",help='Specify the file in which the model is stored')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', default="./dataset/test/")
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images', default="results/")
    parser.add_argument('--batch_size', '-b', metavar = 'B', type=int, default=2, help='Size of the mini-batch')
    parser.add_argument('--lossl1weight', '-l1weight', metavar='L1W',dest='l1weight', type=float, default=1.0, help='L1 Loss Weight')    
    parser.add_argument('--losslpipsweight', '-lpipsweight', metavar='LPIPSW',dest='lpipsweight', type=float, default=1.0, help='LPIPS Loss Weight')    
    parser.add_argument('--lossadvweight', '-advweight', metavar='ADVW',dest='advweight', type=float, default=0.05, help='Adversarial Loss Weight')    
    return parser.parse_args()

if __name__ == '__main__':
    args=get_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # n_channels=5 for RGB+ InputAge + Target Age
    # n_classes is the number of probabilities you want to get per output pixel
    model = UNet(n_channels=5, n_classes=3, bilinear=args.bilinear) # To get the predicted RGB age delta
    model.to(device=device)
    
    # Initialize the PatchGAN discriminator
    discriminator = PatchGANDiscriminator(in_channels=4)
    discriminator.to(device=device)
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_model(
        model=model,
        discriminator=discriminator,
        test_input=args.input,
        output=args.output,
        batch_size=args.batch_size,
        l1_weight=args.l1weight,
        lpips_weight=args.lpipsweight,
        adv_weight =args.advweight
        )
   