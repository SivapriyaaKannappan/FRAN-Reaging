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
    test_dataset=AgeDataset(test_input) 
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
        for batch_idx1, (img1,img_id1,age1) in enumerate(test_dataloader):
            img1=img1.permute(0,3,1,2) # Reorder dimensions to B, C, H, W 
            img1=img1.to(device)
            disp_img1=img1/255.0  #Normalize the image range to 0 and 1
            age_matrix1=[]
            for idx1 in range(len(age1)):
                age_matrix1.append(torch.Tensor(img1.shape[2],img1.shape[3]).fill_(age1[idx1]))
            final_age_matrix1=torch.stack((age_matrix1),0)
            final_age_matrix11=torch.unsqueeze(final_age_matrix1, axis=1)
           
            final_age_matrix21=final_age_matrix11.to(device)
            input_img1=torch.cat((img1[0::2]/255.0,final_age_matrix21[0::2]/100.0,final_age_matrix21[1::2]/100.0), 1) # i/p img+i/p age+target age
            target_img1=img1[1::2]/255.0
            input_age1=age1[0::2]
            target_age1=age1[1::2]
           
            input_img1=input_img1.to(device)
            # Forward propagation
            pred_img1 = model(input_img1) # returns RGB aging delta
            final_pred_img1=torch.add(pred_img1,disp_img1[0::2]) # Add the aging delta to the normalized input image
            
            # # # # #**************************************************************
            # # # # Visualizing the image output
            # if (img_id1 == '1801' or img_id1 == '1802' or img_id1 == '1803' or img_id1 == '1804' or img_id1 == '1805'):
            #     for i in range(pred_img1.size(0)):
            #         slice_tensor=pred_img1[i, :, :, :]
            #         slice_tensor=torch.clamp(slice_tensor, 0,1)
            #         save_image(slice_tensor,os.path.join("./results/vis_offset/",f'{img_id1[0]}_{count}_{target_age1[i]}_modelout.png'))
                
            #         count+=1
            #         # save_image(slice_tensor,os.path.join("./data/valid_vis/",f'{img_id1[0]}_{target_age1[i]}_modelout.png'))
            # # # #**************************************************************
            # Prepare the real image and the predicted image variations (4 channel image) to the discriminator
            concat_pred_img1=torch.cat((final_pred_img1,final_age_matrix21[1::2]/100.0), 1) # Predicted img+target age (4 channel)
            concat_real_img11=torch.cat((target_img1,final_age_matrix21[1::2]/100.0), 1) # Synthetic Target image +target age (4 channel)
            # Find the incorrect target age matrix
            incorrect_age1=[]
            for k1, ta1 in enumerate(target_age1):
                incorrect_age1+=random.sample([aid for i,aid in enumerate(age1) if aid !=target_age1[k1]],1)
            incorrect_age_matrix1=[]
            for idx in range(len(incorrect_age1)):
                 incorrect_age_matrix1.append(torch.Tensor(img1.shape[2],img1.shape[3]).fill_(incorrect_age1[idx]/100.0))
            final_incorrect_age_matrix1=torch.stack((incorrect_age_matrix1),0)
            final_incorrect_age_matrix11=torch.unsqueeze(final_incorrect_age_matrix1, axis=1)    
            final_incorrect_age_matrix21=final_incorrect_age_matrix11.to(device)
            
            concat_real_img21=torch.cat((target_img1,final_incorrect_age_matrix21), 1) # Synthetic Target image +incorrect target age (4 channel)
                      
            # Compute Generator loss
            disc_pred_img1 = discriminator(concat_pred_img1) # returns 30X30 logits map
            gloss1=computeGAN_GLoss(final_pred_img1,disc_pred_img1,target_img1,l1_weight, lpips_weight, adv_weight)
            test_gloss+=gloss1
           
            # Compute Discriminator loss
            disc_real_img11 = discriminator(concat_real_img11) # returns 30X30 logits map
            disc_real_img21 = discriminator(concat_real_img21) # returns 30X30 logits map
            dloss1=computeGAN_DLoss(disc_pred_img1,disc_real_img11,disc_real_img21, target_age1,test_dataset.age_ids)
            test_dloss+=dloss1
            
            # Save the output
            # for i in range(pred_img1.size(0)):
            #     slice_tensor=pred_img1[i, :, :, :]
            #     # save_image(slice_tensor,os.path.join(output,f'{img_id1[0]}_{i}_{target_age1[i]}_out.png'))
            #     save_image(slice_tensor,os.path.join(output,f'{img_id1[0]}_{target_age1[i]}_out.png'))
            slice_tensor=torch.clamp(final_pred_img1, 0,1)
            concat_out_img=torch.cat((disp_img1[0::2], disp_img1[1::2],slice_tensor),0) # input, target, predicted
            save_image(concat_out_img,os.path.join("./results/",f'{img_id1[0]}_iage_{input_age1.item()}_tage_{target_age1.item()}_modelout.png'))
            
    total_test_loss=test_gloss+test_dloss
    average_test_loss =total_test_loss/len(test_dataloader)
    print("********************")
    print(f' Test Loss: {average_test_loss.item():.4f}\n')
    print("********************")
    
def get_args():
    parser=argparse.ArgumentParser(description='Test FRAN via UNet with input aged/de-aged images and output aged/de-aged images')
    parser.add_argument('--model', '-m',metavar="FILE", default="checkpoints/UNet_Fri_08Dec2023_225708_epoch30.pth",help='Specify the file in which the model is stored')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', default="./resized_dataset/test/")
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
   