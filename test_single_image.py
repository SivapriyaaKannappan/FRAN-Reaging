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
from torch.utils.data import SequentialSampler
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
import torchvision.transforms as transforms
def test_model(model,discriminator,inputimg_path,targetimg_path,input_age,target_age,output,batch_size,l1_weight: float = 1.0, lpips_weight: float =1.0, adv_weight: float = 0.05):   
            # image to a Torch tensor 
            transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                            transforms.PILToTensor()]) 

            img_id=inputimg_path.split(".",2)[1].split("/",4)[-1]
            img=Image.open(inputimg_path)
            img=transform(img) # image to tensor
            if (img.shape[0] > 3):
                img=img[:3,:,:]
            img=img.unsqueeze(0)
            img=img.to(device)
            
            if (targetimg_path != None):
                timg=Image.open(targetimg_path)
                timg=transform(timg) # image to tensor
                timg=timg.unsqueeze(0)
                timg=timg.to(device)
                
                tdisp_img1=timg/255.0  #Normalize the image range to 0 and 1
            
            disp_img1=img/255.0  #Normalize the image range to 0 and 1
            
            iage_matrix=torch.Tensor(img.shape[2],img.shape[3]).fill_(input_age)
            iage_matrix=iage_matrix.unsqueeze(0)
            iage_matrix=iage_matrix.unsqueeze(1).to(device)
            
            tage_matrix=torch.Tensor(img.shape[2],img.shape[3]).fill_(target_age)
            tage_matrix=tage_matrix.unsqueeze(0)
            tage_matrix=tage_matrix.unsqueeze(1).to(device)
          
            input_img=torch.cat((img/255.0,iage_matrix/100.0,tage_matrix/100.0), 1) # i/p img+i/p age+target age
            input_img=input_img.to(device)
            # Forward propagation
            pred_img1 = model(input_img) # returns RGB aging delta
            final_pred_img1=torch.add(pred_img1,disp_img1) # Add the aging delta to the normalized input image
                        
            slice_tensor=torch.clamp(final_pred_img1, 0,1)
            if (targetimg_path != None):
                concat_out_img=torch.cat((disp_img1, tdisp_img1,slice_tensor),0) # input, target, predicted
                save_image(concat_out_img,os.path.join("./results/",f'{img_id}_iage_{input_age}_tage_{target_age}_modelout.png'))
            else:
                concat_out_img=torch.cat((disp_img1,slice_tensor),0) # input, target, predicted
                save_image(concat_out_img,os.path.join("./results/",f'{img_id}_iage_{input_age}_tage_{target_age}_modelout.png'))
    
def get_args():
    parser=argparse.ArgumentParser(description='Test FRAN via UNet with input aged/de-aged images and output aged/de-aged images')
    parser.add_argument('--model', '-m',metavar="FILE", default="checkpoints/UNet_Sat_16Dec2023_151230_epoch100.pth",help='Specify the file in which the model is stored')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parser.add_argument('--inputimage', '-i', metavar='INPUT', dest="input", help='Filename of input image', default="./resized_dataset/test/25/1845.jpg")
    # parser.add_argument('--inputage', '-ia', metavar='INPUTAGE', dest="inputage", help='Input age', default=25)
    # parser.add_argument('--targetage', '-ta', metavar='TARGETAGE',dest="targetage", help='Target age', default=75)
    # parser.add_argument('--targetimage', '-t', metavar='TARGET', dest="target", help='Filename of target image', default="./resized_dataset/test/75/1845.jpg")
    parser.add_argument('--inputimage', '-i', metavar='INPUT', dest="input", help='Filename of input image', default="4_res.jpg")
    parser.add_argument('--inputage', '-ia', metavar='INPUTAGE', dest="inputage", help='Input age', default=25)
    parser.add_argument('--targetage', '-ta', metavar='TARGETAGE',dest="targetage", help='Target age', default=85)
    parser.add_argument('--targetimage', '-t', metavar='TARGET', dest="target", help='Filename of target image', default=None)
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
        inputimg_path=args.input,
        targetimg_path=args.target,
        input_age=args.inputage,
        target_age=args.targetage,
        output=args.output,
        batch_size=args.batch_size,
        l1_weight=args.l1weight,
        lpips_weight=args.lpipsweight,
        adv_weight =args.advweight
        )
   