# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:35:18 2023

@author: Sivapriyaa
"""

import torch
import torch.nn as nn
import lpips # Learned Perceptual Image Patch Similarity Loss

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion1 = nn.L1Loss()
criterion2 = lpips.LPIPS(net="vgg").to(device)
criterion3 = nn.BCEWithLogitsLoss()

def computeGAN_GLoss(final_pred_img, disc_pred_img,target_img,l1_weight, lpips_weight, adv_weight):
    """Calculate loss for Generator G"""
    #1. L1 loss
    loss_l1 = criterion1(final_pred_img, target_img)
    # print("L1 Loss:", loss_l1.item())
    #2. LPIPS Loss
    loss_lpips = criterion2(final_pred_img, target_img).mean()
    # print("LPIPS Loss:", loss_lpips.item())
    #3. Adversarial loss
    labels_true = torch.ones_like(disc_pred_img)
    loss_g = criterion3(disc_pred_img,labels_true).mean()
    
    gen_loss = l1_weight*loss_l1 + lpips_weight*loss_lpips +adv_weight*loss_g 
    return gen_loss 
    
def computeGAN_DLoss(disc_pred_img,disc_real_img1,disc_real_img2,target_age, age_ids):
    """Calculate loss for discriminator D"""

    labels_false = torch.zeros_like(disc_pred_img)
    labels_true = torch.ones_like(disc_real_img1)
    
    loss_d1 = criterion3(disc_pred_img,labels_false).mean()
    loss_d2 = criterion3(disc_real_img1,labels_true).mean()
    loss_d3 = criterion3(disc_real_img2,labels_false).mean()
   
    disc_loss = (loss_d1 + loss_d2+loss_d3).mean()
    return disc_loss
    
      
    