# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:14:18 2023

@author: Sivapriyaa

Production-Ready Face Re-Aging for Visual Effects (FRAN - Face-Re-Aging Network) 
https://studios.disneyresearch.com/app/uploads/2022/10/Production-Ready-Face-Re-Aging-for-Visual-Effects.pdf

"""
#Divided the RGB image by 255 and the input & target age by 100 in order to normalize
#the image range to 0 and 1 before feeding to the model
# Similarly after the training the predicted image will be between 0 and 1 

# Incorporated face segmentation to re-age specific parts of the image by using facexlib Parsenet and it's alignment.
# Aligned the face image similar to parsenet and normalized the aligned image between -1 to 1 and Parsenet expects BGR image.
#(Referred Parsenet from GFPGAN)

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import SequentialSampler
from unet import UNet
from tqdm import tqdm
import numpy as np
from utils.loss import computeGAN_GLoss, computeGAN_DLoss
from time import gmtime, strftime
from torch.utils.data import DataLoader
from PIL import Image
import random
from torchvision.utils import save_image
import torchvision.transforms as transforms
import dlib

from facexlib.utils.face_restoration_helper import FaceRestoreHelper
# from utils.align_all_parallel import align_face
from torchvision.transforms.functional import normalize
from facexlib.utils.misc import img2tensor, imwrite
import cv2

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
face_helper = FaceRestoreHelper(
    1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model='retinaface_resnet50',
    save_ext='png',
    use_parse=True,
    device=device,
    model_rootpath='weights')
        
def run_alignment(image_path):
    # predictor = dlib.shape_predictor("D:/libraries/dlib-master/shape_predictor_68_face_landmarks.dat")
    # aligned_image = align_face(filepath=image_path, predictor=predictor)
    # print("Aligned image has shape: {}".format(aligned_image.size))
    face_helper.read_image(image_path)
        
    # get face landmarks for each face
    face_helper.get_face_landmarks_5(only_center_face=True, eye_dist_threshold=5)
    # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
    # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
    # align and warp each face
    face_helper.align_warp_face()

    # face restoration
    aligned_img = face_helper.cropped_faces[0]    
    aligned_img = aligned_img[:,:,::-1]
    aligned_img = Image.fromarray(aligned_img)
    return aligned_img

def seg_mask():
    from facexlib.parsing.parsenet import ParseNet  #facexlib.parsing.parsenet 
    from facexlib.utils import load_file_from_url
    model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
    model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'
    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=None)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model

def test_model(model,inputimg_path,targetimg_path,input_age,target_age,output,batch_size,l1_weight: float = 1.0, lpips_weight: float =1.0, adv_weight: float = 0.05,use_parse=False):
    # image to a Torch tensor 
    transform = transforms.Compose([transforms.Resize(size=(512, 512)),
                                    transforms.ToTensor()])
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

    img_id=inputimg_path.split(".",2)[0].split("/",4)[-1]
    aligned_img=run_alignment(inputimg_path) # face image aligned similar to SAM (via dlib)
    
    # img=Image.open(inputimg_path)
    img=transform(aligned_img) # image to tensor
    if (img.shape[0] > 3):
        img=img[:3,:,:]
    img=img.unsqueeze(0)
    img=img.to(device)
    
    if (targetimg_path != None):
        aligned_timg=run_alignment(targetimg_path)
        # timg=Image.open(targetimg_path)
        timg=transform(aligned_timg) # image to tensor
        timg=timg.unsqueeze(0)
        timg=timg.to(device)
        
        # tdisp_img1=timg/255.0  #Normalize the image range to 0 and 1
        # tdisp_img1 = (timg + 1) * 0.5
        tdisp_img1 = timg
    
    # disp_img1=img/255.0  #Normalize the image range to 0 and 1
    # disp_img1  = (img + 1) * 0.5
    disp_img1 = img
    
    iage_matrix=torch.Tensor(img.shape[2],img.shape[3]).fill_(input_age/100.0)
    iage_matrix=iage_matrix.unsqueeze(0)
    iage_matrix=iage_matrix.unsqueeze(1).to(device)
    
    tage_matrix=torch.Tensor(img.shape[2],img.shape[3]).fill_(target_age/100.0)
    tage_matrix=tage_matrix.unsqueeze(0)
    tage_matrix=tage_matrix.unsqueeze(1).to(device)
    
    #******************************************************************
    if use_parse:
        # # Segmentation mask of an re-aged image to focus on the specific facial features
        aligned_img=np.array(aligned_img)
        aligned_img = cv2.resize(aligned_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        aligned_img = img2tensor(aligned_img.astype('float32') / 255., bgr2rgb=False, float32=True) # Parsenet expects BGR image
        normalize(aligned_img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True) # Normalize between -1 and 1
        norm_img = torch.unsqueeze(aligned_img, 0).to(device)
        
        with torch.no_grad():
            mask=seg_mask()
            out=mask(norm_img)[0]
        out = out.argmax(dim=1).squeeze().cpu().numpy() 
        
        mask = np.zeros(out.shape)
        # The masks are ['background', 'skin','nose','eye_g','r-eye','l_eye','r_brow', 'l_brow', 'r_ear', 'l_ear', 'teeth',
        		# 'u_lip', 'l_lip', 'hair', 'hat',? , ? ,'neck', 'cloth']
        # MASK_COLORMAP = [0, 255, 255, 255, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0]
        # MASK_COLORMAP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0] # eyes and eyebrows
        MASK_COLORMAP = [0, 0, 0,0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0,255, 0] # eyes and eyebrows
        
        for idx, color in enumerate(MASK_COLORMAP):
            mask[out == idx] = color
       
        # #*****************************
        # save_image(img,"img.png")
        
        # # Convert mask into uint8 to SAVE
        # mask=mask.astype(np.uint8)
        # cv2.imwrite("mask.png",mask)
        #*****************************
       
        norm_mask=mask/255.
        
        iage_matrix1=iage_matrix.cpu().detach().numpy()
        masked_iage_matrix=norm_mask*iage_matrix1
        final_masked_iage_matrix=torch.Tensor(masked_iage_matrix) # Convert numpy to torch tensor
        final_masked_iage_matrix=final_masked_iage_matrix.to(device)
        
        tage_matrix1=tage_matrix.cpu().detach().numpy()
        masked_tage_matrix=norm_mask*tage_matrix1
        final_masked_tage_matrix=torch.Tensor(masked_tage_matrix) # Convert numpy to torch tensor
        final_masked_tage_matrix=final_masked_tage_matrix.to(device)
        # #******************************************************************
        
        input_img=torch.cat((img, final_masked_iage_matrix, final_masked_tage_matrix), 1) # i/p img+i/p age+target age
        input_img=input_img.to(device)
    else:
        input_img=torch.cat((img, iage_matrix, tage_matrix), 1) # i/p img+i/p age+target age
        input_img=input_img.to(device)
        
    # input_img=torch.cat((img, iage_matrix, tage_matrix), 1) # i/p img+i/p age+target age
    # input_img=input_img.to(device)
    
    # Forward propagation
    pred_img1 = model(input_img) # returns RGB aging delta
    final_pred_img1 = torch.add(pred_img1, img) # Add the aging delta to the normalized input image
    # final_pred_img1 = (final_pred_img1 + 1) / 2.0
    slice_tensor=torch.clamp(final_pred_img1, 0, 1)
    
    if (targetimg_path != None):
        concat_out_img=torch.cat((disp_img1, tdisp_img1, slice_tensor),0) # input, target, predicted
        save_image(concat_out_img,os.path.join("./results/",f'{img_id}_iage_{input_age}_tage_{target_age}_modelout_512.png'))
    else:
        concat_out_img=torch.cat((disp_img1,slice_tensor),0) # input, target, predicted
        save_image(concat_out_img,os.path.join("./results/",f'{img_id}_iage_{input_age}_tage_{target_age}_modelout_512.png'))
    
def get_args():
    parser=argparse.ArgumentParser(description='Test FRAN via UNet with input aged/de-aged images and output aged/de-aged images')
    parser.add_argument('--model', '-m',metavar="FILE", default="checkpoints/FRAN_UNet_Tue_23Jan2024_101446_512x512_epoch90.pth",help='Specify the file in which the model is stored')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parser.add_argument('--inputimage', '-i', metavar='INPUT', dest="input", help='Filename of input image', default="./resized_dataset/test/25/1845.jpg")
    # parser.add_argument('--inputage', '-ia', metavar='INPUTAGE', dest="inputage", help='Input age', default=25)
    # parser.add_argument('--targetage', '-ta', metavar='TARGETAGE',dest="targetage", help='Target age', default=75)
    # parser.add_argument('--targetimage', '-t', metavar='TARGET', dest="target", help='Filename of target image', default="./resized_dataset/test/75/1845.jpg")

    parser.add_argument('--inputimage', '-i', metavar='INPUT', dest="input", help='Filename of input image', default="input/She1.jpg")
    parser.add_argument('--inputage', '-ia', metavar='INPUTAGE', dest="inputage", help='Input age', default=20)
    parser.add_argument('--targetage', '-ta', metavar='TARGETAGE',dest="targetage", help='Target age', default=60)

    parser.add_argument('--targetimage', '-t', metavar='TARGET', dest="target", help='Filename of target image', default=None)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images', default="results/")
    parser.add_argument('--batch_size', '-b', metavar = 'B', type=int, default=2, help='Size of the mini-batch')
    parser.add_argument('--lossl1weight', '-l1weight', metavar='L1W',dest='l1weight', type=float, default=1.0, help='L1 Loss Weight')    
    parser.add_argument('--losslpipsweight', '-lpipsweight', metavar='LPIPSW',dest='lpipsweight', type=float, default=1.0, help='LPIPS Loss Weight')    
    parser.add_argument('--lossadvweight', '-advweight', metavar='ADVW',dest='advweight', type=float, default=0.05, help='Adversarial Loss Weight')    
    parser.add_argument('--useparse', '-use_parse',dest='use_parse', type=bool, default=False, help='Whether to Segment specific facial parts for re-aging')    
    return parser.parse_args()

if __name__ == '__main__':
    args=get_args()
    # n_channels=5 for RGB+ InputAge + Target Age
    # n_classes is the number of probabilities you want to get per output pixel
    model = UNet(n_channels=5, n_classes=3, bilinear=args.bilinear) # To get the predicted RGB age delta
    model.to(device=device)
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    test_model(
        model=model,
        inputimg_path=args.input,
        targetimg_path=args.target,
        input_age=args.inputage,
        target_age=args.targetage,
        output=args.output,
        batch_size=args.batch_size,
        l1_weight=args.l1weight,
        lpips_weight=args.lpipsweight,
        adv_weight =args.advweight,
        use_parse=args.use_parse
        )
   