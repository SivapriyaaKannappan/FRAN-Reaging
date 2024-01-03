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

import argparse
import logging
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'
import torch
import wandb
from pathlib import Path
from utils.data_loading import AgeDataset, AgeBatchSampler
import torch.optim as optim
from unet import UNet
import random
from time import gmtime, strftime
from torch.utils.data import DataLoader
from utils.loss import computeGAN_GLoss, computeGAN_DLoss
from patch_gan_discriminator import PatchGANDiscriminator
from torchvision.utils import save_image
from PIL import Image

train_images_dir=Path('./dataset/train/')
val_images_dir=Path('./dataset/val/')
checkpoint_dir=Path('./checkpoints/')

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    
def train_model(
        model,
        discriminator,
        checkpoint_file,
        device,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate_g: float = 1e-4,
        learning_rate_d: float = 1e-5,
        l1_weight: float = 1.0,
        lpips_weight: float =1.0,
        adv_weight: float = 0.05,
        resume_checkpoint=False,
        img_scale=256
        ):
    #**********************************************************
    #Toy dataset testing
    # dataset=AgeDataset(images_dir)
    # dataset[2]
    # sampler=AgeBatchSampler(n_age_ids=3,n_img_ids=10, n_classes_per_batch=1, n_samples_per_class=8)
    # for x in sampler:
    #     print(x)
    # dataloader=DataLoader(dataset=dataset,batch_sampler=sampler,shuffle=False)
    # for batch_idx, (img,age) in enumerate(dataloader):
    #     print(age)
    # print("Alert!")
    #**********************************************************
    
    # 1. Create dataset
      
    train_dataset=AgeDataset(train_images_dir, img_scale, ['colorjitter', 'rotation', 'resize']) # Resize images to 512x512, no random cropping
    val_dataset=AgeDataset(val_images_dir, img_scale) 
    all_ages = [int(age) for age in train_dataset.age_ids]

    # 2. Create samplers
    train_sampler=AgeBatchSampler(train_dataset.n_age_ids,n_img_ids=train_dataset.n_img_ids, n_classes_per_batch=batch_size//2, n_samples_per_class=2)
    val_sampler=AgeBatchSampler(val_dataset.n_age_ids,n_img_ids=val_dataset.n_img_ids, n_classes_per_batch=batch_size//2, n_samples_per_class=2)
    
    # 3. Create data loaders
    train_dataloader=DataLoader(dataset=train_dataset,batch_sampler=train_sampler,shuffle=False)
    val_dataloader=DataLoader(dataset=val_dataset,batch_sampler=val_sampler,shuffle=False)

    # 4. Set up the optimizer, the loss and the learning rate scheduler 
    optimizer_G = optim.Adam(model.parameters(), lr=learning_rate_g)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_d)
            
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.9)  
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.9)  
    
    # Resume Checkpoint
    if resume_checkpoint:
        checkpoint=torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        start_epoch=checkpoint['epoch']+1
    else:
        start_epoch=1
        
    # (Initialize logging)
    experiment = wandb.init(project='FRAN TOY', name='Face Re-Aging', 
                # track hyperparameters and run metadata
                config={
                "architecture": "U-Net",
                "dataset": "Output from SAM Re-aging"
                })
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate_g=learning_rate_g,learning_rate_d=learning_rate_d)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate G:   {learning_rate_g}
        Learning rate D:   {learning_rate_d}
        Device:          {device.type}
       
    ''')
    
    n_iter = 0
    count = 0
   
    # # 5. Begin training
    for epoch in range(start_epoch, epochs + 1):
        #Training Phase
        model.train()
        discriminator.train()
        train_gloss=0.0
        train_dloss=0.0
        
        val_gloss=0.0
        val_dloss=0.0
        
        for batch_idx, (imgA, imgA_id, ageA, imgB, imgB_id, ageB) in enumerate(train_dataloader):
            imgA, imgB = imgA.to(device), imgB.to(device)
            # save_image((imgA + 1.0)/2.0, "src.png")
            # save_image((imgB + 1.0)/2.0, "dst.png")
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
                      
            #Backward propagation and optimization
            #Train G fix D
            set_requires_grad(discriminator, False)  # Ds require no gradients when optimizing Gs
            # set_requires_grad(model, True) #Even if we don't give this line, it implicitly denotes that G is trained and gradients are back propagated, hence commented for efficiency
            optimizer_G.zero_grad()  # Set the gradients to zero
            # Compute Generator loss
            #Right==========================================================================
            disc_pred_img = discriminator(concat_pred_img) # returns 30X30 logits map
            
            gloss=computeGAN_GLoss(final_pred_img, disc_pred_img, imgB, l1_weight, lpips_weight, adv_weight)
            train_gloss+=gloss
            gloss.backward() 
            optimizer_G.step() # Step optimizer to update model parameters
            
            #Train D fix G
            set_requires_grad(discriminator, True) 
            # set_requires_grad(model, False)#Even if we don't give this line, it implicitly denotes that G is trained and gradients are back propagated, hence commented for efficiency 
            optimizer_D.zero_grad()  # Set the gradients to zero
            # Compute Discriminator loss
            disc_pred_img = discriminator(concat_pred_img.detach()) # Detatch-no gradient will be backpropagated along this variable to G
        
            disc_real_img1 = discriminator(concat_real_img1) # returns 30X30 logits map
            disc_real_img2 = discriminator(concat_real_img2) # returns 30X30 logits map
            
            dloss=computeGAN_DLoss(disc_pred_img, disc_real_img1, disc_real_img2, ageB, train_dataset.age_ids)
            train_dloss+=dloss
            dloss.backward() 
            optimizer_D.step() # Step optimizer to update model parameters
            
            # experiment.log({
            #     'epoch': epoch,
            #     'batch_train_gloss': gloss.item(),
            #     'batch_train_dloss': dloss.item()
            # })
            # # print("=============")
            print(f'Epoch [{epoch}/{epochs}],Batch[{batch_idx}/{len(train_dataloader)}], Total Generator Loss/Mini-Batch: {gloss.item():.4f}')
            print(f'Epoch [{epoch}/{epochs}], Batch[{batch_idx}/{len(train_dataloader)}], Total Discriminator Loss/Mini-Batch: {dloss.item():.4f}')
            # print("=============")
            # Empty CUDA cache
            torch.cuda.empty_cache()
            n_iter += 1
        total_train_loss=train_gloss+train_dloss
        average_train_loss=total_train_loss/len(train_dataloader)
        experiment.log({
            'epoch': epoch,
            'total_train_loss': total_train_loss.item(),
            'average_train_loss': average_train_loss.item(),
            
        })
        print("********************")
        print(f'Epoch [{epoch}/{epochs}], Total Batch Loss: {average_train_loss.item():.4f}')
        print("********************")
        scheduler_G.step() # Step scheduler (if needed)
        scheduler_D.step() # Step scheduler (if needed)
        
        ################### Validation Phase #################################
        model.eval()
        discriminator.eval()
             
        with torch.no_grad():
            for batch_idx1, (imgA, imgA_id, ageA, imgB, imgB_id, ageB) in enumerate(val_dataloader):
                imgA, imgB = imgA.to(device), imgB.to(device)
                # disp_img=img/255.0  #Normalize the image range to 0 and 1
                ageA_matrix=[]
                for idx in range(len(ageA)):
                    ageA_matrix.append(torch.Tensor(imgA.shape[2], imgA.shape[3]).fill_(ageA[idx]/100.0))
                ageA_matrix=torch.stack((ageA_matrix), 0)
                ageA_matrix=torch.unsqueeze(ageA_matrix, dim=1)
                ageB_matrix=[]
                for idx in range(len(ageB)):
                    ageB_matrix.append(torch.Tensor(imgB.shape[2], imgB.shape[3]).fill_(ageB[idx]/100.0))
                ageB_matrix=torch.stack((ageB_matrix), 0)
                ageB_matrix=torch.unsqueeze(ageB_matrix, dim=1)            
                
                ageA_matrix, ageB_matrix = ageA_matrix.to(device), ageB_matrix.to(device)
                input_img=torch.cat((imgA, ageA_matrix, ageB_matrix), 1) # i/p img+i/p age+target age
                
                # Forward propagation
                pred_img = model(input_img) # returns RGB aging delta
                final_pred_img = torch.add(pred_img, imgA) # Add the aging delta to the normalized input image
            
                # # # #**************************************************************
                # # # Visualizing the image output
                for i, img_id in enumerate(imgA_id):
                    if img_id in ['1411', '1412', '1413', '1414', '1415']:
                        offset_img = pred_img[i, ...]
                        minval = offset_img.min()
                        maxval = offset_img.max()
                        offset_img = (offset_img - minval) / (maxval - minval + 1e-10)
                        save_image(offset_img, os.path.join("./results/vis_offset/",f'{img_id}_{count}_{ageA[i]}_{ageB[i]}_offset.png'))                        
                    
                        slice_tensor = (final_pred_img[i, :, :, :] + 1.0) / 2.0
                        # slice_tensor=torch.clamp(slice_tensor, 0,1)
                        save_image(slice_tensor,os.path.join("./results/vis_offset/",f'{img_id}_{count}_{ageA[i]}_{ageB[i]}_modelout.png'))
                    
                        count+=1
                        # save_image(slice_tensor,os.path.join("./data/valid_vis/",f'{img_id1[0]}_{target_age1[i]}_modelout.png'))
                
                
                # # #**************************************************************
                
                # Prepare the real image and the predicted image variations (4 channel image) to the discriminator
                concat_pred_img = torch.cat((final_pred_img, ageB_matrix), 1) # Predicted img+target age (4 channel)
                concat_real_img1 = torch.cat((imgB, ageB_matrix), 1) # Synthetic Target image +target age (4 channel)
                # Find the incorrect target age matrix
                incorrect_age=[]
                for k1, ta1 in enumerate(ageB):
                    incorrect_age += random.sample([aid for i,aid in enumerate(all_ages) if aid !=ageB[k1]],1)
                incorrect_age_matrix=[]
                for idx in range(len(incorrect_age)):
                     incorrect_age_matrix.append(torch.Tensor(imgA.shape[2], imgA.shape[3]).fill_(incorrect_age[idx]/100.0))
                incorrect_age_matrix = torch.stack((incorrect_age_matrix),0)
                incorrect_age_matrix = torch.unsqueeze(incorrect_age_matrix, dim=1)    
                incorrect_age_matrix = incorrect_age_matrix.to(device)
                
                concat_real_img2 = torch.cat((imgB, incorrect_age_matrix), 1) # Synthetic Target image +incorrect target age (4 channel)
                          
                # Compute Generator loss
                disc_pred_img = discriminator(concat_pred_img) # returns 30X30 logits map
                gloss1 = computeGAN_GLoss(final_pred_img, disc_pred_img, imgB, l1_weight, lpips_weight, adv_weight)
                val_gloss += gloss1
               
                # Compute Discriminator loss
                disc_real_img1 = discriminator(concat_real_img1) # returns 30X30 logits map
                disc_real_img2 = discriminator(concat_real_img2) # returns 30X30 logits map
                          
                dloss1=computeGAN_DLoss(disc_pred_img, disc_real_img1, disc_real_img2, ageB, val_dataset.age_ids)
                val_dloss+=dloss1
            total_val_loss=val_gloss+val_dloss
            average_val_loss =total_val_loss/len(val_dataloader)
                    
        logging.info('Average Validation Loss: {}'.format(average_val_loss))
        try:
            experiment.log({
                'epoch': epoch,
                'learning rate G': learning_rate_g,
                'learning rate D': learning_rate_d,
                'total validation Loss': total_val_loss,
                'average validation Loss': average_val_loss,
            })
        except:
            pass
        print(f'Epoch[{epoch}/{epochs}]\n'
              f' Avg Training Loss: {average_train_loss.item():.4f}\n'
              f' Avg Validation Loss: {average_val_loss.item():.4f}\n')
              # f' Validation Accuracy: {average_val_accuracy:.4f}')
        print("********************")
        if (epoch%5 == 0):
            # Save the model every 5 epochs
            torch.save({'epoch'               : epoch,
                              'model_state_dict'    : model.state_dict(),
                              'optimizer_G_state_dict': optimizer_G.state_dict(),
                              'optimizer_D_state_dict': optimizer_D.state_dict(),
                              'scheduler_G_state_dict': scheduler_G.state_dict(),
                              'scheduler_D_state_dict': scheduler_D.state_dict(),
                              }, os.path.join(checkpoint_dir, \
                              '{}_{}_epoch{}.pth'.format(model.__class__.__name__, strftime("%a_%d%b%Y_%H%M%S",gmtime()), epoch)))   
   
    print("Training finished")
    wandb.finish()
   
def get_args():
    parser=argparse.ArgumentParser(description='Train FRAN via UNet with input aged/de-aged images and output aged/de-aged images')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--resume_checkpoint', '-resume', metavar='R', dest='resume', type=bool, default=True, help='Resume checkpoint or not')
    parser.add_argument('--checkpoint_file', '-chkpt', metavar='CP', dest='chkpt', type=str, default="checkpoints/UNet_Mon_01Jan2024_224515_epoch20.pth", help='Name of the checkpoint file')
    parser.add_argument('--start_epoch', '-se', metavar='SE', type=int, default=1, help='Starting epoch')
    parser.add_argument('--batch_size', '-b', metavar = 'B', type=int, default=8, help='Size of the mini-batch')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--learning_rate_g', '-lg', metavar='LRG',dest='lrg', type=float, default=1e-4, help='Learning rate for Generator')    
    parser.add_argument('--learning_rate_d', '-ld', metavar='LRD',dest='lrd', type=float, default=1e-4, help='Learning rate for Discriminator')    
    parser.add_argument('--lossl1weight', '-l1weight', metavar='L1W',dest='l1weight', type=float, default=1.0, help='L1 Loss Weight')    
    parser.add_argument('--losslpipsweight', '-lpipsweight', metavar='LPIPSW',dest='lpipsweight', type=float, default=1.0, help='LPIPS Loss Weight')    
    parser.add_argument('--lossadvweight', '-advweight', metavar='ADVW',dest='advweight', type=float, default=0.05, help='Adversarial Loss Weight')   
    parser.add_argument('--imgscale', '-imgscale', dest='imgscale', type=int, default=512, help='Size of the image')    
  
    
    return parser.parse_args()


if __name__ == '__main__':
    args=get_args()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # n_channels=5 for RGB+ InputAge + Target Age
    # n_classes is the number of probabilities you want to get per output pixel
    model = UNet(n_channels=5, n_classes=3, bilinear=args.bilinear) # To get the predicted RGB age delta
    model.to(device=device)
    
    # Initialize the PatchGAN discriminator
    discriminator = PatchGANDiscriminator(in_channels=4) # RGB Image + target age (4 channel)
    discriminator.to(device=device)
    
    
    train_model(
        model=model,
        discriminator=discriminator,
        checkpoint_file=args.chkpt,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate_g=args.lrg,
        learning_rate_d=args.lrd,
        l1_weight=args.l1weight,
        lpips_weight=args.lpipsweight,
        adv_weight =args.advweight,
        resume_checkpoint=args.resume,
        img_scale=args.imgscale
        )
   