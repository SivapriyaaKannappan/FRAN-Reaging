# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:14:18 2023

@author: Sivapriyaa

The dataset has 16 different age categories from 10 to 85 in step 5.
There are 2000 identities. So totally we have 2000 * 16 =32,000 images in the dataset.

Training comprises of aged/de-aged samples for each identity, images should be randomly retrieved as pairs for each identity.

As there are 16 categories of ages, there are 256 image pairs per identity, including itself.

In that case, overall 2000 * 256 = 512,000 possible image pairs can be trained per epoch, therefore we need 64,000 iterations.
For Simplicity,
If the mini-batch has 1 identity with n_sample_per_cls=8, then the size of the mini-batch is 8 with 2000 iterations.

This can be tweaked later.
"""

import torch
import os
import numpy as np
from PIL import Image
from os import listdir
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
# from itertools import product # cartesian product
import random

class AgeDataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir=images_dir
        self.age_ids=os.listdir(images_dir)
        self.img_ids=[img_name.split('.')[0] for img_name in sorted(os.listdir(str(images_dir)+'/'+self.age_ids[0])) ]
        self.n_img_ids = len(self.img_ids)
        self.n_age_ids = len(self.age_ids)
        self.tot_imgs =  self.n_img_ids * self.n_age_ids
        
        # Image to indices 
        imgtoind={}
        for imgidx in range(self.n_img_ids ):
            imgtoind[imgidx] =  self.img_ids[imgidx]
        # print("Image to index:", imgtoind)
        # print("Length of image to index:",len(imgtoind))
        self.imgtoind=imgtoind
        self.n_imgtoind=len(imgtoind)
        # Age to indices
        agetoind={}
        start=int(self.age_ids[0])
        step=5
        stop=int(self.age_ids[-1])+step
        
        for ageidx, ages in enumerate(range(start,stop,step)):
            agetoind[ageidx]=ages
        # print("Age to index:", agetoind)
        self.agetoind=agetoind
        
        # Consider a matrix with age as columns and image filenames as rows
        self.onedto2dind=[]
        for jdx in range(self.tot_imgs):
            row=jdx//self.n_age_ids
            col=jdx % self.n_age_ids
            self.onedto2dind.append((row, col))
       
    def __getitem__(self, idx): # idx-list of image and age tuples
        (img_idx,age_idx)=self.onedto2dind[idx]   # img_id and age_id
        img_id=self.imgtoind[img_idx]
        age=self.agetoind[age_idx]
        self.img_path = os.path.join(str(self.images_dir),self.age_ids[age_idx],self.img_ids[img_idx]).replace("\\", '/')+'.jpg'
        img=Image.open(self.img_path)
        # img.save(os.path.join("./data/valid_vis/","test.png"))
        img=np.array(img)
        return img, img_id, age
        
    def __len__(self):
        return self.n_age_ids * self.n_img_ids # 3 * 10 = 30 images (16 * 2000 = 32,000 images available in the dataset)

           
class AgeBatchSampler(BatchSampler):
    """ Stratified Sampling - Samples n_classes, and for each of these classes sample n_sample_per_class.
    Returns bacthes of size n_classes * n_sample_per_class """
    def __init__(self,n_age_ids, n_img_ids, n_classes_per_batch, n_samples_per_class):
        
        self.n_classes_per_batch = n_classes_per_batch    #1
        self.n_samples_per_class = n_samples_per_class #8
        self.batch_size = self.n_classes_per_batch * self.n_samples_per_class # 1* 8 = 8 
        self.n_img_ids=n_img_ids
        self.img_ids = np.arange(0, self.n_img_ids)
        self.n_age_ids=n_age_ids
        self.samples_count= self.n_img_ids * self.n_age_ids  # 10 * 3 = 30 (  # 2000 * 8 = 16,000)
        self.total_sample_count = self.n_img_ids * self.n_age_ids  #Maximum possible image-pair combinations(2000 * 256 = 512,000)
       
        self.count=0
        
    def __iter__(self):
        self.count=0
        # start=0
        permuted_img_ids = np.random.permutation(self.img_ids)
        
        while self.count + self.batch_size <= self.total_sample_count:   # Actual samples used
                # samples=random.sample(range( self.samples_count), k=8) # Sample 8 integers/mini-batch that represent the matrix cells 
                # samples = random.choices(range(start, start + self.n_age_ids), k=self.batch_size) # Sample 8 integers/mini-batch from the same identity that represent the matrix cells 
                # start += self.n_age_ids
                sid = self.count // self.batch_size
                sid = sid % self.n_img_ids
                eid = sid + self.n_classes_per_batch
                eid = eid % self.n_img_ids
                if sid < eid:
                    img_ids = permuted_img_ids[sid:eid].tolist()
                else:
                    img_ids = permuted_img_ids[sid:].tolist()
                    img_ids += permuted_img_ids[:eid].tolist()
                samples = []
                for img_id in img_ids:
                    sample = random.choices(range(img_id * self.n_age_ids, (img_id + 1) * self.n_age_ids), k=2)
                    samples += sample
                self.count += self.batch_size
                yield samples
                         
    def __len__(self):  # Maximum possible combinations 
        # n_batch = self.total_samples_count // self.batch_size  # 512,000 / 8 = 64,000 iterations
        n_batch = self.total_sample_count // self.batch_size  # 16,000 / 8 = 2000 iterations
        return n_batch




