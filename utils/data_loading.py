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
import dlib
from utils.align_all_parallel import align_face
from torchvision import transforms

def get_transform(preproc, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in preproc:
        osize = [params['load_size'], params['load_size']]
        transform_list.append(transforms.Resize(osize, method))

    if 'colorjitter' in preproc:
        transform_list.append(transforms.ColorJitter(
            params['colorjitter_brightness'], params['colorjitter_contrast'], 
            params['colorjitter_saturation'], params['colorjitter_hue']))

    if 'rotation' in preproc:
        transform_list.append(transforms.RandomRotation(params['rotation_degrees']))
        
    if 'crop' in preproc:
        if params is None:
            transform_list.append(transforms.RandomCrop(params['crop_size']))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], params['crop_size'])))
        
    if preproc is None:
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if 'flip' in preproc:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            # transform_list += [transforms.Normalize((0.5,), (0.5,))]
            transform_list += [transforms.Normalize((0.,), (1.0,))]
        else:
            # transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform_list += [transforms.Normalize((0., 0., 0.), (1.0, 1.0, 1.0))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
        

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

class AgeDataset(Dataset):
    def __init__(self, images_dir, img_scale, preproc=[]):
        self.images_dir=images_dir
        self.age_ids=os.listdir(images_dir)
        self.img_ids=[img_name.split('.')[0] for img_name in sorted(os.listdir(str(images_dir)+'/'+self.age_ids[0])) ]
        self.n_img_ids = len(self.img_ids)
        self.n_age_ids = len(self.age_ids)
        self.tot_imgs =  self.n_img_ids * self.n_age_ids
        self.img_scale=img_scale
        # Image to indices 
        imgtoind={}
        for imgidx in range(self.n_img_ids ):
            imgtoind[imgidx] =  self.img_ids[imgidx]
        # print("Image to index:", imgtoind)
        # print("Length of image to index:",len(imgtoind))
        self.imgtoind=imgtoind
        self.n_imgtoind=len(imgtoind)
        # Age to indices
        ind2age={}
        start=int(self.age_ids[0])
        step=5
        stop=int(self.age_ids[-1])+step
        
        for ageidx, ages in enumerate(range(start,stop,step)):
            ind2age[ageidx]=ages
        # print("Age to index:", agetoind)
        self.ind2age=ind2age
        
        # Consider a matrix with age as columns and image filenames as rows
        self.onedto2dind=[]
        for jdx in range(self.tot_imgs):
            row=jdx//self.n_age_ids
            col=jdx % self.n_age_ids
            self.onedto2dind.append((row, col))
       
        self.preproc = preproc
            
    def __getitem__(self, idx): # idx-list of image and age tuples
        idxA, idxB = idx // self.tot_imgs, idx % self.tot_imgs
        (imgA_idx,ageA_idx)=self.onedto2dind[idxA]   # img_id and age_id
        imgA_id=self.imgtoind[imgA_idx]
        ageA=self.ind2age[ageA_idx]
        (imgB_idx,ageB_idx)=self.onedto2dind[idxB]   # img_id and age_id
        imgB_id=self.imgtoind[imgB_idx]
        ageB=self.ind2age[ageB_idx]
        
        imgA_path = os.path.join(str(self.images_dir),self.age_ids[ageA_idx],self.img_ids[imgA_idx]).replace("\\", '/')+'.jpg'
        imgA=Image.open(imgA_path)
        imgB_path = os.path.join(str(self.images_dir),self.age_ids[ageB_idx],self.img_ids[imgB_idx]).replace("\\", '/')+'.jpg'
        imgB=Image.open(imgB_path)
        
        # img=img.resize((self.img_scale,self.img_scale),Image.ANTIALIAS) # resize the image
        # preproc = ['colorjitter', 'rotation', 'crop']
        # preproc = ['colorjitter', 'rotation', 'resize']
        params = {'colorjitter_brightness': [random.uniform(0.9, 1.1)]*2,
                  'colorjitter_contrast': [random.uniform(0.9, 1.1)]*2,
                  'colorjitter_saturation': [random.uniform(0.9, 1.1)]*2,
                   'colorjitter_hue': [random.uniform(0, 0)]*2, 
                  'rotation_degrees': [random.uniform(-3, 3)]*2,
                  'crop_size': self.img_scale,
                  'crop_pos': [random.randint(0, np.maximum(0, imgA.size[0] - self.img_scale)),
                               random.randint(0, np.maximum(0, imgA.size[1] - self.img_scale))],
                  'load_size': 512
                  }
        transform = get_transform(self.preproc, params)
        imgA = transform(imgA)
        imgB = transform(imgB)

        return imgA, imgA_id, ageA, imgB, imgB_id, ageB
        
    def __len__(self):
        return self.n_age_ids * self.n_img_ids # 3 * 10 = 30 images (16 * 2000 = 32,000 images available in the dataset)

           
class AgeBatchSampler(BatchSampler):
    """ Stratified Sampling - Samples n_classes, and for each of these classes sample n_sample_per_class.
    Returns bacthes of size n_classes * n_sample_per_class """
    def __init__(self,n_age_ids, n_img_ids, n_classes_per_batch, n_samples_per_class):
        
        self.n_classes_per_batch = n_classes_per_batch    #4
        self.n_samples_per_class = n_samples_per_class #2
        self.batch_size = self.n_classes_per_batch * self.n_samples_per_class # 4 * 2 = 8 
        self.n_img_ids=n_img_ids
        self.img_ids = np.arange(0, self.n_img_ids)
        self.n_age_ids=n_age_ids
        # self.samples_count= self.n_img_ids * self.n_age_ids  # 1400 * 16 = 22,400
        self.total_sample_count = self.n_img_ids * self.n_age_ids  #Maximum possible image-pair combinations (1400 * 16 = 22,400)
       
        self.count=0
        
    def __iter__(self):
        self.count=0
        # start=0
        permuted_img_ids = np.random.permutation(self.img_ids) # image ids are shuffled every epoch
        # while self.count + self.batch_size <= self.n_img_ids * self.batch_size:   # For testing
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
                samples = [samples[2*i]*self.total_sample_count + samples[2*i + 1] for i in range(self.n_classes_per_batch)]
                yield samples
                         
    def __len__(self):  # Maximum possible combinations 
        # n_batch = self.total_samples_count // self.batch_size  # 512,000 / 8 = 64,000 iterations
        n_batch = self.total_sample_count // self.batch_size  # 16,000 / 8 = 2000 iterations
        return n_batch


