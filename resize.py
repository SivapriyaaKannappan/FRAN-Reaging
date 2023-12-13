# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:43:59 2023

@author: Sivapriyaa
"""
import os,sys
from PIL import Image
from pathlib import Path
data_folder=Path("D:/E2ID/DownloadedfromGithub/GANs/AgeTransformation/FRAN/dataset/")
output_data_folder= Path("D:/E2ID/DownloadedfromGithub/GANs/AgeTransformation/FRAN/resized_dataset/")
train_percent=0.7
val_percent=0.2
test_percent=0.1
def resize():
    print("Hello")
    for root, dirs, files in os.walk(data_folder):
        for idx in range(len(dirs)):
            data_folder1=os.path.join(data_folder,dirs[idx])
            for root1, dirs1, files1 in os.walk(data_folder1):
                for i,fname in enumerate(files1):
                    train_split=int(train_percent*len(files1))
                    val_split=int(val_percent*len(files1))
                    test_split=int(test_percent*len(files1))
                    if (i<train_split):
                        im=Image.open(os.path.join(data_folder1,fname))
                        filename=fname.split('.',2)[0]
                        imresize=im.resize((256,256), Image.ANTIALIAS)
                        output_path=os.path.join(output_data_folder,'train',dirs[idx])
                        if not os.path.exists(output_path):
                            os.mkdir(output_path)
                        imresize.save(os.path.join(output_path,filename+'.jpg'))
                   
                    elif (train_split <= i < train_split+val_split):
                        im=Image.open(os.path.join(data_folder1,fname))
                        filename=fname.split('.',2)[0]
                        imresize=im.resize((256,256), Image.ANTIALIAS)
                        output_path=os.path.join(output_data_folder,'val',dirs[idx])
                        if not os.path.exists(output_path):
                            os.mkdir(output_path)
                        imresize.save(os.path.join(output_path,filename+'.jpg'))
                    elif (train_split+val_split <= i < len(files1)):
                         im=Image.open(os.path.join(data_folder1,fname))
                         filename=fname.split('.',2)[0]
                         imresize=im.resize((256,256), Image.ANTIALIAS)
                         output_path=os.path.join(output_data_folder,'test',dirs[idx])
                         if not os.path.exists(output_path):
                             os.mkdir(output_path)
                         imresize.save(os.path.join(output_path,filename+'.jpg'))
                        
if __name__ == '__main__':
    resize()