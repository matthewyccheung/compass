import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

class ebhi(Dataset):
    def __init__(self,img_dir,seg_dir,transform=False,size=128):
        self.img_dir=img_dir
        self.seg_dir=seg_dir
        self.transform=transform
        self.size=size
        self.resize=transforms.Resize(size=(size,size))
        self.ids=os.listdir(img_dir)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self,idx):
        img_path=f'{self.img_dir}{self.ids[idx]}'
        img=Image.open(img_path).convert('RGB')
        seg_path=f'{self.seg_dir}{self.ids[idx]}'
        seg=Image.open(seg_path)
        # Resize
        img=self.resize(img)
        seg=self.resize(seg)
        if self.transform:
            img,seg=self._transform(img,seg)
        # Transform to tensor
        img=TF.to_tensor(img)
        seg=(TF.to_tensor(seg)>0.5).float()
        return img,seg
    def _transform(self,img,seg):
        # Random crop
        i,j,h,w=transforms.RandomCrop.get_params(img,output_size=(self.size,self.size))
        img=TF.crop(img,i,j,h,w)
        seg=TF.crop(seg,i,j,h,w)
        # Random horizontal flipping
        if torch.rand(1)>0.5:
            img=TF.hflip(img)
            seg=TF.hflip(seg)
        # Random vertical flipping
        if torch.rand(1)>0.5:
            img=TF.vflip(img)
            seg=TF.vflip(seg)
        return img,seg