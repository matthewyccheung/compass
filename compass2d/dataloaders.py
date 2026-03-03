"""Simple 2D dataset loaders used by the COMPASS experiments."""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

class ebhi(Dataset):
    def __init__(self,img_paths=None,seg_paths=None,img_dir=None,seg_dir=None,transform=False,size=128):
        if (img_paths==None)&(seg_paths==None):
            self.img_dir=img_dir
            self.seg_dir=seg_dir
            self.img_paths=None
            self.seg_paths=None
            self.ids=os.listdir(img_dir)
        elif (img_dir==None)&(seg_dir==None):
            self.img_paths=img_paths
            self.seg_paths=seg_paths
            self.img_dir=None
            self.seg_dir=None
        self.transform=transform
        self.size=size
        self.resize=transforms.Resize(size=(size,size))
    def __len__(self):
        if self.img_paths==None:
            return len(self.ids)
        else:
            return len(self.img_paths)
    def __getitem__(self,idx):
        if self.img_dir==None:
            img_path=self.img_paths[idx]
            seg_path=self.seg_paths[idx]
        elif self.img_paths==None:
            img_path=f'{self.img_dir}{self.ids[idx]}'
            seg_path=f'{self.seg_dir}{self.ids[idx]}'    
        img=Image.open(img_path).convert('RGB')
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

class ham10k(Dataset):
    def __init__(self,img_paths=None,seg_paths=None,img_dir=None,seg_dir=None,
                 transform=False,size=128):
        if (img_paths==None)&(seg_paths==None):
            self.img_dir=img_dir
            self.seg_dir=seg_dir
            self.img_paths=None
            self.seg_paths=None
            self.img_ids=os.listdir(img_dir)
            self.seg_ids=os.listdir(seg_dir)
        elif (img_dir==None)&(seg_dir==None):
            self.img_paths=img_paths
            self.seg_paths=seg_paths
            self.img_dir=None
            self.seg_dir=None
        self.transform=transform
        self.size=size
        self.resize=transforms.Resize(size=(size,size))
    def __len__(self):
        if self.img_paths==None:
            return len(self.img_ids)
        else:
            return len(self.img_paths)
    def __getitem__(self,idx):
        if self.img_dir==None:
            img_path=self.img_paths[idx]
            seg_path=self.seg_paths[idx]
        elif self.img_paths==None:
            img_path=f'{self.img_dir}{self.img_ids[idx]}'
            seg_path=f'{self.seg_dir}{self.seg_ids[idx]}'    
        img=Image.open(img_path).convert('RGB')
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

class tn3k(Dataset):
    def __init__(self,img_dir,seg_dir,transform=False,size=128):
        self.img_dir=img_dir
        self.seg_dir=seg_dir
        self.transform=transform
        self.size=size
        self.resize=transforms.Resize(size=(size,size))
        self.ids=[filename.split('.jpg')[0] for filename in os.listdir(img_dir)]
    def __len__(self):
        return len(self.ids)
    def __getitem__(self,idx):
        img_path=f'{self.img_dir}{self.ids[idx]}.jpg'
        img=Image.open(img_path).convert('L')
        seg_path=f'{self.seg_dir}{self.ids[idx]}.jpg'
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

class kvasir(Dataset):
    def __init__(self,img_dir,seg_dir,transform=False,size=128):
        self.img_dir=img_dir
        self.seg_dir=seg_dir
        self.transform=transform
        self.size=size
        self.resize=transforms.Resize(size=(size,size))
        self.ids=[filename.split('.jpg')[0] for filename in os.listdir(img_dir)]
    def __len__(self):
        return len(self.ids)
    def __getitem__(self,idx):
        img_path=f'{self.img_dir}{self.ids[idx]}.jpg'
        img=Image.open(img_path)
        seg_path=f'{self.seg_dir}{self.ids[idx]}.jpg'
        seg=Image.open(seg_path).convert('L')
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

 
