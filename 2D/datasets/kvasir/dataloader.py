import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

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

if __name__=='__main__':
    root_dir='/scratch/yc130/Kvasir-SEG/'
    dataset=kvasir(img_dir=f'{root_dir}imagesTr/',seg_dir=f'{root_dir}labelsTr/',size=128)
    dataloader=DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4)
    for i,(img,seg) in enumerate(dataloader):
        print(img.min(),img.max(),img.shape)
        print(seg.min(),seg.max(),seg.shape)
        break
    breakpoint()