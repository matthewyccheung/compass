import os
import torch
from torch.utils.data import Dataset,DataLoader,Subset,random_split
from monai.networks.nets import UNet
from monai.data import decollate_batch
from monai.transforms import Activations,AsDiscrete,Compose
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)
from compass2d.dataloaders import ham10k  # noqa: E402
from datetime import datetime

device='cuda:0'
data_dir='/scratch/yc130/ham10k/'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
root_dir=f'{data_dir}/models/UNet_{str(timestamp)}/'
if os.path.exists(root_dir)==False:
    os.mkdir(root_dir)
img_dir=f'{data_dir}HAM10000_imgs/'
seg_dir=f'{data_dir}HAM10000_segs/'
size=128
batch_size=32
shuffle=True
num_workers=4

dataset_size=len(os.listdir(img_dir))
train_set=ham10k(img_dir=f'{data_dir}imagesTr/',seg_dir=f'{data_dir}labelsTr/',transform=True,size=128)
val_set=ham10k(img_dir=f'{data_dir}imagesVa/',seg_dir=f'{data_dir}labelsVa/',transform=False,size=128)
test_set=ham10k(img_dir=f'{data_dir}imagesTs/',seg_dir=f'{data_dir}labelsTs/',transform=False,size=128)
train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
val_loader=DataLoader(val_set,batch_size=1,shuffle=False,num_workers=num_workers)
test_loader=DataLoader(test_set,batch_size=1,shuffle=False,num_workers=num_workers)

model_config={'spatial_dims':2,
              'in_channels':3,
              'out_channels':1,
              'channels':(32,64,128,256),
              'strides':(2,2,2),
              'num_res_units':2,
              'norm':'batch',
              'dropout':0.1}
model=UNet(**model_config).to(device)
model.load_state_dict(torch.load('/scratch/yc130/ham10k/models/UNet_20250603_162419/model_37.pth'))

loss_fn=DiceLoss(smooth_nr=0,smooth_dr=1e-5,squared_pred=True,to_onehot_y=False,sigmoid=True)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
dice_metric=DiceMetric(include_background=True,reduction="mean")
post_trans=Compose([Activations(sigmoid=True),AsDiscrete(threshold=0.5)])

vol_pred=[]
vol_gt=[]
with torch.no_grad():
    running_vloss=0.0
    bottlenecks=[]
    for i,(val_x,val_y) in enumerate(val_loader):
        print(i)
        x1=model.model[0](val_x.to(device))
        bottlenecks.append(x1)
        x2=model.model[1](x1)
        val_yhat=model.model[2](x2)
        val_y=val_y.to(device)
        vloss=loss_fn(val_yhat,val_y)
        dice_metric([post_trans(vi) for vi in decollate_batch(val_yhat)],val_y)
        running_vloss+=vloss.item()
        vol_pred.append(val_yhat.sum())
        vol_gt.append(val_y.sum())
    avg_vloss=running_vloss/len(val_loader)
    avg_vdice=dice_metric.aggregate().item()
    dice_metric.reset()
vol_pred=torch.stack(vol_pred).ravel()
vol_gt=torch.stack(vol_gt).ravel()
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(vol_pred,vol_gt)
plt.savefig('scatter.png')
