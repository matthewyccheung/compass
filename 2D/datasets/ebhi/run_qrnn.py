import os
import torch
from torch.utils.data import Dataset,DataLoader,Subset,random_split
from monai.networks.nets import UNet, SegResNet
from monai.data import decollate_batch
from monai.transforms import Activations,AsDiscrete,Compose
from monai.losses import DiceLoss,TverskyLoss
from monai.metrics import DiceMetric
from datetime import datetime
import pickle
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)
from compass2d.dataloaders import *  # noqa: E402

# shift='standard'
# shift='test'
shift=None
# shift='prevalence'
size=128
batch_size=32
shuffle=True
num_workers=16

# arch='QRUNet'
arch='QRSegResNet'
root_dir='/scratch/yc130/EBHI-SEG/'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

if shift==None:
    model_dir=f'{root_dir}models/{arch}_{str(timestamp)}/'
else:
    model_dir=f'{root_dir}models/{arch}_{shift}_{str(timestamp)}/'
if os.path.exists(model_dir)==False:
    os.makedirs(model_dir,exist_ok=True)
device='cuda:0'

if shift==None:
    train_set=ebhi(img_dir=f'{root_dir}imagesTr/',seg_dir=f'{root_dir}labelsTr/',size=128,transform=True)
    val_set=ebhi(img_dir=f'{root_dir}imagesVa/',seg_dir=f'{root_dir}labelsVa/',size=128,transform=False)
    test_set=ebhi(img_dir=f'{root_dir}imagesTs/',seg_dir=f'{root_dir}labelsTs/',size=128,transform=False)
else:
    with open(f'{root_dir}{shift}_fnames.pkl','rb') as f:
        fnames=pickle.load(f)
    tr_img_files=fnames["train_images"]
    tr_seg_files=fnames["train_labels"]
    tr_labels=fnames["train_classes"]
    va_img_files=fnames["cal_images"]
    va_seg_files=fnames["cal_labels"]
    va_labels=fnames["cal_classes"]
    ts_img_files=fnames["test_images"]
    ts_seg_files=fnames["test_labels"]
    ts_labels=fnames["test_classes"]
    train_set=ebhi(img_paths=tr_img_files,seg_paths=tr_seg_files,size=128,transform=True)
    val_set=ebhi(img_paths=va_img_files,seg_paths=va_seg_files,size=128,transform=False)
    test_set=ebhi(img_paths=ts_img_files,seg_paths=ts_seg_files,size=128,transform=False)
train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
val_loader=DataLoader(val_set,batch_size=1,shuffle=False,num_workers=num_workers)
test_loader=DataLoader(test_set,batch_size=1,shuffle=False,num_workers=num_workers)

if arch=='QRUNet':
    model_config={'spatial_dims':2,
                  'in_channels':3,
                  'out_channels':3,
                  'channels':(32,64,128,256),
                  'strides':(2,2,2),
                  'num_res_units':2,
                  'norm':'batch',
                  'dropout':0.1}
    model_class=UNet
elif arch=='QRSegResNet':
    model_config = {
        'spatial_dims': 2,
        'in_channels': 3,
        'out_channels': 3,
        'init_filters': 32,
        'blocks_down': (1, 2, 2, 4),
        'blocks_up': (1, 1, 1),
        'norm': 'batch',
        'dropout_prob': 0.1,
        'use_conv_final': True,
    }
    model_class=SegResNet
model=model_class(**model_config).to(device)

# loss_fn=DiceLoss(smooth_nr=0,smooth_dr=1e-5,squared_pred=True,to_onehot_y=False,sigmoid=True)
gamma=0.2
loss_fn_lo=TverskyLoss(alpha=1-gamma,beta=gamma,sigmoid=True)
loss_fn_hi=TverskyLoss(alpha=gamma,beta=1-gamma,sigmoid=True)
loss_fn=TverskyLoss(alpha=0.5,beta=0.5,sigmoid=True)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
dice_metric=DiceMetric(include_background=True,reduction='mean')
post_trans=Compose([Activations(sigmoid=True),AsDiscrete(threshold=0.5)])

# run=wandb.init(**wandb_config)
num_epochs=100
best_vloss=1e5
for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    running_dice=0.0
    running_dice_batch=0.0
    for i,(x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        x=x.to(device)
        y=y.to(device)        
        yhat=model(x)
        loss=loss_fn(yhat[:,0,...].unsqueeze(1),y)+loss_fn_lo(yhat[:,1,...].unsqueeze(1),y)+loss_fn_hi(yhat[:,2,...].unsqueeze(1),y)
        dice_metric([post_trans(ti) for ti in decollate_batch(yhat[:,0,...].unsqueeze(1))],y)
        running_loss+=loss.item()
        loss.backward()
        optimizer.step()
    avg_tloss=running_loss/len(train_loader)
    avg_tdice=dice_metric.aggregate().item()
    dice_metric.reset()
    model.eval()
    with torch.no_grad():
        running_vloss=0.0
        for i,(val_x,val_y) in enumerate(val_loader):
            val_yhat=model(val_x.to(device))
            val_y=val_y.to(device)
            vloss=loss_fn(val_yhat[:,0,...].unsqueeze(1),val_y)+loss_fn_lo(val_yhat[:,1,...].unsqueeze(1),val_y)+loss_fn_hi(val_yhat[:,2,...].unsqueeze(1),val_y)
            dice_metric([post_trans(vi) for vi in decollate_batch(val_yhat[:,0,...].unsqueeze(1))],val_y)
            running_vloss+=vloss.item()
        avg_vloss=running_vloss/len(val_loader)
        avg_vdice=dice_metric.aggregate().item()
        dice_metric.reset()
        model_path=f'{model_dir}model_{str(epoch)}.pth'
        torch.save(model.state_dict(),model_path)
        if avg_vloss<best_vloss:
             best_vloss=avg_vloss
             model_path=f'{model_dir}best_model.pth'
             torch.save(model.state_dict(),model_path)
    print(f'Epoch {str(epoch)}, Tr Loss {avg_tloss:.4f}, Tr Dice: {avg_tdice:.4f}, Val Loss {avg_vloss:.4f}, Val Dice: {avg_vdice:.4f}')
#     run.log({'Epoch':epoch+1,'Tr Loss':avg_tloss,'Tr Dice':avg_tdice,'Tr Batch Dice':avg_tdice_batch,
#              'Val Loss':avg_vloss,'Val Dice':avg_vdice,'Val SSIM':avg_vdice_batch,
#              'Generalization Error Dice':avg_tdice-avg_vdice,'Generalization Error Batch Dice':avg_tdice_batch-avg_vdice_batch})
# run.finish()
