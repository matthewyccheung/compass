"""Model-specific forward hooks and scalar targets (metrics) for COMPASS 2D."""

import torch
from monai.transforms import Activations, AsDiscrete, Compose

def temp_scale(x,temp=10000.0):
    return torch.sigmoid(temp*x)

def post_trans_mask_sum(seg, channel, sigmoid=True):
    """
    Default scalar target: sum of predicted mask pixels (area proxy).

    Returns:
      - seg: discrete mask (B, H, W)
      - v:   scalar per item (B,)
    """
    if sigmoid:
        trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        seg = trans(seg)[:, channel, ...]
    v = seg.flatten(1).sum(1)
    return seg, v


def post_trans_mask_sum_diff(x, channel, sigmoid=True):
    """
    Differentiable version of `post_trans_mask_sum` used for Jacobian-based methods.
    """
    if sigmoid:
        seg = temp_scale(x)[:, channel, ...]
    v = seg.flatten(1).sum(1)
    return seg, v

def post_trans_centroid_y(seg, channel):
    """Calculates the Y-coordinate of the center of mass on the discrete mask."""
    # from monai.transforms import Compose, Activations, AsDiscrete
    # trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # seg = trans(x)[:, channel, ...]
    
    h = seg.shape[-2]
    grid_y = torch.arange(h, device=seg.device, dtype=torch.float32).view(1, h, 1)
    
    mass = seg.flatten(1).sum(1)
    
    centroid_y = torch.zeros_like(mass)
    valid_mask = mass > 0
    centroid_y[valid_mask] = (seg[valid_mask] * grid_y).flatten(1).sum(1) / mass[valid_mask]
    
    return seg,centroid_y

def post_trans_centroid_y_diff(x, channel):
    """Differentiable Y-coordinate of the center of mass."""
    seg = temp_scale(x)[:, channel, ...]
    
    h = seg.shape[-2]
    grid_y = torch.arange(h, device=x.device, dtype=torch.float32).view(1, h, 1)
    
    mass = seg.flatten(1).sum(1)
    
    centroid_y = torch.zeros_like(mass)
    valid_mask = mass > 1e-6
    centroid_y[valid_mask] = (seg[valid_mask] * grid_y).flatten(1).sum(1) / mass[valid_mask]
    
    return seg, centroid_y
    
def get_fns(arch,post_trans_diff):
    if arch=='UNet':
        def compute_jacobian(model,latent,channel):
            latent=latent.requires_grad_(True)
            yhat=forward_latent(model,latent)
            seg,vhat=post_trans_diff(yhat,channel)
            grads=[]
            for i in range(vhat.shape[0]):
                grad=torch.autograd.grad(vhat[i].ravel(),latent,
                                         retain_graph=True,create_graph=False,
                                         only_inputs=True)[0][i]  # take ith sample
                grads.append(grad.detach().cpu())  # shape: [C, H, W]
            return yhat,vhat,torch.stack(grads)  # shape: [B, C, H, W]
        def forward_x(model,x):
            x1=model.model[0](x)
            latent=model.model[1](x1)
            return latent
        def forward_latent(model,latent):
            return model.model[2](latent)
    elif arch=='UNetShallow':
        def compute_jacobian(model,latent,channel):
            latent=latent.requires_grad_(True)
            yhat=forward_latent(model,latent)
            seg,vhat=post_trans_diff(yhat,channel)
            grads=[]
            for i in range(vhat.shape[0]):
                grad=torch.autograd.grad(vhat[i].ravel(),latent,
                                         retain_graph=True,create_graph=False,
                                         only_inputs=True)[0][i]  # take ith sample
                grads.append(grad.detach().cpu())  # shape: [C, H, W]
            return yhat,vhat,torch.stack(grads)  # shape: [B, C, H, W]
        def forward_x(model,x):
            x1=model.model[0](x)
            x2=model.model[1](x1)
            latent=model.model[2][0](x2)
            return latent
        def forward_latent(model,latent):
            return model.model[2][1](latent)
    elif arch=='UNetBottleneck':
        def compute_jacobian(model,latent,channel):
            latent=latent.requires_grad_(True)
            yhat=forward_latent(model,latent)
            seg,vhat=post_trans_diff(yhat,channel)
            grads=[]
            for i in range(vhat.shape[0]):
                grad=torch.autograd.grad(vhat[i].ravel(),latent,
                                         retain_graph=True,create_graph=False,
                                         only_inputs=True)[0][i]  # take ith sample
                grads.append(grad.detach().cpu())  # shape: [C, H, W]
            return yhat,vhat,torch.stack(grads)  # shape: [B, C, H, W]
        def forward_x(model,x):
            latent=model.model[0](x)
            return latent
        def forward_latent(model,latent):
            x1=model.model[1](latent)
            x2=model.model[2][0](x1)
            return model.model[2][1](x2)
    elif arch=='UNetLogits':
        def compute_jacobian(model,latent,channel):
            latent=latent.requires_grad_(True)
            yhat=forward_latent(model,latent)
            seg,vhat=post_trans_diff(yhat,channel)
            grads=[]
            for i in range(vhat.shape[0]):
                grad=torch.autograd.grad(vhat[i].ravel(),latent,
                                         retain_graph=True,create_graph=False,
                                         only_inputs=True)[0][i]  # take ith sample
                grads.append(grad.detach().cpu())  # shape: [C, H, W]
            return yhat,vhat,torch.stack(grads)  # shape: [B, C, H, W]
        def forward_x(model,x):
            x1=model.model[0](x)
            x2=model.model[1](x1)
            x3=model.model[2][0](x2)
            latent=model.model[2][1](x3)
            return latent
        def forward_latent(model,latent):
            return latent
    elif arch=='SegResNet':
        def compute_jacobian(model,latent,channel):
            latent=latent.requires_grad_(True)
            yhat=forward_latent(model,latent)
            seg,vhat=post_trans_diff(yhat,channel)
            grads=[]
            for i in range(vhat.shape[0]):
                grad=torch.autograd.grad(vhat[i].ravel(),latent,
                                         retain_graph=True,create_graph=False,
                                         only_inputs=True)[0][i]  # take ith sample
                grads.append(grad.detach().cpu())  # shape: [C, H, W]
            return yhat,vhat,torch.stack(grads)  # shape: [B, C, H, W]
        def forward_x(model,x):
            x,down_x=model.encode(x)
            down_x.reverse()
            for i,(up,upl) in enumerate(zip(model.up_samples,model.up_layers)):
                x=up(x)+down_x[i+1]
                x=upl(x)
            return x
        def forward_latent(model,latent):
            x = model.conv_final(latent)
            return x
    return forward_x,forward_latent,compute_jacobian
