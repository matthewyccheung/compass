import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import math
from torch.utils.data import Subset,ConcatDataset,DataLoader
import torch
from .data_utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms.functional as TF

def gaussian_kernel(x,mag,J):
    smooth_grad = TF.gaussian_blur(J, sigma=0.1, kernel_size=3)  # Non-uniform spatial regulation
    delta=smooth_grad/smooth_grad.norm()
    x+=mag*delta
    return x

def j_kernel(x,J):
    x+=mag*J/J.norm(p=2)
    # x+=J
    return x

def get_spatial_func(spatial):
    if spatial==None:
        spatial_func=lambda x,mag,J: x
    elif spatial=='Gaussian':
        spatial_func=gaussian_kernel
    elif spatial=='J':
        spatial_func=j_kernel
        
    return spatial_func

def _find_beta_binary_logits_asymmetric(yhat_i, v_i, post_trans, channel, device, 
                                        beta_range=1.0, max_search_steps=100,
                                        search_threshold=1e-5,spatial=None):
    """
    Finds two non-conformity scores (beta_lo, beta_hi) for a single logit tensor.
    """
    # --- Search for beta_hi (upper bound) ---
    low_hi, high_hi = 0.0, beta_range
    final_beta_hi = high_hi

    spatial_func=get_spatial_func(spatial)
    
    with torch.no_grad():
        for _ in range(max_search_steps):
            if (high_hi - low_hi) < search_threshold:
                break
            mid_hi = (low_hi + high_hi) / 2
            
            # Perturb logits only in the positive direction
            new_yhat = yhat_i + mid_hi
            _, new_vol = post_trans(new_yhat, channel)
    
            if new_vol.item() >= v_i.item():
                # If prediction is high enough, try a smaller beta
                high_hi = mid_hi
                final_beta_hi = mid_hi
            else:
                # If prediction is too low, need a larger beta
                low_hi = mid_hi
                
        # --- Search for beta_lo (lower bound) ---
        low_lo, high_lo = 0.0, beta_range
        final_beta_lo = high_lo
    
        for _ in range(max_search_steps):
            if (high_lo - low_lo) < search_threshold:
                break
            mid_lo = (low_lo + high_lo) / 2
    
            # Perturb logits only in the negative direction
            new_yhat = yhat_i - mid_lo
            _, new_vol = post_trans(new_yhat, channel)
    
            if new_vol.item() <= v_i.item():
                # If prediction is low enough, try a smaller beta
                high_lo = mid_lo
                final_beta_lo = mid_lo
            else:
                # If prediction is too high, need a larger beta
                low_lo = mid_lo

    return (final_beta_lo, final_beta_hi)
    

def _find_beta_binary_logits(yhat_i, v_i, post_trans, channel, device, 
                            beta_range=1.0, max_search_steps=1000, 
                             search_threshold=1e-5):
    """
    Finds the non-conformity score (beta) for a single logit tensor using a binary search.
    The score is the smallest beta that creates a covering interval.
    Search stops when precision threshold is met or max steps are reached.
    """
    low, high = 0.0, beta_range
    final_beta = high  # Default to the max range if the search fails to converge

    with torch.no_grad():
        for _ in range(max_search_steps):
            # Check for precision-based stopping condition
            if (high - low) < search_threshold:
                break
    
            mid = (low + high) / 2
            
            # Perturb logits in both positive and negative directions
            new_yhats = torch.vstack((yhat_i + mid, yhat_i - mid))
            _, new_vols = post_trans(new_yhats, channel)
            
            lower_bound = min(new_vols[0].item(), new_vols[1].item())
            upper_bound = max(new_vols[0].item(), new_vols[1].item())
            
            # Check for coverage
            if lower_bound <= v_i.item() <= upper_bound:
                # If covered, this is a potential solution. Try for a smaller beta.
                high = mid
                final_beta = mid
            else:
                # If not covered, the beta is too small.
                low = mid
            
    return final_beta

def cover(lb,ub,y):
    return ((y>=lb)&(y<=ub)).sum()/len(y)

def _compute_weighted_quantile(scores, weights, quantile_level):
    # This helper function now expects and handles numpy arrays
    scores, weights = np.array(scores), np.array(weights)
    weights = np.maximum(weights, 0)
    sorted_indices = np.argsort(scores)
    sorted_scores, sorted_weights = scores[sorted_indices], weights[sorted_indices]
    total_weight = np.sum(sorted_weights)
    if total_weight <= 0: return np.max(scores)
    target_weight_sum = quantile_level * total_weight
    cumulative_weights = np.cumsum(sorted_weights)
    quantile_index = np.searchsorted(cumulative_weights, target_weight_sum)
    return sorted_scores[min(quantile_index, len(sorted_scores) - 1)]

def COMPASSL(fnames, model, channel, post_trans, device,
              alpha=0.1, beta_range=1.0, 
              max_search_steps=1000, search_threshold=1e-5,
              sym=True,weights=None, return_scores=False, return_raw=False,spatial=None):
    """
    Unified (a)symmetric Conformalized Range Correction on logits with optional WCP.
    """
    # --- 1. Data Loading and Score Calculation ---
    va_vs = load_var(fnames, 'va', 'v').ravel()
    ts_vs = load_var(fnames, 'ts', 'v').ravel()
    
    scores, beta_pos_scores, beta_neg_scores = None, None, None
    if sym:
        scores = []
        for i in range(len(va_vs)):
            yhat_i = load_var_instance(fnames, 'va', 'yhat', i).unsqueeze(0).to(device)
            v_i = va_vs[i].to(device)
            score = _find_beta_binary_logits(yhat_i, v_i, post_trans, 
                                             channel, device, beta_range=beta_range, 
                                           max_search_steps=max_search_steps, 
                                             search_threshold=search_threshold)
            scores.append(score)
        scores = torch.tensor(scores)
    else: # Asymmetric
        betas = []
        for i in range(len(va_vs)):
            yhat_i = load_var_instance(fnames, 'va', 'yhat', i).unsqueeze(0).to(device)
            v_i = va_vs[i].to(device)
            beta_tuple = _find_beta_binary_logits_asymmetric(yhat_i, v_i, post_trans, channel,
                                                             device, beta_range=beta_range,
                                                            max_search_steps=max_search_steps,
                                                             search_threshold=search_threshold)
            betas.append(beta_tuple)
        beta_neg_scores = torch.tensor([b[0] for b in betas])
        beta_pos_scores = torch.tensor([b[1] for b in betas])

    # --- 2. Calibration ---
    beta_q, beta_q_pos, beta_q_neg = 0, 0, 0
    if weights is not None:
        # print(f"Using WCP for {'Symmetric' if sym else 'Asymmetric'} LogitsCRC ({weighting_method})...")
        if sym:
            beta_q = _compute_weighted_quantile(scores.numpy(), weights, 1 - alpha)
        else:
            beta_q_pos = _compute_weighted_quantile(beta_pos_scores.numpy(), weights, 1 - alpha / 2.0)
            beta_q_neg = _compute_weighted_quantile(beta_neg_scores.numpy(), weights, 1 - alpha / 2.0)
    else:
        # print(f"Using standard quantile for {'Symmetric' if sym else 'Asymmetric'} LogitsCRC...")
        n = len(va_vs)
        if sym:
            k = math.ceil((1 - alpha) * (n + 1))
            q_level = min(k / (n + 1), 1.0)
            beta_q = torch.quantile(scores, q_level, interpolation='higher')
        else:
            k_hi = math.ceil((1 - alpha / 2.0) * (n + 1))
            q_hi = min(k_hi / (n + 1), 1.0)
            beta_q_pos = torch.quantile(beta_pos_scores, q_hi, interpolation='higher')
            
            k_lo = math.ceil((1 - alpha / 2.0) * (n + 1))
            q_lo = min(k_lo / (n + 1), 1.0)
            beta_q_neg = torch.quantile(beta_neg_scores, q_lo, interpolation='higher')
            
    # --- 3. Inference ---
    pred_vols_hi, pred_vols_lo = [], []
    with torch.no_grad():
        for i in range(len(ts_vs)):
            ts_yhat = load_var_instance(fnames, 'ts', 'yhat', i).unsqueeze(0).to(device)
            if sym:
                new_yhats = torch.vstack((ts_yhat + beta_q, ts_yhat - beta_q))
                _, new_vols = post_trans(new_yhats, channel)
                pred_vols_lo.append(min(new_vols[0].item(), new_vols[1].item()))
                pred_vols_hi.append(max(new_vols[0].item(), new_vols[1].item()))
            else: # Asymmetric
                yhat_hi = ts_yhat + beta_q_pos
                yhat_lo = ts_yhat - beta_q_neg
                _, vol_hi = post_trans(yhat_hi, channel)
                _, vol_lo = post_trans(yhat_lo, channel)
                pred_vols_lo.append(vol_lo.item())
                pred_vols_hi.append(vol_hi.item())

    pred_vols_hi = torch.tensor(pred_vols_hi)
    pred_vols_lo = torch.tensor(pred_vols_lo)

    logits_len = (pred_vols_hi - pred_vols_lo).mean().item()
    logits_cov = (((ts_vs <= pred_vols_hi) & (ts_vs >= pred_vols_lo)).sum() / len(ts_vs)).item()
    
    mode_str = "Symmetric" if sym else "Asymmetric"
    print(f'{mode_str} COMPASS-L Len: {logits_len:.4f}, Cov: {logits_cov:.4f}')
    
    if return_scores:
        scores_to_return = scores if sym else (beta_neg_scores, beta_pos_scores)
        return logits_len, logits_cov, scores_to_return
    elif return_raw:
        return logits_len, logits_cov, np.stack([pred_vols_hi,pred_vols_lo,ts_vs])
    
    return logits_len, logits_cov











# --- Unified LogitsCRC Function with All WCP Options ---

def Logits(fnames, model, channel, post_trans, device,
              alpha=0.1, beta_range=1.0, 
              max_search_steps=1000, search_threshold=1e-5,
              sym=True,weights=None, return_scores=False, return_raw=False):
    """
    Unified (a)symmetric Conformalized Range Correction on logits with optional WCP.
    """
    # --- 1. Data Loading and Score Calculation ---
    va_vs = load_var(fnames, 'va', 'v').ravel()
    ts_vs = load_var(fnames, 'ts', 'v').ravel()
    
    scores, beta_pos_scores, beta_neg_scores = None, None, None
    if sym:
        scores = []
        for i in range(len(va_vs)):
            yhat_i = load_var_instance(fnames, 'va', 'yhat', i).unsqueeze(0).to(device)
            v_i = va_vs[i].to(device)
            score = _find_beta_binary_logits(yhat_i, v_i, post_trans, channel, device, beta_range=beta_range, 
                                           max_search_steps=max_search_steps, search_threshold=search_threshold)
            scores.append(score)
        scores = torch.tensor(scores)
    else: # Asymmetric
        betas = []
        for i in range(len(va_vs)):
            yhat_i = load_var_instance(fnames, 'va', 'yhat', i).unsqueeze(0).to(device)
            v_i = va_vs[i].to(device)
            beta_tuple = _find_beta_binary_logits_asymmetric(yhat_i, v_i, post_trans, channel, device, beta_range=beta_range,
                                                            max_search_steps=max_search_steps, search_threshold=search_threshold)
            betas.append(beta_tuple)
        beta_neg_scores = torch.tensor([b[0] for b in betas])
        beta_pos_scores = torch.tensor([b[1] for b in betas])

    # --- 2. Calibration ---
    beta_q, beta_q_pos, beta_q_neg = 0, 0, 0
    if weights is not None:
        # print(f"Using WCP for {'Symmetric' if sym else 'Asymmetric'} LogitsCRC ({weighting_method})...")
        if sym:
            beta_q = _compute_weighted_quantile(scores.numpy(), weights, 1 - alpha)
        else:
            beta_q_pos = _compute_weighted_quantile(beta_pos_scores.numpy(), weights, 1 - alpha / 2.0)
            beta_q_neg = _compute_weighted_quantile(beta_neg_scores.numpy(), weights, 1 - alpha / 2.0)
    else:
        # print(f"Using standard quantile for {'Symmetric' if sym else 'Asymmetric'} LogitsCRC...")
        n = len(va_vs)
        if sym:
            k = math.ceil((1 - alpha) * (n + 1))
            q_level = min(k / (n + 1), 1.0)
            beta_q = torch.quantile(scores, q_level, interpolation='higher')
        else:
            k_hi = math.ceil((1 - alpha / 2.0) * (n + 1))
            q_hi = min(k_hi / (n + 1), 1.0)
            beta_q_pos = torch.quantile(beta_pos_scores, q_hi, interpolation='higher')
            
            k_lo = math.ceil((1 - alpha / 2.0) * (n + 1))
            q_lo = min(k_lo / (n + 1), 1.0)
            beta_q_neg = torch.quantile(beta_neg_scores, q_lo, interpolation='higher')
            
    # --- 3. Inference ---
    pred_vols_hi, pred_vols_lo = [], []
    with torch.no_grad():
        for i in range(len(ts_vs)):
            ts_yhat = load_var_instance(fnames, 'ts', 'yhat', i).unsqueeze(0).to(device)
            if sym:
                new_yhats = torch.vstack((ts_yhat + beta_q, ts_yhat - beta_q))
                _, new_vols = post_trans(new_yhats, channel)
                pred_vols_lo.append(min(new_vols[0].item(), new_vols[1].item()))
                pred_vols_hi.append(max(new_vols[0].item(), new_vols[1].item()))
            else: # Asymmetric
                yhat_hi = ts_yhat + beta_q_pos
                yhat_lo = ts_yhat - beta_q_neg
                _, vol_hi = post_trans(yhat_hi, channel)
                _, vol_lo = post_trans(yhat_lo, channel)
                pred_vols_lo.append(vol_lo.item())
                pred_vols_hi.append(vol_hi.item())

    pred_vols_hi = torch.tensor(pred_vols_hi)
    pred_vols_lo = torch.tensor(pred_vols_lo)

    logits_len = (pred_vols_hi - pred_vols_lo).mean().item()
    logits_cov = (((ts_vs <= pred_vols_hi) & (ts_vs >= pred_vols_lo)).sum() / len(ts_vs)).item()
    
    mode_str = "Symmetric" if sym else "Asymmetric"
    print(f'{mode_str} COMPASS-L Len: {logits_len:.4f}, Cov: {logits_cov:.4f}')
    
    if return_scores:
        scores_to_return = scores if sym else (beta_neg_scores, beta_pos_scores)
        return logits_len, logits_cov, scores_to_return
    elif return_raw:
        return logits_len, logits_cov, np.stack([pred_vols_hi,pred_vols_lo,ts_vs])
    
    return logits_len, logits_cov

def Logits_Sweep(fnames, model, channel, post_trans, device, test_set,
                beta_range=1.0, beta_steps=20,
                plot_metric=True, plot_images=False, num_images_to_plot=4,
                num_betas_to_plot=5, return_plot_items=False, plot_indices=None):

    print("Loading logits for sweep...")
    ts_yhats=load_var(fnames,'ts','yhat')
    torch.cuda.empty_cache()
    
    pos_betas = np.linspace(0.0, beta_range, beta_steps)
    neg_betas = -pos_betas[1:][::-1]
    betas = np.concatenate([neg_betas, pos_betas])
    
    o_vols = np.zeros((len(betas), len(ts_yhats)))
    n_vols = np.zeros((len(betas), len(ts_yhats)))
    batch_size = 32

    with torch.no_grad():
        _, base_vols = post_trans(ts_yhats.to(device), channel)
        for i, beta in enumerate(betas):
            new_yhats = ts_yhats.to(device) + beta
            _, new_vol = post_trans(new_yhats, channel)
            o_vols[i, :] = base_vols.cpu().numpy()
            n_vols[i, :] = new_vol.cpu().numpy()

    metric_change_percent = np.divide((n_vols - o_vols), o_vols, out=np.zeros_like(n_vols), where=o_vols!=0) * 100

    if plot_metric:
        plt.figure(figsize=(8,6))
        plt.plot(betas, metric_change_percent)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Beta (β)')
        plt.ylabel('Change in Metric (%)')
        plt.title('Effect of COMPASS-L (logits) Perturbation on Metric')
        plt.ylim([-50, 50])
        plt.show()

    if plot_images:
        if test_set is None:
            raise ValueError("The 'test_set' dataset must be provided when plot_images is True.")
        
        print(f"Generating segmentation grid for {num_images_to_plot} images...")
        if plot_indices==None:
            plot_indices = random.sample(range(len(ts_yhats)), k=min(num_images_to_plot, len(ts_yhats)))
        plot_betas = np.linspace(-beta_range, beta_range, num_betas_to_plot)

        fig, axes = plt.subplots(len(plot_indices), num_betas_to_plot, figsize=(15, 3 * len(plot_indices)),
                                 gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
        
        for i, img_idx in enumerate(plot_indices):
            original_yhat = ts_yhats[img_idx].unsqueeze(0)
            item = test_set[img_idx]
            original_image_tensor = item['image'] if isinstance(item, dict) else item[0]
            
            for j, beta in enumerate(plot_betas):
                ax = axes[i, j] if len(plot_indices) > 1 else axes[j]
                
                new_yhat = original_yhat + beta
                seg, _ = post_trans(new_yhat.to(device), channel)
                seg = seg.cpu()
                
                img_to_plot = original_image_tensor.permute(1, 2, 0).numpy()
                if img_to_plot.shape[2] == 1:
                    img_to_plot = img_to_plot.squeeze()
                
                ax.imshow(img_to_plot, cmap='gray')
                mask = np.ma.masked_where(seg.squeeze() == 0, seg.squeeze())
                ax.imshow(mask, cmap='autumn', alpha=0.6)
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                if i == 0:
                    ax.set_title(f'β = {beta:.2f}', fontsize=12)
                if j == 0:
                    ax.set_ylabel(f'Sample {img_idx}', rotation=90, size='large', labelpad=20)
        
        # plt.suptitle('Segmentation Overlay vs. Logit Perturbation', fontsize=16, y=1.02)
        plt.tight_layout(pad=0.5)
        plt.show()

    if return_plot_items:
        # This part would need implementation to collect and return plot items
        return betas, metric_change_percent, None, None # Placeholder
    else:
        return betas, metric_change_percent

# =================================================================================
# ✅ NEW FUNCTIONS: Sweep by Delta V
# =================================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os

def LogitsSweep_by_DV(fnames, model, channel, post_trans, device, test_set,
                      delta_v_range_percent=50, delta_v_steps=20,
                      plot_metric=True, plot_images=False, num_images_to_plot=4,
                      num_deltas_to_plot=5, search_steps=30, beta_search_range=100.0,
                      plot_indices=None, return_plot_data=False, show_plots=True,show_images=False):
    """
    Finds the required `beta` to achieve a target `delta_v_percent` by perturbing logits,
    with full visualization and data return capabilities.
    """
    def _sort_fnames_numerically(file_list):
        return sorted(file_list, key=lambda f: int(os.path.basename(f).split('_')[-1].replace('.pt', '')))

    print("Loading logits for sweep by ∆V...")
    ts_yhat_fnames = _sort_fnames_numerically(fnames.get('ts', {}).get('yhat', []))
    if not ts_yhat_fnames:
        raise FileNotFoundError("Could not find 'yhat' files for the 'ts' split in fnames.")
        
    ts_yhats = torch.cat([torch.load(f) for f in ts_yhat_fnames], dim=0)
    torch.cuda.empty_cache()

    delta_v_targets = np.linspace(-delta_v_range_percent, delta_v_range_percent, delta_v_steps)
    found_betas = np.zeros((len(delta_v_targets), len(ts_yhats)))

    with torch.no_grad():
        _, base_vols = post_trans(ts_yhats.to(device), channel)
        base_vols = base_vols.cpu()

        # This loop calculates betas for ALL samples for the metric plot.
        for i in tqdm(range(len(ts_yhats)), desc="Searching Betas for each image"):
            if base_vols[i] == 0: continue
            
            for j, target_dv in enumerate(delta_v_targets):
                low, high = -beta_search_range, beta_search_range
                for _ in range(search_steps):
                    beta = (low + high) / 2
                    new_yhat = ts_yhats[i].unsqueeze(0).to(device) + beta
                    _, new_vol = post_trans(new_yhat, channel)
                    current_dv = (new_vol.item() - base_vols[i]) / base_vols[i] * 100
                    
                    if current_dv < target_dv: low = beta
                    else: high = beta
                found_betas[j, i] = (low + high) / 2

    if plot_metric:
        plt.figure(figsize=(8, 6))
        plt.plot(delta_v_targets, found_betas)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('δA (%)')
        plt.ylabel('Found Beta (β)')
        plt.title('Beta required to achieve Target ∆V (COMPASS-L / logits)')
        if show_plots:
            plt.show()

    stacked_images, stacked_segs = None, None

    if plot_images:
        if test_set is None:
            raise ValueError("The 'test_set' dataset must be provided when plot_images is True.")
        
        if plot_indices is None:
            plot_indices = random.sample(range(len(ts_yhats)), k=min(num_images_to_plot, len(ts_yhats)))
        
        print(f"Generating segmentation grid for images at indices: {plot_indices}...")
        
        plot_delta_indices = np.linspace(0, len(delta_v_targets) - 1, num_deltas_to_plot, dtype=int)
        
        fig, axes = plt.subplots(len(plot_indices), num_deltas_to_plot + 1, figsize=(15, 3 * len(plot_indices)-3),gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

        plotted_images_list = []
        plotted_segs_list = []

        for i, img_idx in enumerate(plot_indices):
            ax_row = axes[i] if len(plot_indices) > 1 else axes
            
            original_yhat = ts_yhats[img_idx].unsqueeze(0)
            item = test_set[img_idx]
            original_image_tensor = item[0]
            original_seg_tensor = item[1]
            plotted_images_list.append(original_image_tensor)

            # Display Ground Truth
            ax_row[-1].imshow(original_image_tensor.permute(1, 2, 0).numpy(), cmap='gray')
            # ✅ FIX: Use contour on the raw, unmasked tensor to draw the boundary
            ax_row[-1].contour(original_seg_tensor.squeeze(), levels=[0.5], colors='red', linewidths=1.5)
            ax_row[-1].set_xticks([]); ax_row[-1].set_yticks([])
            if i == 0: ax_row[-1].set_title('Ground Truth')

            segs_for_this_image = []
            for j, delta_idx in enumerate(plot_delta_indices):
                ax = ax_row[j]
                
                beta = found_betas[delta_idx, img_idx]
                target_dv = delta_v_targets[delta_idx]
                
                new_yhat = (original_yhat + beta).to(device)
                seg, _ = post_trans(new_yhat, channel)
                seg = seg.cpu()
                segs_for_this_image.append(seg.squeeze(0))

                img_to_plot = original_image_tensor.permute(1, 2, 0).numpy()
                ax.imshow(img_to_plot, cmap='gray')
                # ✅ FIX: Use contour on the raw segmentation, not a masked array
                ax.contour(seg.squeeze(), levels=[0.5], colors='red', linewidths=1.5)
                
                beta_text = f"β = {beta:.2f}"
                ax.text(0.95, 0.05, beta_text, ha='right', va='bottom', transform=ax.transAxes,
                        color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
                
                ax.set_xticks([]); ax.set_yticks([])
                
                if i == 0:
                    ax.set_title(f'δA = {target_dv:.0f}%')
                # if j == 0:
                #     ax.set_ylabel(f'Sample {img_idx}', rotation=90, size='large', labelpad=20)
            
            plotted_segs_list.append(torch.stack(segs_for_this_image))

        plt.tight_layout(pad=0.5)
        if show_images:
            plt.show()
        else:
            plt.close()

        if return_plot_data:
            stacked_images = torch.stack(plotted_images_list)
            stacked_segs = torch.stack(plotted_segs_list)

    if return_plot_data:
        return delta_v_targets, found_betas, stacked_images, stacked_segs
    else:
        return delta_v_targets, found_betas
