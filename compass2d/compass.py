from sklearn.decomposition import PCA
from .data_utils import *
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import datetime



import torch
import numpy as np
import math
import torchvision.transforms.functional as TF
import torch.fft
from sklearn.decomposition import PCA

def COMPASS(
    fnames, 
    model, 
    channel, 
    forward_func, 
    post_trans, 
    device,
    method='pca',          # Options: 'pca', 'uniform', 'gaussian', 'fourier'
    logits=False,
    n_components=1,        # For 'pca'
    sigma=2.0,             # For 'gaussian'
    kernel_size=9,         # For 'gaussian'
    bandwidth=0.1,          # For 'fourier'
    alpha=0.1, 
    beta_range=1.0, 
    max_search_steps=1000, 
    search_threshold=1e-5,
    sym=True, 
    weights=None,
    return_scores=False,
    return_raw=False
):
    """
    Unified COMPASS implementation handling Training, Calibration, and Inference.
    """
    
    # --- 1. Setup & Training (if required) ---
    # print(f"Initializing COMPASS ({method})...")
    reduce_model = None
    
    if 'pca' in method:
        # Load Training Jacobians for PCA fitting
        # Assumes 'tr_J' exists and is flattened or aggregated per channel
        # print("Fitting PCA on Training Jacobians...")
        tr_Js = load_var(fnames, 'tr', 'J')
        if hasattr(tr_Js, 'cpu'): tr_Js = tr_Js.cpu().numpy()
        
        # Fit PCA
        reduce_model = PCA(n_components=n_components)
        reduce_model.fit(tr_Js)
        del tr_Js
        torch.cuda.empty_cache()
    elif 'mean' in method:
        # Load Training Jacobians for PCA fitting
        # Assumes 'tr_J' exists and is flattened or aggregated per channel
        tr_Js = load_var(fnames, 'tr', 'J')
        tr_Js=tr_Js.mean(0)
        reduce_model=tr_Js/tr_Js.norm(2)
        del tr_Js
        torch.cuda.empty_cache()
    elif 'median' in method:
        tr_Js = load_var(fnames, 'tr', 'J') # (N, C)
        # Calculate median per channel
        median_J = torch.median(tr_Js.float(),dim=0)[0]
        reduce_model = median_J/median_J.norm(2) # L-Inf norm
        del tr_Js
        torch.cuda.empty_cache()
    elif 'sign' in method:
        tr_Js = load_var(fnames, 'tr', 'J')
        # -1 for negative gradient, +1 for positive
        signs = torch.sign(tr_Js) 
        # Average the votes
        vote = signs.mean(0)
        # Final direction is the sign of the majority vote
        reduce_model=torch.sign(vote).float().to(device)

    # Define the specific Delta function based on method
    delta_fn = _get_delta_function(method, reduce_model, device, sigma, kernel_size, bandwidth)

    # --- 2. Calibration ---
    # print("Starting Calibration...")
    
    # Load Validation Ground Truth
    va_vs = load_var(fnames, 'va', 'v').ravel()
    if hasattr(va_vs, 'cpu'): va_vs = va_vs.cpu().numpy()
    
    calib_scores_sym = []
    calib_scores_neg = []
    calib_scores_pos = []
    
    with torch.no_grad():
        for i in range(len(va_vs)):
            # Load Latent (z)
            if logits:
                z_i = load_var_instance(fnames, 'va', 'yhat', i).to(device)
            else:
                z_i = load_var_instance(fnames, 'va', 'latent', i).to(device)
            v_i = va_vs[i]
            
            # Compute Delta
            dl_i = delta_fn(i, z_i, 'va', fnames)
            
            # Find Score
            if sym:
                score = _generic_binary_search(
                    z_i, dl_i, v_i, model, forward_func, post_trans, channel,
                    beta_range, max_search_steps, search_threshold, mode='symmetric'
                )
                calib_scores_sym.append(score)
            else:
                beta_lo, beta_hi = _generic_binary_search(
                    z_i, dl_i, v_i, model, forward_func, post_trans, channel,
                    beta_range, max_search_steps, search_threshold, mode='asymmetric'
                )
                calib_scores_neg.append(beta_lo) # Lower bound beta
                calib_scores_pos.append(beta_hi) # Upper bound beta

    # Compute Calibrated Beta(s)
    n = len(va_vs)
    beta_hat, beta_q_pos, beta_q_neg = 0, 0, 0
    
    if weights is not None:
        # Weighted CP Logic
        target_coverage = 1 - alpha
        if sym:
            beta_hat = _compute_weighted_quantile(np.array(calib_scores_sym), weights, target_coverage)
        else:
            # Asymmetric WCP
            beta_q_pos = _compute_weighted_quantile(np.array(calib_scores_pos), weights, 1 - alpha/2)
            beta_q_neg = _compute_weighted_quantile(np.array(calib_scores_neg), weights, 1 - alpha/2)
    else:
        # Standard CP Logic (Finite Sample Correction)
        # Using n in denominator for standard implementation
        if sym:
            q = math.ceil((n + 1) * (1 - alpha)) / n
            beta_hat = np.quantile(calib_scores_sym, min(q, 1.0), method='higher')
        else:
            # Both upper and lower bounds need high quantile of their respective error scores
            q_asym = math.ceil((n + 1) * (1 - alpha/2)) / n
            q_asym = min(q_asym, 1.0)
            beta_q_pos = np.quantile(calib_scores_pos, q_asym, method='higher')
            beta_q_neg = np.quantile(calib_scores_neg, q_asym, method='higher')

    # --- 3. Inference (Testing) ---
    # print("Starting Inference...")
    
    # Load Test Ground Truth for Evaluation
    ts_vs = load_var(fnames, 'ts', 'v').ravel()
    if hasattr(ts_vs, 'cpu'): ts_vs = ts_vs.cpu().numpy()
    
    pred_hi, pred_lo = [], []
    
    with torch.no_grad():
        for i in range(len(ts_vs)):
            if logits:
                z_i = load_var_instance(fnames, 'ts', 'yhat', i).to(device)
            else:
                z_i = load_var_instance(fnames, 'ts', 'latent', i).to(device)
            
            # Compute Delta
            dl_i = delta_fn(i, z_i, 'ts', fnames)
            
            # Apply Calibrated Beta
            if sym:
                # Endpoint-COMPASS Construction
                latents = torch.stack([
                    z_i - beta_hat * dl_i,
                    z_i + beta_hat * dl_i
                ])
            else:
                # Asymmetric Construction
                latents = torch.stack([
                    z_i - beta_q_neg * dl_i, # Lower beta pushes down
                    z_i + beta_q_pos * dl_i  # Upper beta pushes up
                ])
            
            outputs = forward_func(model, latents.float())
            _, metrics = post_trans(outputs, channel)
            
            pred_lo.append(min(metrics[0].item(), metrics[1].item()))
            pred_hi.append(max(metrics[0].item(), metrics[1].item()))
            
    pred_hi = np.array(pred_hi)
    pred_lo = np.array(pred_lo)
    
    # --- 4. Metrics & Returns ---
    mean_len = (pred_hi - pred_lo).mean().item()
    coverage = cover(pred_lo, pred_hi, ts_vs) # Assuming cover() is defined in env
    
    # print(f"{'Symmetric' if sym else 'Asymmetric'} COMPASS ({method}) | Len: {mean_len:.4f} | Cov: {coverage:.4f}")
    
    if return_scores:
        return mean_len, coverage, (calib_scores_sym if sym else (calib_scores_neg, calib_scores_pos))
    elif return_raw:
        return mean_len, coverage, np.stack([pred_hi, pred_lo, ts_vs])
    
    return mean_len, coverage

# -------------------------------------------------------------------------
# Delta Function Factory & Implementations
# -------------------------------------------------------------------------

def _get_delta_function(method, reduce_model, device, sigma, kernel_size, bandwidth):
    """Factory to create the specific delta function closure."""
    
    if method == 'pca':
        def delta_pca(index, z, split, fnames):
            # Standard COMPASS-J
            J = load_var_instance(fnames, split, 'J', index) 
            J = J.reshape(1, -1) # (1, C)
            
            # Project uncentered vector onto components
            coeffs = J @ reduce_model.components_.T
            
            # Reconstruct: inverse_transform adds the mean back
            recon = reduce_model.inverse_transform(coeffs)
            
            # Normalize (L2 over the vector)
            # recon/=recon.norm(p=torch.inf,dim=1,keepdim=True)
            # recon/=recon.norm(p=2,dim=1,keepdim=True)
            recon/=recon.std(dim=1,keepdim=True)
            delta = recon.float().to(device)
            
            if z.dim() == 3:
                return delta.view(-1, 1, 1)
            else:
                return delta.view_as(z)
        return delta_pca

    elif method=='pca_spatial':
        def delta_pca_spatial(index, z, split, fnames):
            # "Projected Spatial" Method
            # 1. Get Global Semantic Vector (v) from PCA (1st component)
            v = torch.from_numpy(reduce_model.components_[0]).float().to(device) # (C,)
            
            # 2. Load Full Spatial Jacobian (C, H, W)
            J = load_var_instance(fnames, split, 'Jfull', index)
            J = J.float().to(device)
            if J.dim() == 3: J = J.unsqueeze(0) # (1, C, H, W)
            
            # 3. Compute Alignment Map
            # Dot product of local gradient with global vector
            v_broadcast = v.view(1, -1, 1, 1)
            alignment_map = (J * v_broadcast).sum(dim=1, keepdim=True) # (1, 1, H, W)
            
            # 4. Rectify (Only push pixels that agree with global direction)
            alignment_map = torch.nn.functional.relu(alignment_map)
            
            # 5. Construct Delta
            delta = v_broadcast * alignment_map
            
            # 6. Normalize
            scale = delta.abs().max()
            if scale > 1e-9: delta = delta / scale
                
            return delta.view_as(z)
        return delta_pca_spatial

    elif method == 'pca_activation_weighted':
        def delta_activation(index, z, split, fnames):
            # "Activation-Weighted" Method
            # 1. Global PCA Vector
            v = torch.from_numpy(reduce_model.components_[0]).float().to(device)
            v = v.view(1, -1, 1, 1)
            
            # 2. Use feature magnitude as spatial mask
            # z is already loaded and passed to this function
            spatial_weight = z.abs()
            
            # 3. Modulate
            delta = v * spatial_weight
            
            # 4. Normalize
            scale = delta.abs().max()
            if scale > 1e-9: delta = delta / scale
                
            return delta.view_as(z)
        return delta_activation

    elif method == 'pca_uncertainty_weighted':
        def delta_uncertainty(index, z, split, fnames):
            # "Uncertainty-Weighted" Feature Perturbation
            # 1. Global PCA Vector
            v = torch.from_numpy(reduce_model.components_[0]).float().to(device)
            v = v.view(1, -1, 1, 1)
            
            # 2. Calculate Entropy from Logits
            yhat = load_var_instance(fnames, split, 'yhat', index).to(device)
            if yhat.dim() == 3: yhat = yhat.unsqueeze(0)
            
            probs = torch.sigmoid(yhat)
            entropy = - (probs * torch.log(probs + 1e-6) + (1-probs) * torch.log(1-probs + 1e-6))
            
            # 3. Upsample Entropy to Feature Resolution if needed
            if z.shape[-1] != entropy.shape[-1]:
                entropy = torch.nn.functional.interpolate(entropy, size=z.shape[-2:], mode='bilinear')
            
            # 4. Modulate
            delta = v * entropy
            
            # 5. Normalize
            scale = delta.abs().max()
            if scale > 1e-9: delta = delta / scale
                
            return delta.view_as(z)
        return delta_uncertainty
    
    
    elif method=='mean':
        def delta_mean(index, z, split, fnames):
            delta = reduce_model.to(device)
            if z.dim() == 3:
                return delta.view(-1, 1, 1)
            else:
                return delta.view_as(z)
        return delta_mean

    elif method=='median':
        def delta_median(index, z, split, fnames):
            delta = reduce_model.to(device)
            if z.dim() == 3:
                return delta.view(-1, 1, 1)
            else:
                return delta.view_as(z)
        return delta_median
    
    elif method == 'uniform':
        def delta_ones(index, z, split, fnames):
            return torch.ones_like(z)
        return delta_ones

    elif method == 'jacobian':
        def delta_jacobian(index, z, split, fnames):
            # Naive Jacobian Perturbation with Fixes for NaNs
            J = load_var_instance(fnames, split, 'J', index)
            # J = J.flatten(1).sum(1).float().to(device)
            J=J.float().to(device)
            delta = J/J.norm(p=torch.inf)
            return delta.expand_as(z)
        return delta_jacobian

    elif method == 'gaussian':
        def delta_gaussian(index, z, split, fnames):
            J = load_var_instance(fnames, split, 'Jfull', index)
            J = J.float().to(device)
            if J.dim() == 3: J = J.unsqueeze(0)
            
            J_smooth = TF.gaussian_blur(J, kernel_size=kernel_size, sigma=sigma)
            
            scale = J_smooth.abs().max()
            delta = (J_smooth / scale) if scale > 0 else J_smooth
            return delta.view_as(z)
        return delta_gaussian

    elif method == 'fourier':
        def delta_fourier(index, z, split, fnames):
            J = load_var_instance(fnames, split, 'Jfull', index)
            J = J.float().to(device)
            if J.dim() == 3: J = J.unsqueeze(0)
            
            freq = torch.fft.fft2(J)
            freq_shift = torch.fft.fftshift(freq)
            
            _, _, H, W = J.shape
            Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            center_y, center_x = H // 2, W // 2
            dist = torch.sqrt((X - center_x)**2 + (Y - center_y)**2).to(device)
            mask = (dist <= bandwidth).float().unsqueeze(0).unsqueeze(0)
            
            freq_filtered = freq_shift * mask
            freq_unshift = torch.fft.ifftshift(freq_filtered)
            J_smooth = torch.fft.ifft2(freq_unshift).real
            
            scale = J_smooth.abs().max()
            delta = (J_smooth / scale) if scale > 0 else J_smooth
            return delta.view_as(z)
        return delta_fourier

    else:
        raise ValueError(f"Unknown COMPASS method: {method}")

# -------------------------------------------------------------------------
# Helper Functions (Binary Search & Quantile)
# -------------------------------------------------------------------------

def _generic_binary_search(z, delta, target, model, forward_func, post_trans, channel, 
                           beta_range, max_steps, threshold, mode='symmetric'):
    """
    Performs binary search to find the minimal beta that covers the target.
    """
    def evaluate_beta(b_lo, b_hi):
        # b_lo is negative magnitude, b_hi is positive magnitude
        latents = torch.stack([
            z - b_lo * delta, 
            z + b_hi * delta
        ])
        outputs = forward_func(model, latents.float()) 
        _, metrics = post_trans(outputs, channel)
        return metrics[0].item(), metrics[1].item() # (val_lo, val_hi)

    if mode == 'symmetric':
        low, high = 0.0, beta_range
        final_beta = high
        
        for _ in range(max_steps):
            if (high - low) < threshold: break
            mid = (low + high) / 2.0
            val_lo, val_hi = evaluate_beta(mid, mid)
            pred_min, pred_max = min(val_lo, val_hi), max(val_lo, val_hi)
            
            if pred_min <= target <= pred_max:
                high = mid
                final_beta = mid
            else:
                low = mid
        return final_beta

    elif mode == 'asymmetric':
        # Upper Bound Search (Positive Beta)
        low, high = 0.0, beta_range
        final_beta_hi = high
        for _ in range(max_steps):
            if (high - low) < threshold: break
            mid = (low + high) / 2.0
            _, val_hi = evaluate_beta(0.0, mid)
            if val_hi >= target: # Assuming monotonic increasing
                high = mid
                final_beta_hi = mid
            else:
                low = mid
                
        # Lower Bound Search (Negative Beta)
        low, high = 0.0, beta_range
        final_beta_lo = high
        for _ in range(max_steps):
            if (high - low) < threshold: break
            mid = (low + high) / 2.0
            val_lo, _ = evaluate_beta(mid, 0.0)
            if val_lo <= target: # Assuming monotonic increasing
                high = mid
                final_beta_lo = mid
            else:
                low = mid
        return final_beta_lo, final_beta_hi

def _compute_weighted_quantile(scores, weights, q):
    # Placeholder for WCP logic
    return np.quantile(scores, q, method='higher')























def _beta_binary_search_symmetric(latent_i, dl_i, v_i, model, forward_latent, post_trans, channel, beta_range, max_search_steps=1000, search_threshold=1e-10):
    """
    Finds the non-conformity score (beta) for a single sample for the symmetric case.
    The score is the smallest beta that creates a covering interval.
    """
    low, high = 0.0, beta_range
    final_beta = high  # Default to the max range if search fails
    with torch.no_grad():
        for _ in range(max_search_steps):
            # Check for precision-based stopping condition
            if (high - low) < search_threshold:
                break
                
            mid = (low + high) / 2
            
            # Perturb in both positive and negative directions
            new_latents = torch.vstack((
                latent_i.unsqueeze(0) - mid * dl_i.unsqueeze(0),
                latent_i.unsqueeze(0) + mid * dl_i.unsqueeze(0)
            ))
            new_yhats = forward_latent(model, new_latents.float())
            _, new_vols = post_trans(new_yhats, channel)
            
            lower_bound = min(new_vols[0].item(), new_vols[1].item())
            upper_bound = max(new_vols[0].item(), new_vols[1].item())
    
            # Check for coverage
            if lower_bound <= v_i.item() <= upper_bound:
                # If covered, we might be able to use a smaller beta
                high = mid
                final_beta = mid
            else:
                # If not covered, we need a larger beta
                low = mid
            
    return final_beta
    
def _beta_binary_search_asymmetric(latent_i, dl_i, v_i, model, forward_latent, post_trans, channel, beta_range,search_threshold=1e-8, max_search_steps=1000):
    """
    Finds the non-conformity scores (betas) for a single sample using a binary search
    that stops when the predicted volume is within a threshold of the true volume.
    """
    # --- Search for the positive beta (upper bound) ---
    low_pos, high_pos = 0.0, beta_range
    final_beta_pos = high_pos

    with torch.no_grad():
        for _ in range(max_search_steps):
            mid_pos = (low_pos + high_pos) / 2
            new_latent = latent_i.unsqueeze(0) + mid_pos * dl_i.unsqueeze(0)
            new_yhat = forward_latent(model, new_latent.float())
            _, new_vol = post_trans(new_yhat, channel)
            
            if abs(new_vol.item() - v_i.item()) < search_threshold:
                final_beta_pos = mid_pos
                break
            if new_vol.item() > v_i.item():
                high_pos = mid_pos
            else:
                low_pos = mid_pos
            if (high_pos - low_pos) < 1e-7:
                final_beta_pos = high_pos
                break
        
        # --- Search for the negative beta (lower bound) ---
        low_neg, high_neg = -beta_range, 0.0
        final_beta_neg = low_neg
    
        for _ in range(max_search_steps):
            mid_neg = (low_neg + high_neg) / 2
            new_latent = latent_i.unsqueeze(0) + mid_neg * dl_i.unsqueeze(0)
            new_yhat = forward_latent(model, new_latent.float())
            _, new_vol = post_trans(new_yhat, channel)
    
            if abs(new_vol.item() - v_i.item()) < search_threshold:
                final_beta_neg = mid_neg
                break
            if new_vol.item() > v_i.item():  # Prediction is too high, beta is not negative enough
                high_neg = mid_neg
            else:  # Prediction is too low, beta is too negative
                low_neg = mid_neg
            if (high_neg - low_neg) < 1e-7:
                final_beta_neg = low_neg
                break

    return (final_beta_pos, final_beta_neg)

def COMPASSJ(fnames, model, channel, forward_latent, post_trans, device,
             beta_range=1.0, n_components=1, alpha=0.1,
             search_threshold=1e-8, max_search_steps=1000,
             sym=True, weights=None, return_scores=False, return_raw=False, return_calib=False):
    """
    Unified (a)symmetric COMPASS with optional WCP, including weight stabilization.
    """
    # --- 1. Setup and Score Calculation ---
    torch.cuda.empty_cache()
    tr_Js = load_var(fnames, 'tr', 'J')
    if hasattr(tr_Js, 'cpu'): tr_Js = tr_Js.cpu().numpy()
    
    pca = PCA(n_components)
    pca.fit(tr_Js)
    del tr_Js

    def project_reconstruct(Js,pca,method='standard',lambda_mean=1.0):
        if method=='lambda':
            mean_direction = pca.mean_
            projected_centered = pca.transform(Js) 
            variance_direction = pca.inverse_transform(projected_centered)
            dl_np = variance_direction + (lambda_mean * mean_direction)
            dl_np = dl_np.reshape(dl_np.shape[0], dl_np.shape[1], 1, 1)
            dl_np /= np.linalg.norm(dl_np, ord=2, axis=1, keepdims=True)
            dl = torch.from_numpy(dl_np).float().to(device)
        else:
            dl_np = pca.inverse_transform((Js @ pca.components_.T))
            dl_np = dl_np.reshape(dl_np.shape[0], dl_np.shape[1], 1, 1)
            dl_np /= np.linalg.norm(dl_np, ord=2, axis=1, keepdims=True)
            dl = torch.from_numpy(dl_np).float().to(device)
        return dl
    
    va_Js = load_var(fnames, 'va', 'J')
    if hasattr(va_Js, 'cpu'): va_Js = va_Js.cpu().numpy()
    # dl_np = pca.inverse_transform((va_Js @ pca.components_.T))
    # dl_np = dl_np.reshape(dl_np.shape[0], dl_np.shape[1], 1, 1)
    # dl_np /= np.linalg.norm(dl_np, ord=2, axis=1, keepdims=True)
    # dl = torch.from_numpy(dl_np).float().to(device)
    dl=project_reconstruct(va_Js,pca)
    del va_Js
    
    va_vs = load_var(fnames, 'va', 'v')
    if hasattr(va_vs, 'cpu'): va_vs = va_vs.cpu().numpy()
    va_vs = va_vs.ravel()
    
    scores, beta_pos_scores, beta_neg_scores = None, None, None
    with torch.no_grad():
        if sym:
            scores = []
            for i in range(len(va_vs)):
                va_latent = load_var_instance(fnames, 'va', 'latent', i)
                score = _beta_binary_search_symmetric(
                    va_latent.to(device), dl[i], torch.tensor(va_vs[i]).to(device), model,
                    forward_latent, post_trans, channel, beta_range,
                    max_search_steps=max_search_steps, search_threshold=search_threshold
                )
                scores.append(score)
            scores = np.array(scores)
        else: # Asymmetric
            betas = []
            for i in range(len(va_vs)):
                va_latent = load_var_instance(fnames, 'va', 'latent', i)
                beta_tuple = _beta_binary_search_asymmetric(
                    va_latent.to(device), dl[i], torch.tensor(va_vs[i]).to(device), model,
                    forward_latent, post_trans, channel, beta_range,
                    max_search_steps=max_search_steps, search_threshold=search_threshold
                )
                betas.append(beta_tuple)
            beta_pos_scores = np.array([b[0] for b in betas])
            beta_neg_scores = np.array([b[1] for b in betas])

    # --- 2. Calibration ---
    beta_hat, beta_q_pos, beta_q_neg = 0, 0, 0
    if weights is not None:
        if sym:
            beta_hat = _compute_weighted_quantile(scores, weights, 1 - alpha)
        else:
            beta_q_pos = _compute_weighted_quantile(beta_pos_scores, weights, 1 - alpha / 2.0)
            beta_q_neg = _compute_weighted_quantile(beta_neg_scores, weights, alpha / 2.0)
    else:
        # print(f"Using standard quantile for {'Symmetric' if sym else 'Asymmetric'} COMPASS...")
        n = len(va_vs)
        if sym:
            q_level = math.ceil((n + 1) * (1 - alpha)) / n
            beta_hat = np.quantile(scores, min(q_level, 1.0), method='higher')
        else:
            k_hi = math.ceil((1 - alpha / 2.0) * (n + 1))
            q_hi = min(k_hi / (n + 1), 1.0)
            beta_q_pos = np.quantile(beta_pos_scores, q_hi, method='higher')
            
            k_lo = math.floor((alpha / 2.0) * (n + 1))
            q_lo = k_lo / (n + 1)
            beta_q_neg = np.quantile(beta_neg_scores, q_lo, method='lower')
   
    # pred_vols_hi, pred_vols_lo = [], []
    # with torch.no_grad():
    #     for i in range(len(va_vs)):
    #         va_latent = load_var_instance(fnames, 'va', 'latent', i)
    #         if sym:
    #             new_latents = torch.vstack((
    #                 va_latent.to(device) - beta_hat * dl[i].unsqueeze(0),
    #                 va_latent.to(device) + beta_hat * dl[i].unsqueeze(0)
    #             ))
    #         else: # Asymmetric
    #             new_latents = torch.vstack((
    #                 va_latent.to(device) + beta_q_pos * dl[i].unsqueeze(0),
    #                 va_latent.to(device) + beta_q_neg * dl[i].unsqueeze(0)
    #             ))

    #         _, new_vols = post_trans(forward_latent(model, new_latents.float()), channel)
            
    #         pred_vols_lo.append(min(new_vols[0].item(), new_vols[1].item()))
    #         pred_vols_hi.append(max(new_vols[0].item(), new_vols[1].item()))

    # pred_vols_hi = np.array(pred_vols_hi)
    # pred_vols_lo = np.array(pred_vols_lo)
    
    # va_compass_len = (pred_vols_hi - pred_vols_lo).mean().item()
    # va_compass_cov = cover(pred_vols_lo, pred_vols_hi, va_vs)
    # del va_latent,va_vs,new_latents,pred_vols_hi,pred_vols_lo

    # --- 3. Inference ---
    ts_Js = load_var(fnames, 'ts', 'J')
    if hasattr(ts_Js, 'cpu'): ts_Js = ts_Js.cpu().numpy()
    
    # dl_np = pca.inverse_transform((ts_Js @ pca.components_.T))
    # dl_np = dl_np.reshape(dl_np.shape[0], dl_np.shape[1], 1, 1)
    # dl_np /= np.linalg.norm(dl_np, ord=2, axis=1, keepdims=True)
    # dl = torch.from_numpy(dl_np).float().to(device)
    dl=project_reconstruct(ts_Js,pca)
    del ts_Js
    
    ts_vs = load_var(fnames, 'ts', 'v')
    if hasattr(ts_vs, 'cpu'): ts_vs = ts_vs.cpu().numpy()
    ts_vs = ts_vs.ravel()
    
    pred_vols_hi, pred_vols_lo = [], []
    with torch.no_grad():
        for i in range(len(ts_vs)):
            ts_latent = load_var_instance(fnames, 'ts', 'latent', i)
            if sym:
                new_latents = torch.vstack((
                    ts_latent.to(device) - beta_hat * dl[i].unsqueeze(0),
                    ts_latent.to(device) + beta_hat * dl[i].unsqueeze(0)
                ))
            else: # Asymmetric
                new_latents = torch.vstack((
                    ts_latent.to(device) + beta_q_pos * dl[i].unsqueeze(0),
                    ts_latent.to(device) + beta_q_neg * dl[i].unsqueeze(0)
                ))

            _, new_vols = post_trans(forward_latent(model, new_latents.float()), channel)
            
            pred_vols_lo.append(min(new_vols[0].item(), new_vols[1].item()))
            pred_vols_hi.append(max(new_vols[0].item(), new_vols[1].item()))

    pred_vols_hi = np.array(pred_vols_hi)
    pred_vols_lo = np.array(pred_vols_lo)
    
    compass_len = (pred_vols_hi - pred_vols_lo).mean().item()
    compass_cov = cover(pred_vols_lo, pred_vols_hi, ts_vs)
    
    if return_scores:
        return compass_len, compass_cov, scores if sym else (beta_pos_scores, beta_neg_scores)
    elif return_raw:
        return compass_len, compass_cov, np.stack([pred_vols_hi,pred_vols_lo,ts_vs])
    elif return_calib:
        return compass_len, compass_cov, va_compass_len, va_compass_cov
    return compass_len, compass_cov


def _compute_weighted_quantile(scores, weights, quantile_level):
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

def cover(lower, upper, true): 
    return np.mean((true >= lower) & (true <= upper))

def COMPASS_Sweep(fnames, model, channel, forward_latent, post_trans, device, test_set,
              beta_range=1.0, beta_steps=20, n_components=1, 
              plot_metric=True, plot_images=False, 
              num_images_to_plot=4,num_betas_to_plot=5,return_plot_items=False,plot_indices=None):
    """
    Performs a beta sweep along the principal components of the Jacobian,
    calculates the effect on a given metric, and optionally plots visualizations.
    """

    # --- 1. Compute PCA on the Jacobian ---
    # print("Loading Jacobians and fitting PCA...")
    tr_Js = load_var(fnames,'tr','J')
    ts_Js = load_var(fnames,'ts','J')
    ts_latents = load_var(fnames,'ts','latent')
    ts_vs = load_var(fnames,'ts','v')
    
    pca = PCA(n_components=n_components)
    pca.fit(tr_Js)
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    torch.cuda.empty_cache()
    
    # --- 2. Perform Beta Sweep for Metric Calculation ---
    # print("Performing beta sweep for metric calculation...")
    pos_betas = np.linspace(0.0, beta_range, beta_steps)
    neg_betas = -pos_betas[1:][::-1]  # Avoid duplicating 0
    betas = np.concatenate([neg_betas, pos_betas])  # Full symmetric set
    
    o_vols = np.zeros((len(betas), len(ts_Js)))
    n_vols = np.zeros((len(betas), len(ts_Js)))
    batch_size = 32  # Adjust batch size based on GPU memory

    for ai, beta in enumerate(betas):
        transformed_Js = torch.tensor(ts_Js @ pca.components_.T)
        dl = torch.tensor(pca.inverse_transform(transformed_Js)).unsqueeze(-1).unsqueeze(-1)
        dl /= dl.norm(p=2, dim=1, keepdim=True)
        
        new_latents = ts_latents + beta * dl
        
        for li in range(0, new_latents.shape[0], batch_size):
            o_yhat = forward_latent(model, ts_latents[li:li+batch_size].float().to(device)).detach().cpu()
            _, o_vol = post_trans(o_yhat, channel)
            
            n_yhat = forward_latent(model, new_latents[li:li+batch_size].float().to(device)).detach().cpu()
            _, n_vol = post_trans(n_yhat, channel)
            
            o_vols[ai, li:li+batch_size] = o_vol.numpy()
            n_vols[ai, li:li+batch_size] = n_vol.numpy()

    metric_change_percent = (n_vols - o_vols) / o_vols * 100

    # --- 3. Plot Metric vs. Beta Curve ---
    if plot_metric:
        plt.figure(figsize=(7, 6))
        plt.plot(betas, metric_change_percent)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('β')
        plt.ylabel('Change in Metric (%)')
        # plt.title(f'Effect of Latent Perturbation on Metric')
        plt.ylim([-50, 50])
        plt.show()

    # --- 4. Plot Segmentation Grid Visualization ---
    if plot_images:
        if test_set is None:
            raise ValueError("The 'test_set' dataset must be provided when plot_images is True.")
        
        print(f"Generating segmentation grid for {num_images_to_plot} images...")
        if plot_indices==None:
            plot_indices = random.sample(range(len(ts_latents)), k=min(num_images_to_plot, len(ts_latents)))
        plot_betas = np.linspace(-beta_range, beta_range, num_betas_to_plot)

        fig, axes = plt.subplots(len(plot_indices), num_betas_to_plot, figsize=(15, 3 * len(plot_indices)),gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
        return_plot_imgs=[]
        return_plot_segs=[]
        for i, img_idx in enumerate(plot_indices):
            latent = ts_latents[img_idx].unsqueeze(0)
            J = torch.tensor(ts_Js[img_idx]).unsqueeze(0)
            
            item = test_set[img_idx]
            original_image_tensor = item['image'] if isinstance(item, dict) else item[0]
            
            transformed_J = J @ pca.components_.T
            dl = torch.tensor(pca.inverse_transform(transformed_J)).unsqueeze(-1).unsqueeze(-1)
            dl /= dl.norm(p=2, dim=1, keepdim=True)
            return_plot_imgs_beta=[]
            return_plot_segs_beta=[]
            for j, beta in enumerate(plot_betas):
                ax = axes[i, j] if len(plot_indices) > 1 else axes[j]
                
                new_latent = latent + beta * dl
                yhat = forward_latent(model, new_latent.float().to(device)).detach().cpu()
                seg, _ = post_trans(yhat, channel)
                
                img_to_plot = original_image_tensor.permute(1, 2, 0).numpy()
                if img_to_plot.shape[2] == 1:
                    img_to_plot = img_to_plot.squeeze()

                return_plot_imgs_beta.append(img_to_plot)
                return_plot_segs_beta.append(seg.squeeze())
                
                ax.imshow(img_to_plot, cmap='gray')
                mask = np.ma.masked_where(seg.squeeze() == 0, seg.squeeze())
                # ax.imshow(mask, cmap='autumn', alpha=0.6)
                ax.contour(seg.squeeze(),levels=[0.5],colors='red',linewidths=1.5)
                ax.set_xticks([])
                ax.set_yticks([])
                
                if i == 0:
                    ax.set_title(f'β = {beta:.2f}', fontsize=12)
                if j == 0:
                    ax.set_ylabel(f'Sample {img_idx}', rotation=90, size='large')

            return_plot_imgs.append(np.stack(return_plot_imgs_beta))
            return_plot_segs.append(np.stack(return_plot_imgs_beta))
        
        # plt.suptitle('β Perturbation', fontsize=16, y=1.02)
        plt.show()

        return_plot_imgs=np.stack(return_plot_imgs)
        return_plot_segs=np.stack(return_plot_imgs)
    
    if return_plot_items:
        return betas, metric_change_percent,return_plot_imgs,return_plot_segs
    else:
        return betas, metric_change_percent

def COMPASS_Sweep_by_DV(fnames, model, channel, forward_latent, post_trans, device, test_set,
                        delta_v_range_percent=50, delta_v_steps=5, n_components=1,
                        plot_metric=True, plot_images=False, num_images_to_plot=4,
                        num_deltas_to_plot=5, search_steps=30, beta_search_range=100.0,
                        plot_indices=None, return_plot_data=False, show_images=False):
    """
    Finds the required `beta` to achieve a target `delta_v_percent` by perturbing the latent space.
    """

    print("Loading Jacobians and fitting PCA for sweep by ∆V...")
    tr_Js = load_var(fnames,'tr','J')
    ts_Js = load_var(fnames,'ts','J')
    ts_latents = load_var(fnames,'ts','latent')
    
    pca = PCA(n_components=n_components)
    pca.fit(tr_Js)
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    
    transformed_Js = torch.tensor(ts_Js @ pca.components_.T)
    dl = torch.tensor(pca.inverse_transform(transformed_Js)).unsqueeze(-1).unsqueeze(-1)
    dl /= dl.norm(p=2, dim=1, keepdim=True)
    
    delta_v_targets = np.linspace(-delta_v_range_percent, delta_v_range_percent, delta_v_steps)
    found_betas = np.zeros((len(delta_v_targets), len(ts_latents)))

    if plot_indices is None:
        plot_indices = random.sample(range(len(ts_latents)), k=min(num_images_to_plot, len(ts_latents)))
    
    with torch.no_grad():
        _, base_vols = post_trans(forward_latent(model, ts_latents.to(device)), channel)
        base_vols = base_vols.cpu()

        for i, img_idx in enumerate(plot_indices):
            if base_vols[img_idx] == 0: continue
            
            for j, target_dv in enumerate(delta_v_targets):
                low, high = -beta_search_range, beta_search_range
                for _ in range(search_steps):
                    beta = (low + high) / 2
                    new_latent = ts_latents[img_idx].unsqueeze(0) + beta * dl[img_idx].unsqueeze(0)
                    yhat = forward_latent(model, new_latent.float().to(device))
                    _, new_vol = post_trans(yhat, channel)
                    current_dv = (new_vol.item() - base_vols[img_idx]) / base_vols[img_idx] * 100
                    
                    if current_dv < target_dv: low = beta
                    else: high = beta
                found_betas[j, i] = (low + high) / 2
                
    if plot_metric:
        plt.figure(figsize=(8, 6))
        plt.plot(delta_v_targets, found_betas[:, :len(plot_indices)])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Target ∆V (%)')
        plt.ylabel('Found Beta (β)')
        plt.title(f'Beta required to achieve Target ∆V (COMPASS, {n_components} PC)')
        plt.show()

    # Initialize return variables for plot data
    stacked_images, stacked_segs = None, None

    if plot_images:
        if test_set is None:
            raise ValueError("The 'test_set' dataset must be provided when plot_images is True.")
        
        print(f"Generating segmentation grid for {num_images_to_plot} images...")
        
        plot_delta_indices = np.linspace(0, len(delta_v_targets) - 1, num_deltas_to_plot, dtype=int)
        
        fig, axes = plt.subplots(len(plot_indices), num_deltas_to_plot + 1, figsize=(15, 3 * len(plot_indices)-3),gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

        plotted_images_list = []
        plotted_segs_list = []

        for i, img_idx in enumerate(plot_indices):
            ax_row = axes[i] if len(plot_indices) > 1 else axes

            latent = ts_latents[img_idx].unsqueeze(0)
            item = test_set[img_idx]
            original_image_tensor = item[0]
            original_seg_tensor = item[1]
            plotted_images_list.append(original_image_tensor)

            # Display Ground Truth
            ax_row[-1].imshow(original_image_tensor.permute(1, 2, 0).numpy(), cmap='gray')
            mask = np.ma.masked_where(original_seg_tensor.squeeze() == 0, original_seg_tensor.squeeze())
            # ax_row[-1].imshow(mask, cmap='autumn_r', alpha=0.6)
            ax_row[-1].contour(original_seg_tensor.squeeze(),levels=[0.5],colors='red',linewidths=1.5)
            ax_row[-1].set_xticks([])
            ax_row[-1].set_yticks([])
            if i == 0:
                ax_row[-1].set_title('Ground Truth')
            
            perturbation_direction = dl[img_idx].unsqueeze(0)
            
            segs_for_this_image = []
            for j, delta_idx in enumerate(plot_delta_indices):
                ax = ax_row[j]
                
                beta = found_betas[delta_idx, i]
                target_dv = delta_v_targets[delta_idx]
                
                new_latent = latent + beta * perturbation_direction
                yhat = forward_latent(model, new_latent.float().to(device)).detach().cpu()
                seg, _ = post_trans(yhat, channel)
                segs_for_this_image.append(seg.squeeze(0)) # Store seg tensor

                img_to_plot = original_image_tensor.permute(1, 2, 0).numpy()
                
                ax.imshow(img_to_plot, cmap='gray')
                mask = np.ma.masked_where(seg.squeeze() == 0, seg.squeeze())
                # ax.imshow(mask, cmap='autumn', alpha=0.6)
                ax.contour(seg.squeeze(),levels=[0.5],colors='red',linewidths=1.5)
                
                beta_text = f"β = {beta:.2f}"
                ax.text(0.95, 0.05, beta_text,
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes, color='white', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                if i == 0:
                    ax.set_title(f'δA = {target_dv:.0f}%', fontsize=12)
                # if j == 0:
                #     ax.set_ylabel(f'Sample {img_idx}', rotation=90, size='large', labelpad=20)
            
            plotted_segs_list.append(torch.stack(segs_for_this_image))
        
        # plt.tight_layout(pad=0.5)
        if show_images:
            plt.show()
        else:
            plt.close()
        
        if return_plot_data:
            stacked_images = torch.stack(plotted_images_list)
            # This results in shape (num_images, num_deltas, C, H, W)
            stacked_segs = torch.stack(plotted_segs_list)

    if return_plot_data:
        return delta_v_targets, found_betas, stacked_images, stacked_segs
    else:
        return delta_v_targets, found_betas

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .data_utils import load_var, load_var_instance  # Assumes you have these helpers
from .logits import *

def _find_beta_binary_logits(yhat_i, v_i, post_trans, channel, device, 
                                    beta_range=1.0, max_search_steps=1000, search_threshold=1e-5):
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

def test_compass_theory(
    fnames, model, channel, forward_latent, post_trans, post_trans_diff, device,
    method='J', n_components=1, beta_range=1.0,
    search_threshold=1e-8, max_search_steps=1000, beta_small=1e-4,
    use_log_scale=False
):
    """
    Analyzes the theoretical efficiency of COMPASS by plotting
    K_x * R_COMPASS vs. the standard residual score R_SCP, with an option for log-scale analysis.
    """
    print(f"--- Analyzing COMPASS Efficiency for Method: {method} ---")
    
    # --- 1. Setup based on the method ---
    dl = None
    if method == 'J':
        tr_Js = load_var(fnames, 'tr', 'J')
        pca = PCA(n_components=n_components)
        pca.fit(tr_Js)
        del tr_Js

        va_Js = load_var(fnames, 'va', 'J')
        va_Js_np = va_Js.cpu().numpy() if isinstance(va_Js, torch.Tensor) else va_Js
        
        dl_np = pca.inverse_transform((va_Js_np @ pca.components_.T))
        dl_np = dl_np.reshape(dl_np.shape[0], -1, 1, 1)
        dl_np /= np.linalg.norm(dl_np, ord=2, axis=1, keepdims=True)
        dl = torch.from_numpy(dl_np).to(device)
        del va_Js, va_Js_np, dl_np
        
        representation_name = 'latent'
    elif method == 'L':
        representation_name = 'yhat'
    else:
        raise ValueError("Method must be 'J' or 'L'")

    num_cal_samples = len(fnames['va']['v'])

    # --- 2. Iterate through calibration set to gather scores ---
    r_scp_list, r_compass_list, k_x_list = [], [], []
    
    for i in range(num_cal_samples):
        v_i = load_var_instance(fnames, 'va', 'v', i).to(device)
        representation_i = load_var_instance(fnames, 'va', representation_name, i).to(device)
        
        with torch.no_grad():
            if method == 'J':
                yhat_initial = forward_latent(model, representation_i.unsqueeze(0).float())
            else: # method == 'L'
                yhat_initial = representation_i.unsqueeze(0).float()
            
            _, v_hat_initial = post_trans_diff(yhat_initial, channel)
            r_scp = abs(v_i.item() - v_hat_initial.item())
            
            if method == 'J':
                dl_i = dl[i]
                r_compass = _beta_binary_search_symmetric(
                    representation_i, dl_i, v_i, model, forward_latent, post_trans_diff, channel,
                    beta_range, max_search_steps, search_threshold
                )
            else: # method == 'L'
                r_compass = _find_beta_binary_logits(
                    representation_i.unsqueeze(0), v_i, post_trans, channel, device,
                    beta_range, max_search_steps, search_threshold
                )

            dl_i = dl[i] if method == 'J' else torch.ones_like(representation_i)
            
            rep_pos = representation_i + beta_small * dl_i
            if method == 'J':
                yhat_pos = forward_latent(model, rep_pos.unsqueeze(0).float())
            else:
                yhat_pos = rep_pos.unsqueeze(0).float()
            _, v_pos = post_trans_diff(yhat_pos, channel)

            rep_neg = representation_i - beta_small * dl_i
            if method == 'J':
                yhat_neg = forward_latent(model, rep_neg.unsqueeze(0).float())
            else:
                yhat_neg = rep_neg.unsqueeze(0).float()
            _, v_neg = post_trans_diff(yhat_neg, channel)

            k_x = abs(v_pos.item() - v_neg.item()) / (2 * beta_small)

            if k_x > 1e-9:
                r_scp_list.append(r_scp)
                r_compass_list.append(r_compass)
                k_x_list.append(k_x)

    # --- 3. Create the plot to visualize the relationship ---
    r_scp_arr = np.array(r_scp_list)
    r_compass_arr = np.array(r_compass_list)
    k_x_arr = np.array(k_x_list)
    
    x_axis_data = r_scp_arr
    y_axis_data = k_x_arr * r_compass_arr
    
    xlabel = '$R_{SCP}$'
    ylabel = r'$K_x \cdot R_{COMPASS}$'
    title_suffix = 'Linear Scale'

    if use_log_scale:
        # Filter out non-positive values before log transform
        positive_mask = (x_axis_data > 1e-9) & (y_axis_data > 1e-9)
        if np.sum(~positive_mask) > 0:
            print(f"Warning: Removed {np.sum(~positive_mask)} non-positive samples for log-scale plot.")
        
        x_axis_data_log = np.log10(x_axis_data[positive_mask])
        y_axis_data_log = np.log10(y_axis_data[positive_mask])
        
        xlabel = 'log($R_{SCP}$)'
        ylabel = r'log($K_x \cdot R_{COMPASS}$)'
        title_suffix = 'Log Scale'
        
        # Use log-transformed data for correlation and plotting
        plot_x = x_axis_data_log
        plot_y = y_axis_data_log
    else:
        plot_x = x_axis_data
        plot_y = y_axis_data

    # Compute the Pearson correlation coefficient on the appropriate scale
    correlation_matrix = np.corrcoef(plot_x, plot_y)
    correlation = correlation_matrix[0, 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(plot_x, plot_y, alpha=0.5, label='Calibration Samples')
    
    max_val = max(np.percentile(plot_x, 99.5), np.percentile(plot_y, 99.5))
    min_val = min(np.percentile(plot_x, 0.5), np.percentile(plot_y, 0.5))
    plot_range = [min(0, min_val) if not use_log_scale else min_val, max_val]

    plt.plot(plot_range, plot_range, 'r--', label='y = x')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Score Analysis ({method} | {title_suffix})\nCorrelation: {correlation:.2f}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    plt.show()

    # --- 4. Calculate and report violations of the theoretical inequality ---
    num_violations = np.sum(y_axis_data > x_axis_data + 1e-6) # Add tolerance
    print(f"Found {num_violations} out of {len(x_axis_data)} samples that violate the theoretical inequality.")

    return {
        "r_scp": r_scp_arr,
        "r_compass": r_compass_arr,
        "k_x": k_x_arr,
        "theoretical_bound": x_axis_data }
