
def PLS_PJCP(fnames, model, channel, forward_latent, post_trans, device,
                      beta_range=1.0, n_components=1, alpha=0.1,
                      search_threshold=1e-5, max_search_steps=100):
    """
    Symmetric PJCP using Partial Least Squares (PLS) regression.
    """
    # --- Setup & Training PLS Model ---
    torch.cuda.empty_cache()
    tr_Js = load_var(fnames, 'tr', 'J')
    tr_latents = load_var(fnames, 'tr', 'latent')
    
    delta_beta = beta_range / 2.0
    if isinstance(tr_Js, torch.Tensor): tr_Js_np = tr_Js.cpu().numpy()
    else: tr_Js_np = tr_Js

    dl_np_temp = tr_Js_np.copy()
    dl_np_temp = dl_np_temp.reshape(dl_np_temp.shape[0], dl_np_temp.shape[1], 1, 1)
    dl_np_temp /= np.linalg.norm(dl_np_temp, ord=2, axis=1, keepdims=True)
    dl_temp = torch.from_numpy(dl_np_temp).to(device)

    with torch.no_grad():
        _, vols_zero_tr = post_trans(forward_latent(model, tr_latents.to(device).float()), channel)
        latents_perturbed_tr = tr_latents.to(device) + delta_beta * dl_temp
        _, vols_perturbed_tr = post_trans(forward_latent(model, latents_perturbed_tr.float()), channel)
        delta_vols = (vols_perturbed_tr - vols_zero_tr).cpu().numpy()

    # --- FIX STARTS HERE ---
    # Filter out any NaN values from the training data before fitting PLS
    nan_mask = ~np.isnan(delta_vols)
    if np.any(~nan_mask):
        print(f"Warning: Found and removed {np.sum(~nan_mask)} NaN values from PLS training data.")
        tr_Js_np_clean = tr_Js_np[nan_mask]
        delta_vols_clean = delta_vols[nan_mask]
    else:
        tr_Js_np_clean = tr_Js_np
        delta_vols_clean = delta_vols
    # --- FIX ENDS HERE ---

    pls = PLSRegression(n_components=n_components)
    pls.fit(tr_Js_np_clean, delta_vols_clean)
    pls_direction = pls.x_weights_
    del tr_Js, tr_latents, dl_temp, delta_vols

    # --- Calibration Stage ---
    va_Js = load_var(fnames, 'va', 'J')
    va_latents = load_var(fnames, 'va', 'latent')
    va_vs = load_var(fnames, 'va', 'v')

    if isinstance(va_Js, torch.Tensor): va_Js_np = va_Js.cpu().numpy()
    else: va_Js_np = va_Js

    dl_np = (va_Js_np @ pls_direction) @ pls_direction.T
    dl_np = dl_np.reshape(dl_np.shape[0], dl_np.shape[1], 1, 1)
    dl_np /= np.linalg.norm(dl_np, ord=2, axis=1, keepdims=True)
    dl = torch.from_numpy(dl_np).to(device)
    del va_Js, dl_np

    scores = []
    with torch.no_grad():
        for i in range(len(va_latents)):
            score = _beta_binary_search_symmetric(
                va_latents[i].to(device), dl[i], va_vs[i].to(device),
                model, forward_latent, post_trans, channel,
                beta_range, max_search_steps=max_search_steps, search_threshold=search_threshold
            )
            scores.append(score)
    del va_latents, va_vs, dl

    # --- Calibration ---
    scores = torch.tensor(scores)
    n = len(scores)
    k = math.ceil((1 - alpha) * (n + 1))
    q_level = min(k / (n + 1), 1.0)
    beta_hat = torch.quantile(scores, q=q_level, interpolation="higher")

    # --- Inference Stage ---
    ts_Js = load_var(fnames, 'ts', 'J')
    ts_latents = load_var(fnames, 'ts', 'latent')
    ts_vs = load_var(fnames, 'ts', 'v').ravel()

    if isinstance(ts_Js, torch.Tensor): ts_Js_np = ts_Js.cpu().numpy()
    else: ts_Js_np = ts_Js
    
    dl_np = (ts_Js_np @ pls_direction) @ pls_direction.T
    dl_np = dl_np.reshape(dl_np.shape[0], dl_np.shape[1], 1, 1)
    dl_np /= np.linalg.norm(dl_np, ord=2, axis=1, keepdims=True)
    dl = torch.from_numpy(dl_np).to(device)
    del ts_Js, dl_np

    pred_vols_hi, pred_vols_lo = [], []
    with torch.no_grad():
        for i in range(len(ts_latents)):
            new_latents = torch.vstack((
                ts_latents[i].unsqueeze(0).to(device) - beta_hat * dl[i].unsqueeze(0),
                ts_latents[i].unsqueeze(0).to(device) + beta_hat * dl[i].unsqueeze(0)
            ))
            new_yhats = forward_latent(model, new_latents.float())
            _, new_vols = post_trans(new_yhats, channel)
            pred_vols_lo.append(min(new_vols[0].item(), new_vols[1].item()))
            pred_vols_hi.append(max(new_vols[0].item(), new_vols[1].item()))

    pred_vols_hi = torch.tensor(pred_vols_hi)
    pred_vols_lo = torch.tensor(pred_vols_lo)
    
    pls_pjcp_len = (pred_vols_hi - pred_vols_lo).mean().item()
    pls_pjcp_cov = (((ts_vs <= pred_vols_hi) & (ts_vs >= pred_vols_lo)).sum() / len(ts_vs)).item()
    del ts_latents, ts_vs, dl
    
    print(f"Symmetric PLS-PJCP Len: {pls_pjcp_len:.4f}, Cov: {pls_pjcp_cov:.4f}")
    return pls_pjcp_len, pls_pjcp_cov

def A_PLS_PJCP(fnames, model, channel, forward_latent, post_trans, device,
                       beta_range=1.0, n_components=1, alpha=0.1,
                       search_threshold=1e-8, max_search_steps=1000):
    """
    Asymmetric PJCP using Partial Least Squares (PLS) regression.
    """
    # --- Setup & Training PLS Model (Identical to Symmetric version) ---
    torch.cuda.empty_cache()
    tr_Js = load_var(fnames, 'tr', 'J')
    tr_latents = load_var(fnames, 'tr', 'latent')
    
    delta_beta = beta_range / 2.0
    if isinstance(tr_Js, torch.Tensor): tr_Js_np = tr_Js.cpu().numpy()
    else: tr_Js_np = tr_Js

    dl_np_temp = tr_Js_np.copy()
    dl_np_temp = dl_np_temp.reshape(dl_np_temp.shape[0], dl_np_temp.shape[1], 1, 1)
    dl_np_temp /= np.linalg.norm(dl_np_temp, ord=2, axis=1, keepdims=True)
    dl_temp = torch.from_numpy(dl_np_temp).to(device)

    with torch.no_grad():
        _, vols_zero_tr = post_trans(forward_latent(model, tr_latents.to(device).float()), channel)
        latents_perturbed_tr = tr_latents.to(device) + delta_beta * dl_temp
        _, vols_perturbed_tr = post_trans(forward_latent(model, latents_perturbed_tr.float()), channel)
        delta_vols = (vols_perturbed_tr - vols_zero_tr).cpu().numpy()

    # --- FIX STARTS HERE ---
    # Filter out any NaN values from the training data before fitting PLS
    nan_mask = ~np.isnan(delta_vols)
    if np.any(~nan_mask):
        print(f"Warning: Found and removed {np.sum(~nan_mask)} NaN values from PLS training data.")
        tr_Js_np_clean = tr_Js_np[nan_mask]
        delta_vols_clean = delta_vols[nan_mask]
    else:
        tr_Js_np_clean = tr_Js_np
        delta_vols_clean = delta_vols
    # --- FIX ENDS HERE ---

    pls = PLSRegression(n_components=n_components)
    pls.fit(tr_Js_np_clean, delta_vols_clean)
    pls_direction = pls.x_weights_
    del tr_Js, tr_latents, dl_temp, delta_vols

    # --- Calibration Stage ---
    va_Js = load_var(fnames, 'va', 'J')
    va_latents = load_var(fnames, 'va', 'latent')
    va_vs = load_var(fnames, 'va', 'v')

    if isinstance(va_Js, torch.Tensor): va_Js_np = va_Js.cpu().numpy()
    else: va_Js_np = va_Js

    dl_np = (va_Js_np @ pls_direction) @ pls_direction.T
    dl_np = dl_np.reshape(dl_np.shape[0], dl_np.shape[1], 1, 1)
    dl_np /= np.linalg.norm(dl_np, ord=2, axis=1, keepdims=True)
    dl = torch.from_numpy(dl_np).to(device)
    del va_Js, dl_np

    betas = []
    with torch.no_grad():
        for i in range(len(va_latents)):
            beta_tuple = _beta_binary_search_asymmetric(
                va_latents[i].to(device), dl[i], va_vs[i].to(device),
                model, forward_latent, post_trans, channel,
                beta_range, threshold=search_threshold, max_search_steps=max_search_steps
            )
            betas.append(beta_tuple)
    del va_latents, va_vs, dl

    # --- Calibration ---
    n = len(betas)
    alpha_lo, alpha_hi = alpha / 2.0, alpha / 2.0
    beta_pos_scores = torch.tensor([b[0] for b in betas])
    beta_neg_scores = torch.tensor([b[1] for b in betas])

    k_hi = math.ceil((1 - alpha_hi) * (n + 1))
    q_hi = min(k_hi / (n + 1), 1.0)
    beta_q_pos = torch.quantile(beta_pos_scores, q=q_hi, interpolation="higher")

    k_lo = math.floor(alpha_lo * (n + 1))
    q_lo = k_lo / (n + 1)
    beta_q_neg = torch.quantile(beta_neg_scores, q=q_lo, interpolation="lower")

    # --- Inference Stage ---
    ts_Js = load_var(fnames, 'ts', 'J')
    ts_latents = load_var(fnames, 'ts', 'latent')
    ts_vs = load_var(fnames, 'ts', 'v').ravel()

    if isinstance(ts_Js, torch.Tensor): ts_Js_np = ts_Js.cpu().numpy()
    else: ts_Js_np = ts_Js
    
    dl_np = (ts_Js_np @ pls_direction) @ pls_direction.T
    dl_np = dl_np.reshape(dl_np.shape[0], dl_np.shape[1], 1, 1)
    dl_np /= np.linalg.norm(dl_np, ord=2, axis=1, keepdims=True)
    dl = torch.from_numpy(dl_np).to(device)
    del ts_Js, dl_np

    pred_vols_hi, pred_vols_lo = [], []
    with torch.no_grad():
        for i in range(len(ts_latents)):
            new_latents = torch.vstack((
                ts_latents[i].unsqueeze(0).to(device) + beta_q_pos * dl[i].unsqueeze(0),
                ts_latents[i].unsqueeze(0).to(device) + beta_q_neg * dl[i].unsqueeze(0)
            ))
            new_yhats = forward_latent(model, new_latents.float())
            _, new_vols = post_trans(new_yhats, channel)
            pred_vols_hi.append(new_vols[0].detach().cpu().unsqueeze(0))
            pred_vols_lo.append(new_vols[1].detach().cpu().unsqueeze(0))

    pred_vols_hi = torch.cat(pred_vols_hi)
    pred_vols_lo = torch.cat(pred_vols_lo)
    
    pls_apjcp_len = (pred_vols_hi - pred_vols_lo).mean().item()
    pls_apjcp_cov = (((ts_vs <= pred_vols_hi) & (ts_vs >= pred_vols_lo)).sum() / len(ts_vs)).item()
    del ts_latents, ts_vs, dl
    
    print(f"Asymmetric PLS-PJCP Len: {pls_apjcp_len:.4f}, Cov: {pls_apjcp_cov:.4f}")
    return pls_apjcp_len, pls_apjcp_cov

def PLS_Sweep(fnames, model, channel, forward_latent, post_trans, device,
              beta_range=1.0, beta_steps=20, n_components=1, plot=True):
    """
    Performs a diagnostic sweep using a PLS-derived perturbation direction
    to visualize the beta vs. delta_volume relationship.
    """
    # --- 1. Train the PLS Model ---
    torch.cuda.empty_cache()
    tr_Js = load_var(fnames, 'tr', 'J')
    tr_latents = load_var(fnames, 'tr', 'latent')
    
    delta_beta = beta_range / 2.0
    
    if isinstance(tr_Js, torch.Tensor): tr_Js_np = tr_Js.cpu().numpy()
    else: tr_Js_np = tr_Js

    # Use the raw Jacobians as the initial perturbation directions to get a clean signal
    dl_np_temp = tr_Js_np.copy()
    dl_np_temp = dl_np_temp.reshape(dl_np_temp.shape[0], dl_np_temp.shape[1], 1, 1)
    dl_np_temp /= np.linalg.norm(dl_np_temp, ord=2, axis=1, keepdims=True)
    dl_temp = torch.from_numpy(dl_np_temp).to(device)

    with torch.no_grad():
        _, vols_zero_tr = post_trans(forward_latent(model, tr_latents.to(device).float()), channel)
        latents_perturbed_tr = tr_latents.to(device) + delta_beta * dl_temp
        _, vols_perturbed_tr = post_trans(forward_latent(model, latents_perturbed_tr.float()), channel)
        delta_vols = (vols_perturbed_tr - vols_zero_tr).cpu().numpy()

    pls = PLSRegression(n_components=n_components)
    pls.fit(tr_Js_np, delta_vols)
    print('Score: ',pls.score(tr_Js_np,delta_vols))
    pls_direction = pls.x_weights_
    del tr_Js, tr_latents, dl_temp, delta_vols
    torch.cuda.empty_cache()

    # --- 2. Perform Sweep on Test Set using PLS Direction ---
    ts_Js = load_var(fnames, 'ts', 'J')
    ts_latents = load_var(fnames, 'ts', 'latent')

    if isinstance(ts_Js, torch.Tensor): ts_Js_np = ts_Js.cpu().numpy()
    else: ts_Js_np = ts_Js
    
    # Project test Jacobians onto the learned PLS direction
    dl_np = (ts_Js_np @ pls_direction) @ pls_direction.T
    dl_np = dl_np.reshape(dl_np.shape[0], dl_np.shape[1], 1, 1)
    dl_np /= np.linalg.norm(dl_np, ord=2, axis=1, keepdims=True)
    dl = torch.from_numpy(dl_np).to(device)

    # Create symmetric beta range for the sweep
    pos_betas = np.linspace(0.0, beta_range, beta_steps)
    neg_betas = -pos_betas[1:][::-1]
    betas = np.concatenate([neg_betas, pos_betas])
    
    # Store original and perturbed volumes
    o_vols = np.zeros((len(betas), len(ts_Js)))
    n_vols = np.zeros((len(betas), len(ts_Js)))

    with torch.no_grad():
        # Get original volumes once
        _, base_vols = post_trans(forward_latent(model, ts_latents.to(device).float()), channel)
        
        for i, beta in enumerate(betas):
            new_latents = ts_latents.to(device) + beta * dl
            _, new_vol = post_trans(forward_latent(model, new_latents.float()), channel)
            
            o_vols[i, :] = base_vols.cpu().numpy()
            n_vols[i, :] = new_vol.cpu().numpy()

    if plot:
        plt.figure(figsize=(10, 6))
        # Calculate percentage change, handle potential division by zero
        delta_v_percent = np.divide((n_vols - o_vols), o_vols, 
                                    out=np.zeros_like(n_vols), 
                                    where=o_vols!=0) * 100
        
        plt.plot(betas, delta_v_percent)
        plt.ylim([-50,50])
        plt.grid(True)
        plt.xlabel('Beta (Perturbation Magnitude)')
        plt.ylabel('∆V (%)')
        plt.title('PLS-Guided Perturbation Sweep')
        plt.grid(True)
        plt.show()
        
    return betas, delta_v_percent
"""PLS-based diagnostic/perturbation utilities for COMPASS 2D."""

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression

from .data_utils import load_var
