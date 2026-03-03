from math import floor
import numpy as np
from .data_utils import *
import math

def _compute_weighted_quantile(scores, weights, quantile_level):
    """Computes a quantile from a set of scores, weighted by their importance."""
    scores = np.array(scores)
    weights = np.array(weights)
    weights = np.maximum(weights, 0) # Ensure weights are non-negative
    
    # Sort scores and corresponding weights
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    total_weight = np.sum(sorted_weights)
    if total_weight <= 0:
        return np.max(scores) # Fallback

    target_weight_sum = quantile_level * total_weight
    cumulative_weights = np.cumsum(sorted_weights)
    
    # Find the first index where cumulative weight exceeds the target
    quantile_index = np.searchsorted(cumulative_weights, target_weight_sum)
    
    # Handle edge case where quantile_index is out of bounds
    if quantile_index >= len(sorted_scores):
        return sorted_scores[-1]
        
    return sorted_scores[quantile_index]

def cover(lower_bound, upper_bound, true_values):
    """Calculates the coverage of a set of prediction intervals."""
    return ((true_values >= lower_bound) & (true_values <= upper_bound)).sum()/len(true_values)

def E2E(fnames, alpha=0.1, sym=True,weights=None, return_raw=False):
    """
    Unified (a)symmetric End-to-End Conformal Prediction with a choice of WCP layers.
    
    Args:
        fnames (dict): For E2E, this should point to the QR model's fnames.
        alpha (float): The desired miscoverage rate.
        sym (bool): If True, runs the symmetric version. If False, runs asymmetric.
        use_wcp_weighting (bool): If True, applies the selected weighting method.
        weighting_method (str): 'label_shift' or 'feature_shift' (uses latents).
    """
    # --- 1. Unpack Data ---
    va_vhats = load_var(fnames, 'va', 'vhat')
    va_vhats_lo, va_vhats_hi = va_vhats[:, 1].ravel(), va_vhats[:, 2].ravel()
    
    ts_vhats = load_var(fnames, 'ts', 'vhat')
    ts_vhats_lo, ts_vhats_hi = ts_vhats[:, 1].ravel(), ts_vhats[:, 2].ravel()

    gt_cal = load_var(fnames, 'va', 'v').ravel()
    gt_ts = load_var(fnames, 'ts', 'v').ravel()
    
    # --- 2. Calculate Non-Conformity Scores ---
    scores, scores_lo, scores_hi = None, None, None
    if sym:
        scores = np.maximum(va_vhats_lo - gt_cal, gt_cal - va_vhats_hi)
    else: # Asymmetric
        scores_lo = va_vhats_lo - gt_cal
        scores_hi = gt_cal - va_vhats_hi

    # --- 3. Calibration: Weighted (WCP) or Standard ---
    q, q_lo, q_hi = 0, 0, 0
    n = len(gt_cal)
    
    if weights is not None:
        if sym:
            q = _compute_weighted_quantile(scores, weights, 1 - alpha)
        else:
            q_lo = _compute_weighted_quantile(scores_lo, weights, 1 - alpha/2)
            q_hi = _compute_weighted_quantile(scores_hi, weights, 1 - alpha/2)
    else:
        # print(f"Using standard quantile for {'Symmetric' if sym else 'Asymmetric'} E2E calibration...")
        if sym:
            adjq = math.ceil((n + 1) * (1 - alpha)) / n
            q = np.quantile(scores, min(adjq, 1.0), method='higher')
        else:
            adjq_asym = math.ceil((n + 1) * (1 - alpha/2)) / n
            q_lo = np.quantile(scores_lo, min(adjq_asym, 1.0), method='higher')
            q_hi = np.quantile(scores_hi, min(adjq_asym, 1.0), method='higher')

    # --- 4. Inference ---
    if sym:
        final_lo = ts_vhats_lo - q
        final_hi = ts_vhats_hi + q
    else: # Asymmetric
        final_lo = ts_vhats_lo - q_lo
        final_hi = ts_vhats_hi + q_hi
    
    e2e_cov = cover(final_lo, final_hi, gt_ts)
    e2e_len = (final_hi - final_lo).mean()
    
    mode_str = "Symmetric" if sym else "Asymmetric"
    print(f'E2E ({mode_str}) Len: {e2e_len:.4f}, E2E Cov: {e2e_cov:.4f}')
    if return_raw:
        return e2e_len.item(), e2e_cov.item(), np.stack([final_hi,final_lo,gt_ts])
    return e2e_len.item(), e2e_cov.item()
