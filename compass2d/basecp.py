import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import math
from torch.utils.data import Subset,ConcatDataset,DataLoader
import torch
from .data_utils import *
import matplotlib.pyplot as plt

def unpack_tensors2npy(fnames):
    tr_vhats = load_var(fnames, 'tr', 'vhat').detach().cpu().numpy().ravel()
    va_vhats = load_var(fnames, 'va', 'vhat').detach().cpu().numpy().ravel()
    ts_vhats = load_var(fnames, 'ts', 'vhat').detach().cpu().numpy().ravel()
    tr_vs = load_var(fnames, 'tr', 'v').detach().cpu().numpy().ravel()
    va_vs = load_var(fnames, 'va', 'v').detach().cpu().numpy().ravel()
    ts_vs = load_var(fnames, 'ts', 'v').detach().cpu().numpy().ravel()
    return tr_vhats,tr_vs,va_vhats,va_vs,ts_vhats,ts_vs

def cover(lb,ub,y):
    return ((y>=lb)&(y<=ub)).sum()/len(y)
    
def SCP(pred_cal,gt_cal,pred_ts,gt_ts,alpha=0.1,return_raw=False):
    err_cal=gt_cal-pred_cal
    n=len(err_cal)
    adjq=math.ceil((n+1)*(1-alpha))/n
    scp_q=np.quantile(np.abs(err_cal),adjq,method='higher')
    scp_cov=cover(pred_ts-scp_q,pred_ts+scp_q,gt_ts)
    scp_len=2*scp_q
    print('SCP Len: ',scp_len,'SCP Cov:',scp_cov)
    if return_raw:
        return scp_len,scp_cov,np.stack([pred_ts+scp_q,pred_ts-scp_q,gt_ts])
    return scp_len,scp_cov
    
def CQR(pred_tr,gt_tr,pred_cal,gt_cal,pred_ts,gt_ts,alpha=0.1,return_raw=False):
    err_cal=gt_cal-pred_cal
    n=len(err_cal)
    adjq=math.ceil((n+1)*(1-alpha))/n
    common_params=dict(learning_rate=0.1,n_estimators=50,
                       max_depth=3,min_samples_leaf=1,min_samples_split=9,)
    gbr5=GradientBoostingRegressor(loss="quantile",alpha=alpha/2, **common_params)
    gbr95=GradientBoostingRegressor(loss="quantile",alpha=1-alpha/2, **common_params)
    gbr5.fit(pred_tr.reshape(-1,1),gt_tr)
    gbr95.fit(pred_tr.reshape(-1,1),gt_tr)
    pred_cal_lower=gbr5.predict(pred_cal.reshape(-1,1))
    pred_cal_upper=gbr95.predict(pred_cal.reshape(-1,1))
    # qlo=np.quantile(va_vhat_lower-va_vs.numpy(),0.9)
    # qhi=np.quantile(va_vs.numpy()-va_vhat_upper,0.9)
    q=np.quantile(np.maximum(pred_cal_lower-gt_cal,gt_cal-pred_cal_upper),adjq)
    pred_ts_lower=gbr5.predict(pred_ts.reshape(-1,1))
    pred_ts_upper=gbr95.predict(pred_ts.reshape(-1,1))
    cqr_cov=((gt_ts>=pred_ts_lower-q)&(gt_ts<=pred_ts_upper+q)).sum()/len(gt_ts)
    cqr_len=((pred_ts_upper+q)-(pred_ts_lower-q)).mean()
    print('CQR Len: ',cqr_len,'CQR Cov: ',cqr_cov)
    if return_raw:
        return cqr_len,cqr_cov,np.stack([pred_ts_upper+q,pred_ts_lower-q,gt_ts])
    return cqr_len,cqr_cov

def Local(pred_tr,gt_tr,pred_cal,gt_cal,pred_ts,gt_ts,alpha=0.1,return_raw=False):
    err_cal=gt_cal-pred_cal
    n=len(err_cal)
    adjq=math.ceil((n+1)*(1-alpha))/n
    rf_config=dict(n_estimators=1000,min_samples_leaf=100,max_features=1)
    mean_estimator = RandomForestRegressor(**rf_config)
    mad_estimator = RandomForestRegressor(**rf_config)
    mean_estimator.fit(pred_tr.reshape(-1,1),gt_tr)
    res=mean_estimator.predict(pred_tr.reshape(-1,1))
    res_err=np.abs(res-gt_tr)
    mad_estimator.fit(pred_tr.reshape(-1,1),res_err)
    mean_pred_cal = mean_estimator.predict(pred_cal.reshape(-1,1))
    mean_pred_test = mean_estimator.predict(pred_ts.reshape(-1,1))
    mad_pred_cal = mad_estimator.predict(pred_cal.reshape(-1,1))
    mad_pred_test = mad_estimator.predict(pred_ts.reshape(-1,1))
    q=np.quantile(np.abs(mean_pred_cal-gt_cal)/np.abs(mad_pred_cal),adjq)
    y_lower_local=mean_pred_test-q*np.abs(mad_pred_test)
    y_upper_local=mean_pred_test+q*np.abs(mad_pred_test)
    local_len=(y_upper_local-y_lower_local).mean()
    local_cov=((gt_ts>=y_lower_local)&(gt_ts<=y_upper_local)).sum()/len(gt_ts)
    print('Local Len: ',local_len,'Local Cov: ',local_cov)
    if return_raw:
        return local_len,local_cov,np.stack([y_upper_local,y_lower_local,gt_ts])
    return local_len,local_cov

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMClassifier
import warnings

# Suppress common LightGBM warnings
warnings.filterwarnings("ignore", message="No further splits with positive gain, best gain: -inf")

# --- Helper Functions ---

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

def SCP_WCP(fnames, output_data, alpha=0.1, weights=None, return_raw=False):
    """
    Split Conformal Prediction (SCP) with an optional WCP layer for label shift.
    Data unpacking is handled internally.
    """
    # --- Unpack Data ---
    pred_cal = load_var(fnames, 'va', 'vhat').ravel()
    gt_cal = load_var(fnames, 'va', 'v').ravel()
    pred_ts = load_var(fnames, 'ts', 'vhat').ravel()
    gt_ts = load_var(fnames, 'ts', 'v').ravel()
    cal_labels = output_data.get('cal_classes', [])
    ts_labels = output_data.get('test_classes', [])
    
    err_cal = gt_cal - pred_cal
    
    print("Using WCP for SCP calibration (Label Shift)...")
    cal_counts = Counter(cal_labels)
    test_counts = Counter(ts_labels)
    P_cal = {k: v / len(cal_labels) for k, v in cal_counts.items()}
    P_test = {k: v / len(ts_labels) for k, v in test_counts.items()}
    
    n = len(err_cal)
    scores = np.abs(err_cal)
    q = _compute_weighted_quantile(scores, weights, 1 - alpha)
    
    scp_len, scp_cov = 2 * q, cover(pred_ts - q, pred_ts + q, gt_ts).item()
    print(f'SCP-WCP Len: {scp_len:.4f}, SCP-WCP Cov: {scp_cov:.4f}')
    if return_raw:
        return scp_len, scp_cov, np.stack([pred_ts+q,pred_ts-q,gt_ts])
    return scp_len, scp_cov


def CQR_WCP(fnames, output_data, alpha=0.1, weights=None, return_raw=False):
    """
    Conformalized Quantile Regression (CQR) with an optional WCP layer for label shift.
    Data unpacking is handled internally.
    """
    # --- Unpack Data ---
    pred_tr = load_var(fnames, 'tr', 'vhat').ravel()
    gt_tr = load_var(fnames, 'tr', 'v').ravel()
    pred_cal = load_var(fnames, 'va', 'vhat').ravel()
    gt_cal = load_var(fnames, 'va', 'v').ravel().numpy()
    pred_ts = load_var(fnames, 'ts', 'vhat').ravel()
    gt_ts = load_var(fnames, 'ts', 'v').ravel().numpy()
    cal_labels = output_data.get('cal_classes', [])
    ts_labels = output_data.get('test_classes', [])

    common_params = dict(learning_rate=0.1, n_estimators=50, max_depth=3, min_samples_leaf=1, min_samples_split=9)
    gbr_lo = GradientBoostingRegressor(loss="quantile", alpha=alpha/2, **common_params)
    gbr_hi = GradientBoostingRegressor(loss="quantile", alpha=1-alpha/2, **common_params)
    gbr_lo.fit(pred_tr.reshape(-1, 1), gt_tr)
    gbr_hi.fit(pred_tr.reshape(-1, 1), gt_tr)
    
    pred_cal_lower = gbr_lo.predict(pred_cal.reshape(-1, 1))
    pred_cal_upper = gbr_hi.predict(pred_cal.reshape(-1, 1))
    scores = np.maximum(pred_cal_lower - gt_cal, gt_cal - pred_cal_upper)

    q = 0
    if weights is not None:
        q = _compute_weighted_quantile(scores, weights, 1 - alpha)
    else:
        n = len(gt_cal)
        adjq = math.ceil((n + 1) * (1 - alpha)) / n
        q = np.quantile(scores, min(adjq, 1.0), method='higher')
    
    pred_ts_lower = gbr_lo.predict(pred_ts.reshape(-1, 1))
    pred_ts_upper = gbr_hi.predict(pred_ts.reshape(-1, 1))
    
    cqr_len = ((pred_ts_upper + q) - (pred_ts_lower - q)).mean()
    cqr_cov = cover(pred_ts_lower - q, pred_ts_upper + q, gt_ts)
    
    print(f'CQR-WCP Len: {cqr_len:.4f}, CQR-WCP Cov: {cqr_cov:.4f}')
    if return_raw:
        return cqr_len, cqr_cov, np.stack([pred_ts_upper+q,pred_ts_lower-q,gt_ts])
    return cqr_len, cqr_cov


def Local_WCP(fnames, output_data, alpha=0.1, weights=None, return_raw=False):
    """
    Locally adaptive conformal prediction with an optional WCP layer for label shift.
    Data unpacking is handled internally.
    """
    # --- Unpack Data ---
    pred_tr = load_var(fnames, 'tr', 'vhat').ravel().numpy()
    gt_tr = load_var(fnames, 'tr', 'v').ravel().numpy()
    pred_cal = load_var(fnames, 'va', 'vhat').ravel().numpy()
    gt_cal = load_var(fnames, 'va', 'v').ravel().numpy()
    pred_ts = load_var(fnames, 'ts', 'vhat').ravel().numpy()
    gt_ts = load_var(fnames, 'ts', 'v').ravel().numpy()
    cal_labels = output_data.get('cal_classes', [])
    ts_labels = output_data.get('test_classes', [])

    rf_config = dict(n_estimators=1000, min_samples_leaf=100, max_features=1)
    mean_estimator = RandomForestRegressor(**rf_config)
    mad_estimator = RandomForestRegressor(**rf_config)
    
    mean_estimator.fit(pred_tr.reshape(-1, 1), gt_tr)
    res_err = np.abs(mean_estimator.predict(pred_tr.reshape(-1, 1)) - gt_tr)
    mad_estimator.fit(pred_tr.reshape(-1, 1), res_err)

    mean_pred_cal = mean_estimator.predict(pred_cal.reshape(-1, 1))
    mad_pred_cal = mad_estimator.predict(pred_cal.reshape(-1, 1))
    scores = np.abs(mean_pred_cal - gt_cal) / np.where(np.abs(mad_pred_cal) > 1e-6, np.abs(mad_pred_cal), 1e-6)

    q = 0
    if weights is not None:
        q = _compute_weighted_quantile(scores, weights, 1 - alpha)
    else:
        n = len(gt_cal)
        adjq = math.ceil((n + 1) * (1 - alpha)) / n
        q = np.quantile(scores, min(adjq, 1.0), method='higher')

    mean_pred_test = mean_estimator.predict(pred_ts.reshape(-1, 1))
    mad_pred_test = mad_estimator.predict(pred_ts.reshape(-1, 1))
    
    y_lower_local = mean_pred_test - q * np.abs(mad_pred_test)
    y_upper_local = mean_pred_test + q * np.abs(mad_pred_test)
    
    local_len = (y_upper_local - y_lower_local).mean()
    local_cov = cover(y_lower_local, y_upper_local, gt_ts)
    
    print(f'Local-WCP Len: {local_len:.4f}, Local-WCP Cov: {local_cov:.4f}')
    if return_raw:
        return local_len, local_cov, np.stack([y_upper_local,y_lower_local,gt_ts])
    return local_len, local_cov
