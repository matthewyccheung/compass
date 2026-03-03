import torch
from torch.utils.data import DataLoader, ConcatDataset
from monai.networks.nets import UNet, SegResNet
from datetime import datetime
import numpy as np
import os
import sys
import pandas as pd
import pickle
import time

# Allow running from repo root or from within `2D/`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_ROOT)

# Import your custom modules
from compass2d.dataloaders import *  # noqa: E402
from compass2d.data_utils import *  # noqa: E402
from compass2d.logits import *  # noqa: E402
from compass2d.basecp import *  # noqa: E402
from compass2d.e2e import *  # noqa: E402
from compass2d.model_fns import *  # noqa: E402
from compass_paths import get_results_dir

# =================================================================================
# ✅ 1. CENTRALIZED CONFIGURATION
# =================================================================================
# All settings are in one place for easy management.
config = {
    # --- Execution Settings ---
    "device": "cuda:3",
    "device2": "cuda:4",
    "dataset": "ebhi",  # Options: 'ebhi', 'ham10k', 'kvasir', 'tn3k'
    "arch": "UNet",  # Options: 'UNet', 'SegResNet'
    "metric": "mask_sum",  # Scalar target (default): sum of mask pixels
    "generate_vars": False,  # Set to False to skip data extraction if already done
    "overwrite": False,
    "evaluation_runs": 100,  # Number of times to run the evaluation loop
    "shift": "",
    # --- General Parameters ---
    "size": 128,
    "num_workers": 64,
    "alpha": 0.1,
    "channel": 0,
}

# =================================================================================
# ✅ 2. CONFIGURATION DICTIONARIES
# =================================================================================
# region: Configuration Dictionaries (Model Specs, Names, Loaders)
model_architectures = {
    'UNet': UNet,
    'SegResNet': SegResNet,
}

dataset_loaders = {
    "ebhi": ebhi,
    "ham10k": ham10k,
    "kvasir": kvasir,
    "tn3k": tn3k,
}

model_configs = {
	    'ebhi': {
	        'UNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 1, 'channels': (32, 64, 128, 256), 'strides': (2, 2, 2), 'num_res_units': 2, 'norm': 'batch', 'dropout': 0.1},
	        'SegResNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 1, 'init_filters': 32, 'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1), 'norm': 'batch', 'dropout_prob': 0.1, 'use_conv_final': True},
	    },
	    'ham10k': {
	        'UNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 1, 'channels': (32, 64, 128, 256), 'strides': (2, 2, 2), 'num_res_units': 2, 'norm': 'batch', 'dropout': 0.1},
	        'SegResNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 1, 'init_filters': 32, 'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1), 'norm': 'batch', 'dropout_prob': 0.1, 'use_conv_final': True},
	    },
		    'tn3k': {
		        'UNet': {'spatial_dims': 2, 'in_channels': 1, 'out_channels': 1, 'channels': (32, 64, 128, 256), 'strides': (2, 2, 2), 'num_res_units': 2, 'norm': 'batch', 'dropout': 0.1},
		        'SegResNet': {'spatial_dims': 2, 'in_channels': 1, 'out_channels': 1, 'init_filters': 32, 'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1), 'norm': 'batch', 'dropout_prob': 0.1, 'use_conv_final': True},
		    },
		    'kvasir': {
		        'UNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 1, 'channels': (32, 64, 128, 256), 'strides': (2, 2, 2), 'num_res_units': 2, 'norm': 'batch', 'dropout': 0.1},
		        'SegResNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 1, 'init_filters': 32, 'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1), 'norm': 'batch', 'dropout_prob': 0.1, 'use_conv_final': True},
		    }
	}

qr_model_configs = {
	    'ebhi': {
	        'UNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 3, 'channels': (32, 64, 128, 256), 'strides': (2, 2, 2), 'num_res_units': 2, 'norm': 'batch', 'dropout': 0.1},
	        'SegResNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 3, 'init_filters': 32, 'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1), 'norm': 'batch', 'dropout_prob': 0.1, 'use_conv_final': True},
	    },
	    'ham10k': {
	        'UNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 3, 'channels': (32, 64, 128, 256), 'strides': (2, 2, 2), 'num_res_units': 2, 'norm': 'batch', 'dropout': 0.1},
	        'SegResNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 3, 'init_filters': 32, 'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1), 'norm': 'batch', 'dropout_prob': 0.1, 'use_conv_final': True},
	    },
		    'tn3k': {
		        'UNet': {'spatial_dims': 2, 'in_channels': 1, 'out_channels': 3, 'channels': (32, 64, 128, 256), 'strides': (2, 2, 2), 'num_res_units': 2, 'norm': 'batch', 'dropout': 0.1},
		        'SegResNet': {'spatial_dims': 2, 'in_channels': 1, 'out_channels': 3, 'init_filters': 32, 'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1), 'norm': 'batch', 'dropout_prob': 0.1, 'use_conv_final': True},
		    },
		    'kvasir': {
		        'UNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 3, 'channels': (32, 64, 128, 256), 'strides': (2, 2, 2), 'num_res_units': 2, 'norm': 'batch', 'dropout': 0.1},
		        'SegResNet': {'spatial_dims': 2, 'in_channels': 3, 'out_channels': 3, 'init_filters': 32, 'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1), 'norm': 'batch', 'dropout_prob': 0.1, 'use_conv_final': True},
		    }
	}

model_names = {
    'ebhi': {'UNet': 'UNet_20250901_171557', 'SegResNet': 'SegResNet_20250901_144612'},
    'ham10k': {'UNet': 'UNet_20250626_084600', 'SegResNet': 'SegResNet_20250805_090029'},
    'tn3k': {'UNet': 'UNet_20250724_211252', 'SegResNet': 'SegResNet_20250805_093255'},
    'kvasir': {'UNet': 'UNet_20250829_181615', 'SegResNet': 'SegResNet_20250829_194313'},
}

qr_model_names = {
    'ebhi': {'UNet': 'QRUNet_20250901_144059', 'SegResNet': 'QRSegResNet_20250901_144625'},
    'ham10k': {'UNet': 'QRUNet_20250611_140620', 'SegResNet': 'QRSegResNet_20250805_090837'},
    'tn3k': {'UNet': 'QRUNet_20250724_211250', 'SegResNet': 'QRSegResNet_20250805_093256'},
    'kvasir': {'UNet': 'QRUNet_20250829_183346', 'SegResNet': 'QRSegResNet_20250829_194337'},
}

if config['shift']=='prevalence':
    model_names['ebhi']={'UNet': 'UNet_noshift_20250907_193114'}
    qr_model_names['ebhi']={'UNet':'QRUNet_noshift_20250907_193131'}
    model_names['ham10k']={'UNet': 'UNet_20250908_090606'}
    qr_model_names['ham10k']={'UNet':'QRUNet_prevalence_20250908_090627'}
    

# endregion

# =================================================================================
# ✅ 3. HELPER FUNCTIONS
# =================================================================================
def get_post_transforms(metric_name):
    """Returns the post-processing functions based on the metric name."""
    transform_map = {
        "mask_sum": (post_trans_mask_sum, post_trans_mask_sum_diff),
    }
    return transform_map.get(metric_name, (None, None))


def get_dataset(dataset_name, size, root_dir=None, fnames=None):
    """Loads dataset splits based on `dataset_name`."""
    loader_class = dataset_loaders.get(dataset_name)
    if not loader_class:
        raise ValueError(
            f"Dataset '{dataset_name}' not recognized. Supported: {sorted(dataset_loaders.keys())}"
        )

    if fnames is not None:
        train_set = loader_class(
            img_paths=fnames["train_images"],
            seg_paths=fnames["train_labels"],
            size=size,
            transform=True,
        )
        val_set = loader_class(
            img_paths=fnames["cal_images"],
            seg_paths=fnames["cal_labels"],
            size=size,
            transform=False,
        )
        test_set = loader_class(
            img_paths=fnames["test_images"],
            seg_paths=fnames["test_labels"],
            size=size,
            transform=False,
        )
        return train_set, val_set, test_set

    if root_dir is None:
        raise ValueError("root_dir must be provided when fnames is None.")

    train_set = loader_class(
        img_dir=f"{root_dir}imagesTr/",
        seg_dir=f"{root_dir}labelsTr/",
        size=size,
        transform=True,
    )
    val_set = loader_class(
        img_dir=f"{root_dir}imagesVa/",
        seg_dir=f"{root_dir}labelsVa/",
        size=size,
        transform=False,
    )
    test_set = loader_class(
        img_dir=f"{root_dir}imagesTs/",
        seg_dir=f"{root_dir}labelsTs/",
        size=size,
        transform=False,
    )
    return train_set, val_set, test_set

def get_model(arch_name, model_spec, device, model_path):
    """Initializes a model, loads its weights, and sets it to evaluation mode."""
    model_class = model_architectures.get(arch_name)
    if not model_class:
        raise ValueError(f"Architecture '{arch_name}' not recognized.")
    
    model = model_class(**model_spec).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# =================================================================================
# ✅ 4. MAIN SCRIPT LOGIC
# =================================================================================

# --- Setup ---
torch.cuda.empty_cache()
post_trans, post_trans_diff = get_post_transforms(config['metric'])
if post_trans is None:
    raise ValueError(f"Metric '{config['metric']}' not recognized.")

# Define root directory based on dataset
dataset_name = config["dataset"]
DATASET_ROOTS = {
    "ebhi": "/scratch/yc130/EBHI-SEG/",
    "ham10k": "/scratch/yc130/ham10k/",
    "kvasir": "/scratch/yc130/Kvasir-SEG/",
    "tn3k": "/scratch/yc130/Thyroid/tn3k/",
}
if dataset_name not in DATASET_ROOTS:
    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. Supported: {sorted(DATASET_ROOTS.keys())}"
    )
root_dir = DATASET_ROOTS[dataset_name]


# --- Data Loading ---
if config['shift']=='prevalence':
    with open(f"{root_dir}{config['shift']}_fnames.pkl", "rb") as f:
        fnames=pickle.load(f)
    train_set, val_set, test_set = get_dataset(dataset_name,config['size'],fnames=fnames)
else:
    train_set, val_set, test_set = get_dataset(dataset_name,config['size'],root_dir=root_dir)
train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=config['num_workers'])
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=config['num_workers'])
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=config['num_workers'])
loaders = [train_loader, val_loader, test_loader]

# --- Model Loading ---
arch = config['arch']
model_spec = model_configs[dataset_name][arch]
qr_model_spec = qr_model_configs[dataset_name][arch]
model_name = model_names[dataset_name][arch]
qr_model_name = qr_model_names[dataset_name][arch]

model_path = os.path.join(root_dir, 'models', model_name, 'best_model.pth')
qr_model_path = os.path.join(root_dir, 'models', qr_model_name, 'best_model.pth')
# # Special case for a specific ham10k model
if (dataset_name == 'ham10k') and (arch == 'UNet') and (config['shift']!='prevalence'):
    qr_model_path = os.path.join(root_dir, 'models', qr_model_name, 'model_49.pth')

model = get_model(arch, model_spec, config['device'], model_path)
qr_model = get_model(arch, qr_model_spec, config['device2'], qr_model_path)

# --- Feature extraction and caching (logits-only) ---
# This runner evaluates COMPASS-L / A-COMPASS-L and does NOT compute/cache Jacobians.
forward_x, forward_latent, _compute_jacobian_unused = get_fns(arch, post_trans_diff)

# Cache under a separate directory to avoid clobbering Jacobian caches.
data_dir = os.path.join(root_dir, 'data', f'{model_name}_Logits', config['metric'] + os.sep)
qr_data_dir = os.path.join(root_dir, 'data', qr_model_name, config['metric'] + os.sep)

if config['generate_vars']:
    print('Extracting variables for standard model (logits-only, no Jacobians)...')
    os.makedirs(data_dir, exist_ok=True)
    DataExtractionWrapperSave(
        loaders,
        data_dir,
        model,
        config['channel'],
        forward_x,
        forward_latent,
        _compute_jacobian_unused,
        post_trans,
        post_trans_diff,
        config['device'],
        jacobians=False,
        full_jacobians=False,
    )

    print('Extracting variables for QR model...')
    os.makedirs(qr_data_dir, exist_ok=True)
    QRDataExtractionWrapperSave(
        loaders,
        qr_data_dir,
        qr_model,
        forward_x,
        forward_latent,
        _compute_jacobian_unused,
        post_trans,
        post_trans_diff,
        config['device2'],
        metric=config['metric'],
    )

# --- Evaluation Loop (logits-only) ---
var_names = ['vhat', 'v', 'latent', 'yhat', 'seghat']
qr_var_names = ['vhat', 'v', 'latent']
split_names = ['tr', 'va', 'ts']

def _missing_cached_vars(fnames_dict, *, required_splits, required_vars):
    missing = []
    for split in required_splits:
        for var in required_vars:
            if len(fnames_dict.get(split, {}).get(var, [])) == 0:
                missing.append((split, var))
    return missing

fnames = get_fnames(data_dir, var_names, split_names)
qr_fnames = get_fnames(qr_data_dir, qr_var_names, split_names)
missing = _missing_cached_vars(fnames, required_splits=split_names, required_vars=var_names)
missing_qr = _missing_cached_vars(qr_fnames, required_splits=split_names, required_vars=qr_var_names)

if missing or missing_qr:
    if not config['generate_vars']:
        print(
            'Cached variables not found (or empty). Re-running variable extraction now. '
            "Set `config['generate_vars']=True` to always regenerate."
        )
        os.makedirs(data_dir, exist_ok=True)
        DataExtractionWrapperSave(
            loaders,
            data_dir,
            model,
            config['channel'],
            forward_x,
            forward_latent,
            _compute_jacobian_unused,
            post_trans,
            post_trans_diff,
            config['device'],
            jacobians=False,
            full_jacobians=False,
        )

        os.makedirs(qr_data_dir, exist_ok=True)
        QRDataExtractionWrapperSave(
            loaders,
            qr_data_dir,
            qr_model,
            forward_x,
            forward_latent,
            _compute_jacobian_unused,
            post_trans,
            post_trans_diff,
            config['device2'],
            metric=config['metric'],
        )

        fnames = get_fnames(data_dir, var_names, split_names)
        qr_fnames = get_fnames(qr_data_dir, qr_var_names, split_names)
        missing = _missing_cached_vars(fnames, required_splits=split_names, required_vars=var_names)
        missing_qr = _missing_cached_vars(qr_fnames, required_splits=split_names, required_vars=qr_var_names)

    if missing or missing_qr:
        details = []
        if missing:
            details.append(f'standard model cache: {data_dir}')
            details.extend([f'  - missing {s}/{v}' for s, v in missing])
        if missing_qr:
            details.append(f'QR model cache: {qr_data_dir}')
            details.extend([f'  - missing {s}/{v}' for s, v in missing_qr])
        raise FileNotFoundError(
            "Cached variable tensors are missing or empty.\n"
            + "\n".join(details)
            + "\n"
            + "Verify your dataset splits are non-empty and that cached variables exist."
        )

if config['shift'] == 'prevalence':
    results_dir = get_results_dir(
        '2D',
        dataset_name,
        config['metric'],
        f"{arch}_results_{config['shift']}3_{config['alpha']}",
    )
    raw_dir = get_results_dir('2D', f"raw{config['shift']}_{config['alpha']}")
else:
    results_dir = get_results_dir('2D', dataset_name, config['metric'], f"{arch}_results3_{config['alpha']}")
    raw_dir = get_results_dir('2D', f"raw_{config['alpha']}")

raw_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

all_results = []
# for i in range(config['evaluation_runs']):
for i in range(0, 50, 1):
    result_path = results_dir / f"df{i}.csv"

    print(f"\n--- Starting Evaluation Run {i+1}/{config['evaluation_runs']} ---")

    if config['shift'] == 'prevalence':
        shift = config['shift']
        with open(os.path.join(root_dir, f"{shift}_fnames.pkl"), 'rb') as f:
            output_data = pickle.load(f)

        if config['dataset'] == 'ebhi':
            minority_classes = ['Adenocarcinoma']
            majority_classes = ['Serrated adenoma', 'Polyp', 'Normal', 'High-grade IN', 'Low-grade IN']
            cal_majority_frac = 0.6
            cal_minority_frac = 0.4
        elif config['dataset'] == 'ham10k':
            minority_classes = ['nv']
            majority_classes = ['akiec', 'bcc', 'bkl', 'df', 'vasc', 'mel']
            cal_majority_frac = 0.7
            cal_minority_frac = 0.3
        else:
            raise ValueError("Prevalence-shift setup only defined for 'ebhi' and 'ham10k'.")

        new_va_indices, new_ts_indices = get_adversarial_shuffle_indices(
            output_data,
            minority_classes,
            majority_classes,
            cal_majority_frac,
            cal_minority_frac,
        )

        fnames = get_fnames(data_dir, var_names, split_names, va_indices=new_va_indices, ts_indices=new_ts_indices)
        qr_fnames = get_fnames(qr_data_dir, qr_var_names, split_names, va_indices=new_va_indices, ts_indices=new_ts_indices)

        weights = precompute_wcp_weights(output_data, fnames, aux_model_type='lightgbm')
        qr_weights = precompute_wcp_weights(output_data, qr_fnames, aux_model_type='lightgbm')

    else:
        fnames = get_fnames(data_dir, var_names, split_names, shuffle=True)
        qr_fnames = get_fnames(qr_data_dir, qr_var_names, split_names, shuffle=True)

    classical_config = {'return_raw': False}

    logits_config = {
        'fnames': fnames,
        'model': model,
        'channel': config['channel'],
        'post_trans': post_trans,
        'device': config['device'],
        'alpha': config['alpha'],
        'beta_range': 100.0,
        'search_threshold': 1e-2,
        'max_search_steps': 1000,
        'return_raw': False,
    }

    # --- Baselines ---
    tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs = unpack_tensors2npy(fnames)

    print('Running baseline methods...')
    if classical_config['return_raw']:
        scp_len, scp_cov, scp_raw = SCP(va_vhats, va_vs, ts_vhats, ts_vs, alpha=config['alpha'], **classical_config)
        cqr_len, cqr_cov, cqr_raw = CQR(tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs, alpha=config['alpha'], **classical_config)
        local_len, local_cov, local_raw = Local(tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs, alpha=config['alpha'], **classical_config)
        e2e_len, e2e_cov, e2e_raw = E2E(qr_fnames, alpha=config['alpha'], **classical_config)

        with open(raw_dir / f"raw_{config['dataset']}_scp_{i}.pkl", 'wb') as f:
            pickle.dump([scp_raw, 0.0], f)
        with open(raw_dir / f"raw_{config['dataset']}_cqr_{i}.pkl", 'wb') as f:
            pickle.dump([cqr_raw, 0.0], f)
        with open(raw_dir / f"raw_{config['dataset']}_local_{i}.pkl", 'wb') as f:
            pickle.dump([local_raw, 0.0], f)
        with open(raw_dir / f"raw_{config['dataset']}_e2e_{i}.pkl", 'wb') as f:
            pickle.dump([e2e_raw, 0.0], f)

        del tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs

    else:
        scp_len, scp_cov = SCP(va_vhats, va_vs, ts_vhats, ts_vs, alpha=config['alpha'])
        cqr_len, cqr_cov = CQR(tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs, alpha=config['alpha'])
        local_len, local_cov = Local(tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs, alpha=config['alpha'])
        del tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs
        e2e_len, e2e_cov = E2E(qr_fnames, alpha=config['alpha'])

    run_data = [
        ['SCP', scp_len, scp_cov],
        ['Local', local_len, local_cov],
        ['CQR', cqr_len, cqr_cov],
        ['E2E-CQR', e2e_len, e2e_cov],
    ]

    # --- COMPASS-L (logits-only) ---
    print('Running COMPASS-L (logits-only) methods...')
    if logits_config['return_raw']:
        start = time.time()
        logits_len, logits_cov, logits_raw = Logits(sym=True, return_scores=False, **logits_config)
        logits_time = time.time() - start

        start = time.time()
        alogits_len, alogits_cov, alogits_raw = Logits(sym=False, return_scores=False, **logits_config)
        alogits_time = time.time() - start

        with open(raw_dir / f"raw_{config['dataset']}_COMPASS-L_{i}.pkl", 'wb') as f:
            pickle.dump([logits_raw, logits_time], f)
        with open(raw_dir / f"raw_{config['dataset']}_A-COMPASS-L_{i}.pkl", 'wb') as f:
            pickle.dump([alogits_raw, alogits_time], f)

    else:
        logits_len, logits_cov = Logits(sym=True, return_scores=False, **logits_config)
        alogits_len, alogits_cov = Logits(sym=False, return_scores=False, **logits_config)

    run_data.extend([
        ['COMPASS-L', logits_len, logits_cov],
        ['A-COMPASS-L', alogits_len, alogits_cov],
    ])

    # --- WCP variants (prevalence shift only) ---
    if config['shift'] == 'prevalence':
        print('Running WCP methods...')

        if classical_config['return_raw']:
            scpwcp_len, scpwcp_cov, scpwcp_raw = SCP_WCP(
                fnames, output_data, alpha=config['alpha'], weights=weights['label_shift'], **classical_config
            )
            cqrwcp_len, cqrwcp_cov, cqrwcp_raw = CQR_WCP(
                fnames, output_data, alpha=config['alpha'], weights=weights['label_shift'], **classical_config
            )
            localwcp_len, localwcp_cov, localwcp_raw = Local_WCP(
                fnames, output_data, alpha=config['alpha'], weights=weights['label_shift'], **classical_config
            )

            e2elabelwcp_len, e2elabelwcp_cov, e2elabelwcp_raw = E2E(
                qr_fnames,
                alpha=config['alpha'],
                weights=qr_weights['label_shift'],
                sym=True,
                **classical_config,
            )
            e2efeatwcp_len, e2efeatwcp_cov, e2efeatwcp_raw = E2E(
                qr_fnames,
                alpha=config['alpha'],
                weights=qr_weights['latent_shift'],
                sym=True,
                **classical_config,
            )

            with open(raw_dir / f"raw_{config['dataset']}_scp_{i}.pkl", 'wb') as f:
                pickle.dump([scpwcp_raw, 0.0], f)
            with open(raw_dir / f"raw_{config['dataset']}_cqr_{i}.pkl", 'wb') as f:
                pickle.dump([cqrwcp_raw, 0.0], f)
            with open(raw_dir / f"raw_{config['dataset']}_local_{i}.pkl", 'wb') as f:
                pickle.dump([localwcp_raw, 0.0], f)
            with open(raw_dir / f"raw_{config['dataset']}_e2elabel_{i}.pkl", 'wb') as f:
                pickle.dump([e2elabelwcp_raw, 0.0], f)
            with open(raw_dir / f"raw_{config['dataset']}_e2efeat_{i}.pkl", 'wb') as f:
                pickle.dump([e2efeatwcp_raw, 0.0], f)

        else:
            scpwcp_len, scpwcp_cov = SCP_WCP(fnames, output_data, alpha=config['alpha'], weights=weights['label_shift'])
            cqrwcp_len, cqrwcp_cov = CQR_WCP(fnames, output_data, alpha=config['alpha'], weights=weights['label_shift'])
            localwcp_len, localwcp_cov = Local_WCP(fnames, output_data, alpha=config['alpha'], weights=weights['label_shift'])
            e2elabelwcp_len, e2elabelwcp_cov = E2E(
                qr_fnames, alpha=config['alpha'], weights=qr_weights['label_shift'], sym=True
            )
            e2efeatwcp_len, e2efeatwcp_cov = E2E(
                qr_fnames, alpha=config['alpha'], weights=qr_weights['latent_shift'], sym=True
            )

        if logits_config['return_raw']:
            logitslabelwcp_len, logitslabelwcp_cov, logitslabelwcp_raw = Logits(
                sym=True,
                weights=weights['label_shift'],
                return_scores=False,
                **logits_config,
            )
            alogitslabelwcp_len, alogitslabelwcp_cov, alogitslabelwcp_raw = Logits(
                sym=False,
                weights=weights['label_shift'],
                return_scores=False,
                **logits_config,
            )

            logitsfeatwcp_len, logitsfeatwcp_cov, logitsfeatwcp_raw = Logits(
                sym=True,
                weights=weights['latent_shift'],
                return_scores=False,
                **logits_config,
            )
            alogitsfeatwcp_len, alogitsfeatwcp_cov, alogitsfeatwcp_raw = Logits(
                sym=False,
                weights=weights['latent_shift'],
                return_scores=False,
                **logits_config,
            )

            with open(raw_dir / f"raw_{config['dataset']}_COMPASS-L-labelwcp_{i}.pkl", 'wb') as f:
                pickle.dump([logitslabelwcp_raw, 0.0], f)
            with open(raw_dir / f"raw_{config['dataset']}_A-COMPASS-L-labelwcp_{i}.pkl", 'wb') as f:
                pickle.dump([alogitslabelwcp_raw, 0.0], f)
            with open(raw_dir / f"raw_{config['dataset']}_COMPASS-L-featwcp_{i}.pkl", 'wb') as f:
                pickle.dump([logitsfeatwcp_raw, 0.0], f)
            with open(raw_dir / f"raw_{config['dataset']}_A-COMPASS-L-featwcp_{i}.pkl", 'wb') as f:
                pickle.dump([alogitsfeatwcp_raw, 0.0], f)

        else:
            logitslabelwcp_len, logitslabelwcp_cov = Logits(
                sym=True, weights=weights['label_shift'], return_scores=False, **logits_config
            )
            alogitslabelwcp_len, alogitslabelwcp_cov = Logits(
                sym=False, weights=weights['label_shift'], return_scores=False, **logits_config
            )

            logitsfeatwcp_len, logitsfeatwcp_cov = Logits(
                sym=True, weights=weights['latent_shift'], return_scores=False, **logits_config
            )
            alogitsfeatwcp_len, alogitsfeatwcp_cov = Logits(
                sym=False, weights=weights['latent_shift'], return_scores=False, **logits_config
            )

        print('COMPASS-L-LabelWCP Len: ', logitslabelwcp_len, 'COMPASS-L-LabelWCP Cov:', logitslabelwcp_cov)
        print('A-COMPASS-L-LabelWCP Len: ', alogitslabelwcp_len, 'A-COMPASS-L-LabelWCP Cov:', alogitslabelwcp_cov)
        print('COMPASS-L-FeatWCP Len: ', logitsfeatwcp_len, 'COMPASS-L-FeatWCP Cov:', logitsfeatwcp_cov)
        print('A-COMPASS-L-FeatWCP Len: ', alogitsfeatwcp_len, 'A-COMPASS-L-FeatWCP Cov:', alogitsfeatwcp_cov)

        run_data.extend([
            ['SCP-WCP', scpwcp_len, scpwcp_cov],
            ['CQR-WCP', cqrwcp_len, cqrwcp_cov],
            ['Local-WCP', localwcp_len, localwcp_cov],
            ['E2E-LabelWCP', e2elabelwcp_len, e2elabelwcp_cov],
            ['E2E-FeatWCP', e2efeatwcp_len, e2efeatwcp_cov],
            ['COMPASS-L-LabelWCP', logitslabelwcp_len, logitslabelwcp_cov],
            ['A-COMPASS-L-LabelWCP', alogitslabelwcp_len, alogitslabelwcp_cov],
            ['COMPASS-L-FeatWCP', logitsfeatwcp_len, logitsfeatwcp_cov],
            ['A-COMPASS-L-FeatWCP', alogitsfeatwcp_len, alogitsfeatwcp_cov],
        ])

    df = pd.DataFrame(run_data, columns=['Method', 'Length', 'Coverage'])
    all_results.append(df)

    # df.to_csv(result_path, index=False)
    # print(f"Results for run {i+1} saved to {result_path}")
    print(pd.concat(all_results).groupby('Method').mean())
