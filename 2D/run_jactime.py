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
from compass2d.compass import *  # noqa: E402
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
    'device': 'cuda:1',
    'device2': 'cuda:2',
    'dataset': 'ham10k',  # Options: 'ebhi', 'ham10k', 'kvasir', 'tn3k'
    'arch': 'UNet',     # Options: 'UNet', 'SegResNet'
    'metric': 'mask_sum',    # Scalar target (default): sum of mask pixels
    'generate_vars': False, # Set to False to skip data extraction if already done
    'overwrite': False,
    'evaluation_runs': 100, # Number of times to run the evaluation loop
    'shift':'',

    # --- General Parameters ---
    'size': 128,
    'num_workers': 64,
    'alpha': 0.1,
    'channel': 0,
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
    'ebhi': ebhi,
    'ham10k': ham10k,
    'kvasir': kvasir,
    'tn3k': tn3k,
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
        'mask_sum': (post_trans_mask_sum, post_trans_mask_sum_diff),
    }
    return transform_map.get(metric_name, (None, None))

def get_dataset(dataset_name,size,root_dir=None,fnames=None):
    """Loads the correct dataset splits based on the dataset name."""
    loader_class = dataset_loaders.get(dataset_name)
    if not loader_class:
        raise ValueError(
            f"Dataset '{dataset_name}' not recognized. "
            f"Supported: {sorted(dataset_loaders.keys())}"
        )
    if fnames is not None:
        # specify img and seg instance paths
        train_set = loader_class(img_paths=fnames['train_images'], seg_paths=fnames['train_labels'], size=size, transform=True)
        val_set = loader_class(img_paths=fnames['cal_images'], seg_paths=fnames['cal_labels'], size=size, transform=False)
        test_set = loader_class(img_paths=fnames['test_images'], seg_paths=fnames['test_labels'], size=size, transform=False)
    else:
        train_set = loader_class(img_dir=f'{root_dir}imagesTr/', seg_dir=f'{root_dir}labelsTr/', size=size, transform=True)
        val_set = loader_class(img_dir=f'{root_dir}imagesVa/', seg_dir=f'{root_dir}labelsVa/', size=size, transform=False)
        test_set = loader_class(img_dir=f'{root_dir}imagesTs/', seg_dir=f'{root_dir}labelsTs/', size=size, transform=False)
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
dataset_name = config['dataset']
DATASET_ROOTS = {
    "ebhi": "/scratch/yc130/EBHI-SEG/",
    "ham10k": "/scratch/yc130/ham10k/",
    "kvasir": "/scratch/yc130/Kvasir-SEG/",
    "tn3k": "/scratch/yc130/Thyroid/tn3k/",
}
if dataset_name not in DATASET_ROOTS:
    raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {sorted(DATASET_ROOTS.keys())}")
root_dir = DATASET_ROOTS[dataset_name]


# --- Data Loading ---
if config['shift']=='prevalence':
    shift = config["shift"]
    with open(os.path.join(root_dir, f"{shift}_fnames.pkl"), "rb") as f:
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

# --- Feature and Jacobian Extraction ---
forward_x, forward_latent, compute_jacobian = get_fns(arch, post_trans_diff)
if arch == 'UNet':
    forward_x_shallow, forward_latent_shallow, compute_jacobian_shallow = get_fns(arch + 'Shallow', post_trans_diff)

data_dir = os.path.join(root_dir, 'data', model_name, config['metric']+os.sep)
qr_data_dir = os.path.join(root_dir, 'data', qr_model_name, config['metric']+os.sep)

times=[]
reps=10
for i in range(reps):
    deep_time=TimeJacobians(loaders, data_dir, model, config['channel'],
                              forward_x, forward_latent, compute_jacobian, 
                              post_trans, post_trans_diff, config['device'])
    times.append(deep_time)
    print(f'Iteration {i} out of {reps}\t Time: {deep_time:.2f}')
out_dir = get_results_dir("2D", "timing")
np.save(out_dir / f"{config['dataset']}_deeptime.npy", np.array(times))
