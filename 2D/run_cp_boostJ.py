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
    'device': 'cuda:3',
    'device2': 'cuda:4',
    'dataset': 'ebhi',  # Options: 'ebhi', 'ham10k', 'kvasir', 'tn3k'
    'arch': 'UNet',     # Options: 'UNet', 'SegResNet'
    'metric': 'mask_sum',    # Scalar target (default): sum of mask pixels
    'generate_vars': False, # Set to False to skip data extraction if already done
    'overwrite': True,
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
forward_tologits, forward_logits, compute_jacobian_logits = get_fns(arch+'Logits', post_trans_diff)
data_dir = os.path.join(root_dir, 'data', model_name, config['metric']+os.sep)
qr_data_dir = os.path.join(root_dir, 'data', qr_model_name, config['metric']+os.sep)

if config['generate_vars']:
    print("Extracting variables for standard model...")
    os.makedirs(data_dir, exist_ok=True)
    DataExtractionWrapperSave(loaders, data_dir, model, config['channel'],
                              forward_x, forward_latent, compute_jacobian, post_trans,
                              post_trans_diff, config['device'],full_jacobians=True)
    
    print("Extracting variables for QR model...")
    os.makedirs(qr_data_dir, exist_ok=True)
    QRDataExtractionWrapperSave(loaders, qr_data_dir, qr_model,
                                forward_x, forward_latent, compute_jacobian, post_trans,
                                post_trans_diff, config['device2'])

# --- Evaluation Loop ---
var_names = ['vhat', 'v', 'J', 'latent', 'yhat', 'seghat','Jfull']
qr_var_names = ['vhat', 'v','latent']
split_names = ['tr', 'va', 'ts']

fnames = get_fnames(data_dir, var_names, split_names)
qr_fnames = get_fnames(qr_data_dir, qr_var_names, split_names)

def _missing_cached_vars(fnames_dict, *, required_splits, required_vars):
    missing = []
    for split in required_splits:
        for var in required_vars:
            if len(fnames_dict.get(split, {}).get(var, [])) == 0:
                missing.append((split, var))
    return missing

missing = _missing_cached_vars(fnames, required_splits=split_names, required_vars=var_names)
missing_qr = _missing_cached_vars(qr_fnames, required_splits=split_names, required_vars=qr_var_names)
if missing or missing_qr:
    if not config["generate_vars"]:
        print(
            "Cached variables not found (or empty). Re-running variable extraction now. "
            "Set `config['generate_vars']=True` to always regenerate."
        )
        os.makedirs(data_dir, exist_ok=True)
        DataExtractionWrapperSave(
            loaders,
            data_dir,
            model,
            config["channel"],
            forward_x,
            forward_latent,
            compute_jacobian,
            post_trans,
            post_trans_diff,
            config["device"],
        )

        os.makedirs(qr_data_dir, exist_ok=True)
        QRDataExtractionWrapperSave(
            loaders,
            qr_data_dir,
            qr_model,
            forward_x,
            forward_latent,
            compute_jacobian,
            post_trans,
            post_trans_diff,
            config["device2"],
            metric=config["metric"],
        )

        fnames = get_fnames(data_dir, var_names, split_names)
        qr_fnames = get_fnames(qr_data_dir, qr_var_names, split_names)
        missing = _missing_cached_vars(fnames, required_splits=split_names, required_vars=var_names)
        missing_qr = _missing_cached_vars(qr_fnames, required_splits=split_names, required_vars=qr_var_names)

    if missing or missing_qr:
        details = []
        if missing:
            details.append(f"standard model cache: {data_dir}")
            details.extend([f"  - missing {s}/{v}" for s, v in missing])
        if missing_qr:
            details.append(f"QR model cache: {qr_data_dir}")
            details.extend([f"  - missing {s}/{v}" for s, v in missing_qr])
        raise FileNotFoundError(
            "Cached variable tensors are missing or empty.\n"
            + "\n".join(details)
            + "\n"
            "Verify your dataset splits are non-empty and that cached variables exist."
        )
    

results_dir = get_results_dir(
    "2D", dataset_name, config["metric"], f"{arch}_results3_{config['alpha']}"
)
# os.makedirs(raw_dir, exist_ok=True)

all_results = []
# for i in range(config['evaluation_runs']):
for i in range(0,50,1):
    result_path = os.path.join(results_dir, f'df{i}.csv')
    #### check if results already exist
    # if os.path.exists(result_path) and not config['overwrite']:
    #     print(f"Results for run {i} already exist. Skipping.")
    #     continue

    # ### check if raw data missing
    # methods=['scp','local','cqr','e2e','logits', 'alogits', 'compassj', 'acompassj', 
    #          'compassjshallow', 'acompassjshallow']
    # exists_all=[]
    # for method in methods:
    #     if os.path.exists(f'{raw_dir}raw_{config['dataset']}_{method}_{i}.pkl')==False:
    #         print(f'{raw_dir}raw_{config['dataset']}_{method}_{i}.pkl')
    #         exists_all.append(False)
    #     else:
    #         exists_all.append(True)
    # if np.all(exists_all):
    #     print(f'Checked {i}, and all exists')
    #     continue
    
    print(f"\n--- Starting Evaluation Run {i+1}/{config['evaluation_runs']} ---")
    
    fnames=get_fnames(data_dir,var_names,split_names,shuffle=True)
    qr_fnames=get_fnames(qr_data_dir,qr_var_names,split_names,shuffle=True)

    classical_config={'return_raw':False}

    compass_config={'fnames':fnames,'model':model,'channel':config['channel'],
                    'forward_func':forward_latent,'post_trans':post_trans,
                    'device':config['device'],'n_components':1,'alpha':config['alpha'],
                    'beta_range': 1e2,'search_threshold':1e-10,'max_search_steps':1000,
                    'return_scores':False,'return_raw':False} 
        
    # Unpack data for this run
    tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs = unpack_tensors2npy(fnames)
    
    # --- Run Conformal Methods ---
    print("Running baseline methods...")
    scp_len, scp_cov = SCP(va_vhats, va_vs, ts_vhats, ts_vs, alpha=config['alpha'])
    cqr_len, cqr_cov = CQR(tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs, alpha=config['alpha'])
    local_len, local_cov = Local(tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs, alpha=config['alpha'])
    del tr_vhats, tr_vs, va_vhats, va_vs, ts_vhats, ts_vs
    e2e_len, e2e_cov = E2E(qr_fnames,alpha=config['alpha'])
    run_data = [['SCP', scp_len, scp_cov], ['Local', local_len, local_cov],
                ['CQR', cqr_len, cqr_cov], ['E2E-CQR', e2e_len, e2e_cov]]

    print("Running COMPASS methods...")
    compass_config['forward_func']=forward_logits
    compass_config['search_threshold']=1e-1
    compass_len,compass_cov=COMPASS(sym=True,logits=True,method='uniform',**compass_config)
    acompass_len,acompass_cov=COMPASS(sym=False,logits=True,method='uniform',**compass_config)
    print(f'COMPASS-L: {compass_len:.2f} {compass_cov:.2f}')
    print(f'A-COMPASS-L: {acompass_len:.2f} {acompass_cov:.2f}')
    run_data.extend([
                    [f'COMPASS-L',compass_len,compass_cov],
                     [f'A-COMPASS-L',acompass_len,acompass_cov]])

    # compassl_config={'fnames':fnames,'model':model,'channel':config['channel'],
    #                'post_trans':post_trans,'device':config['device'],'alpha':config['alpha'],
    #                'beta_range':1e8,'search_threshold':1e-8,
    #                  'max_search_steps':1000,'return_raw':False,
    #                 'spatial':None}
    # compassl_len,compassl_cov=COMPASSL(sym=True,return_scores=False,**compassl_config)
    # acompassl_len,acompassl_cov=COMPASSL(sym=False,return_scores=False,**compassl_config)
    # run_data.extend([['COMPASS-L',compassl_len,compassl_cov],
    #                  ['A-COMPASS-L',acompassl_len,acompassl_cov]])
    
    
    # logits_config={'fnames':fnames,'model':model,'channel':config['channel'],
    #                'post_trans':post_trans,'device':config['device'],'alpha':config['alpha'],
    #                'beta_range':1e5,'search_threshold':1e-8,'max_search_steps':1000,'return_raw':False}
    
    # logits_len,logits_cov=Logits(sym=True,return_scores=False,**logits_config)
    # alogits_len,alogits_cov=Logits(sym=False,return_scores=False,**logits_config)
    # run_data.extend([['COMPASS-L',logits_len,logits_cov],['A-COMPASS-L',alogits_len,alogits_cov]])
    

    
    methods=[
            # 'fourier',
             # 'pca_uncertainty_weighted',
             # 'pca_activation_weighted',
             # 'pca_spatial',
             'pca',
             # 'median',
             'mean',
             # 'gaussian',
             # 'uniform'
            ]
    # compass_config['post_trans']=post_trans_diff
    compass_config['forward_func']=forward_latent
    compass_config['search_threshold']=1e-10
    compass_config['beta_range']=1e5
    for method in methods:
        if method=='gaussian':
            compass_config['beta_range']=1e10
        elif method=='jacobian':
            compass_config['beta_range']=1e30
        else:
            compass_config['beta_range']=1e5
        compass_len,compass_cov=COMPASS(sym=True,method=method,**compass_config)
        acompass_len,acompass_cov=COMPASS(sym=False,method=method,**compass_config)
        print(f'COMPASS-{method}: {compass_len:.2f} {compass_cov:.2f}')
        print(f'A-COMPASS-{method}: {acompass_len:.2f} {acompass_cov:.2f}')
        run_data.extend([
                         [f'COMPASS-{method}',compass_len,compass_cov],
                         [f'A-COMPASS-{method}',acompass_len,acompass_cov]])

    # # Save results for this run
    df = pd.DataFrame(run_data, columns=['Method', 'Length', 'Coverage'])
    # # df['Run'] = i
    all_results.append(df)
    # df.to_csv(result_path, index=False)
    # print(f"Results for run {i+1} saved to {result_path}")
    print(pd.concat(all_results).groupby('Method').mean().sort_values(by='Length'))

# # Combine and save all results
# if all_results:
#     final_df = pd.concat(all_results, ignore_index=True)
#     final_path = os.path.join(results_dir, 'all_runs_summary.csv')
#     final_df.to_csv(final_path, index=False)
#     print(f"\nAll results saved to {final_path}")
