"""2D helper utilities for caching variables and data transforms."""

import torch
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import Activations,AsDiscrete,Compose
from sklearn.decomposition import PCA
import os
import shutil
import glob
import random
from collections import Counter,defaultdict
import numpy as np
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time


def DataExtractionWrapperSave(loaders,root_dir,model,channel,forward_x,
                              forward_latent,compute_jacobian,
                              post_trans,post_trans_diff,device,
                              jacobians=True,full_jacobians=False):
    if os.path.exists(f'{root_dir}'):
        shutil.rmtree(f'{root_dir}')
    os.mkdir(f'{root_dir}')
    os.mkdir(f'{root_dir}vhat/')
    os.mkdir(f'{root_dir}v/')
    os.mkdir(f'{root_dir}yhat/')
    os.mkdir(f'{root_dir}seghat/')
    os.mkdir(f'{root_dir}seg/')
    os.mkdir(f'{root_dir}latent/')
    if jacobians:
        os.mkdir(f'{root_dir}J/')
        if full_jacobians:
            os.mkdir(f'{root_dir}Jfull/')
    for li,loader in enumerate(loaders):
        for i,(x,seg) in enumerate(loader):
            # print(li,i)
            # if os.path.exists(f'{root_dir}/seghat/{li}_{i}.pt')==False:
            x=x.to(device)
            seg=seg.to(device)
            latent=forward_x(model,x)
            if jacobians:
                yhat,vhat,J=compute_jacobian(model,latent,channel)
                Jsum=J.flatten(2).sum(2)
                seghat,_=post_trans(yhat,channel)
                v=seg.flatten(2).sum(2)
            else:
                yhat=forward_latent(model,latent)
                seghat,vhat=post_trans(yhat,channel)
                _,v=post_trans(seg,channel,sigmoid=False)
            torch.save(vhat.detach().cpu(),f'{root_dir}vhat/{li}_{i}.pt')
            torch.save(v.detach().cpu(),f'{root_dir}v/{li}_{i}.pt')
            torch.save(yhat.detach().cpu(),f'{root_dir}yhat/{li}_{i}.pt')
            torch.save(seghat.detach().cpu(),f'{root_dir}seghat/{li}_{i}.pt')
            torch.save(seg.detach().cpu(),f'{root_dir}seg/{li}_{i}.pt')
            torch.save(latent.detach().cpu(),f'{root_dir}latent/{li}_{i}.pt')
            if jacobians:
                torch.save(Jsum.detach().cpu(),f'{root_dir}J/{li}_{i}.pt')
                if full_jacobians:
                    torch.save(J.detach().cpu(),f'{root_dir}Jfull/{li}_{i}.pt')

def QRDataExtractionWrapperSave(loaders,root_dir,model,forward_x,forward_latent,compute_jacobian,post_trans,post_trans_diff,device,metric='mask_sum'):
    if os.path.exists(f'{root_dir}'):
        shutil.rmtree(f'{root_dir}')
    os.mkdir(f'{root_dir}')
    os.mkdir(f'{root_dir}vhat/')
    os.mkdir(f'{root_dir}v/')
    os.mkdir(f'{root_dir}seghat/')
    os.mkdir(f'{root_dir}seg/')
    os.mkdir(f'{root_dir}latent/')
    # os.mkdir(f'{root_dir}J/')
    for li,loader in enumerate(loaders):
        for i,(x,seg) in enumerate(loader):
            # print(li,i)
            # if os.path.exists(f'{root_dir}/seghat/{li}_{i}.pt')==False:
            x=x.to(device)
            seg=seg.to(device)
            # yhat=model(x)
            latent=forward_x(model,x)
            # Jsum=J.flatten(2).sum(2)
            yhat=forward_latent(model,latent)
            # yhat,_,_=compute_jacobian(model,latent,1)
            # yhat,_,_=compute_jacobian(model,latent,2)
            seghat0,vhat0=post_trans(yhat,0)
            seghat1,vhat1=post_trans(yhat,1)
            seghat2,vhat2=post_trans(yhat,2)
            seghat=torch.stack([seghat0,seghat1,seghat2],dim=1)
            vhat=torch.stack([vhat0,vhat1,vhat2],dim=1)
            if metric == 'mask_sum':
                v=seg.flatten(2).sum(2)
            else:
                raise ValueError(f"Unsupported metric {metric!r}. Supported: 'mask_sum'.")
            torch.save(vhat.detach().cpu(),f'{root_dir}vhat/{li}_{i}.pt')
            torch.save(v.detach().cpu(),f'{root_dir}v/{li}_{i}.pt')
            torch.save(seghat.detach().cpu(),f'{root_dir}seghat/{li}_{i}.pt')
            torch.save(seg.detach().cpu(),f'{root_dir}seg/{li}_{i}.pt')
            torch.save(latent.detach().cpu(),f'{root_dir}latent/{li}_{i}.pt')
            # torch.save(Jsum.detach().cpu(),f'{root_dir}J/{li}_{i}.pt')

def TimeJacobians(loaders,root_dir,model,channel,forward_x,forward_latent,compute_jacobian,
                              post_trans,post_trans_diff,device,jacobians=True):
    start=time.time()
    for li,loader in enumerate(loaders):
        for i,(x,seg) in enumerate(loader):
            x=x.to(device)
            seg=seg.to(device)
            latent=forward_x(model,x)
            if jacobians:
                yhat,vhat,J=compute_jacobian(model,latent,channel)
                Jsum=J.flatten(2).sum(2)
                seghat,_=post_trans_diff(yhat,channel)
                v=seg.flatten(2).sum(2)
    end=time.time()
    elapsed=end-start
    return elapsed
    
def get_fnames(data_dir,var_names,split_names,shuffle=False,va_indices=None,ts_indices=None):
    def _sort_fnames_numerically(file_list):
        """Sorts a list of file paths based on the integer index in the filename."""
        return sorted(file_list, key=lambda f: int(os.path.basename(f).split('_')[-1].replace('.pt', '')))
    # get filenames of all variables
    fnames={}
    for si,split_name in enumerate(split_names):
        fnames[split_name]={}
        for vi,var_name in enumerate(var_names):
            pattern = os.path.join(str(data_dir), var_name, f"{si}_*")
            fnames[split_name][var_name] = _sort_fnames_numerically(glob.glob(pattern))
    if (va_indices is not None)&(ts_indices is not None):
        for vi,var_name in enumerate(var_names):
            cat_vhat_fnames=fnames['va'][var_name]+fnames['ts'][var_name]
            va_fnames=[cat_vhat_fnames[i] for i in va_indices]
            ts_fnames=[cat_vhat_fnames[i] for i in ts_indices]
            fnames['va'][var_name]=va_fnames
            fnames['ts'][var_name]=ts_fnames
    elif shuffle:
        # shuffle all variables
        len_va=len(fnames['va']['vhat'])
        len_ts=len(fnames['ts']['vhat'])
        indices=[i for i in range(len_va+len_ts)]
        random.shuffle(indices)
        va_indices=indices[:len_va]
        ts_indices=indices[len_va:]
        for vi,var_name in enumerate(var_names):
            cat_vhat_fnames=fnames['va'][var_name]+fnames['ts'][var_name]
            va_fnames=[cat_vhat_fnames[i] for i in va_indices]
            ts_fnames=[cat_vhat_fnames[i] for i in ts_indices]
            fnames['va'][var_name]=va_fnames
            fnames['ts'][var_name]=ts_fnames
    return fnames

def load_var(fnames,split_name,var_name):
    paths = fnames.get(split_name, {}).get(var_name, [])
    if not paths:
        raise FileNotFoundError(
            f"No cached tensors found for split={split_name!r}, var={var_name!r}. "
            "This usually means variable extraction did not run, produced no outputs, or is pointing at the wrong data directory. "
            "Fix: set `config['generate_vars']=True` and re-run (or verify `<DATASET_ROOT>/data/<MODEL_NAME>/<METRIC>/`)."
        )
    return torch.vstack([torch.load(p, weights_only=False) for p in paths])

def load_var_instance(fnames,split_name,var_name,i):
    paths = fnames.get(split_name, {}).get(var_name, [])
    if not paths:
        raise FileNotFoundError(
            f"No cached tensors found for split={split_name!r}, var={var_name!r}. "
            "Fix: set `config['generate_vars']=True` and re-run (or verify cached variable paths)."
        )
    return torch.load(paths[i], weights_only=False).squeeze(0)

def get_stratified_shuffle_indices(output_data):
    """
    Performs a stratified shuffle on the calibration and test sets and returns the new indices.

    This function combines the calibration and test sets, shuffles the samples
    within each class, and then re-splits them to create new index lists. This ensures
    that the class distribution in the new calibration and test sets is identical
    to the original distribution.

    Args:
        output_data (dict): A dictionary containing 'train_classes', 'cal_classes', and 'test_classes'.

    Returns:
        tuple: A tuple containing:
            - new_cal_indices (list): The list of shuffled original pool indices
                                      for the new calibration set.
            - new_test_indices (list): The list of shuffled original pool indices
                                       for the new test set.
    """
    train_classes = output_data.get('train_classes', [])
    cal_classes = output_data.get('cal_classes', [])
    test_classes = output_data.get('test_classes', [])

    if not cal_classes and not test_classes:
        print("Warning: Calibration and test sets are empty. Returning empty indices.")
        return [], []

    # --- 1. Combine into a single pool and get original counts ---
    combined_classes = cal_classes + test_classes
    original_cal_size = len(cal_classes)
    original_cal_class_counts = Counter(cal_classes)
    original_test_class_counts = Counter(test_classes)

    # --- 2. Group indices by class from the combined pool ---
    indices_by_class = defaultdict(list)
    for i, class_label in enumerate(combined_classes):
        indices_by_class[class_label].append(i)

    # --- 3. Shuffle indices within each class group ---
    for class_label in indices_by_class:
        random.shuffle(indices_by_class[class_label])

    # --- 4. Reconstruct the calibration and test sets (stratified) ---
    new_cal_indices = []
    new_test_indices = []

    for class_label, shuffled_indices in indices_by_class.items():
        # Determine how many samples of this class belong in the calibration set
        num_cal_samples = original_cal_class_counts.get(class_label, 0)
        
        # Allocate the indices
        new_cal_indices.extend(shuffled_indices[:num_cal_samples])
        new_test_indices.extend(shuffled_indices[num_cal_samples:])

    # --- 5. Final shuffle to mix the order of classes within the lists ---
    random.shuffle(new_cal_indices)
    random.shuffle(new_test_indices)
    
    # --- 6. Verification and Printout ---
    assert len(new_cal_indices) == original_cal_size
    assert len(new_cal_indices) + len(new_test_indices) == len(combined_classes)
    
    # Helper function for pretty printing distributions
    def _print_dist_summary(name, labels, all_class_names):
        counts = Counter(labels)
        total = len(labels)
        print(f"\n{name} Distribution ({total} samples):")
        for cls in sorted(all_class_names):
            count = counts.get(cls, 0)
            percent = (count / total) * 100 if total > 0 else 0
            print(f"  - {cls:<20}: {count:>4} samples ({percent:5.1f}%)")

    # Determine all unique classes across all sets for consistent printing
    all_unique_classes = set(train_classes + cal_classes + test_classes)
    
    # Determine the new class distributions for printing
    new_cal_classes = [combined_classes[i] for i in new_cal_indices]
    new_test_classes = [combined_classes[i] for i in new_test_indices]
    
    print("\n--- DATASET DISTRIBUTIONS ---")
    _print_dist_summary("Training Set", train_classes, all_unique_classes)
    _print_dist_summary("Original Calibration Set", cal_classes, all_unique_classes)
    _print_dist_summary("Original Test Set", test_classes, all_unique_classes)
    print("\n--- STRATIFIED SHUFFLE COMPLETE ---")
    _print_dist_summary("New Shuffled Calibration Set", new_cal_classes, all_unique_classes)
    _print_dist_summary("New Shuffled Test Set", new_test_classes, all_unique_classes)
    print("-" * 40)
    
    # Verify that the class counts are identical after the shuffle
    assert Counter(cal_classes) == Counter(new_cal_classes)
    assert Counter(test_classes) == Counter(new_test_classes)

    return new_cal_indices, new_test_indices

def get_shuffled_output_data(output_data, cal_indices, test_indices):
    """
    Creates a new output_data dictionary based on shuffled indices.
    """
    # Combine original cal/test data into pools
    cal_images, cal_labels, cal_classes = output_data['cal_images'], output_data['cal_labels'], output_data['cal_classes']
    test_images, test_labels, test_classes = output_data['test_images'], output_data['test_labels'], output_data['test_classes']
    
    image_pool = cal_images + test_images
    label_pool = cal_labels + test_classes
    class_pool = cal_classes + test_classes

    # Reconstruct the sets using the new indices
    new_cal_images = [image_pool[i] for i in cal_indices]
    new_cal_labels = [label_pool[i] for i in cal_indices]
    new_cal_classes = [class_pool[i] for i in cal_indices]
    
    new_test_images = [image_pool[i] for i in test_indices]
    new_test_labels = [label_pool[i] for i in test_indices]
    new_test_classes = [class_pool[i] for i in test_indices]
    
    # Construct the new dictionary
    shuffled_output_data = {
        "train_images": output_data['train_images'], "train_labels": output_data['train_labels'], "train_classes": output_data['train_classes'],
        "cal_images": new_cal_images, "cal_labels": new_cal_labels, "cal_classes": new_cal_classes,
        "test_images": new_test_images, "test_labels": new_test_labels, "test_classes": new_test_classes,
    }
    return shuffled_output_data


def get_adversarial_shuffle_indices(output_data, minority_classes,majority_classes,
                                    cal_majority_frac,cal_minority_frac):
    """
    Creates an adversarial split from a data pool and returns the indices.

    This function is designed to be called inside an experimental loop. It takes the
    full calibration/test data pool, shuffles it, and then constructs a new split where:
    - The calibration set is systematically "easier" (majority-class heavy).
    - The test set is systematically "harder" (minority-class heavy).

    Args:
        output_data (dict): The dictionary containing 'train_classes', 'cal_classes', and 'test_classes'.
        minority_classes (list): A list of minority class names.
        majority_classes (list): A list of majority class names.
        

    Returns:
        tuple: A tuple containing:
            - new_cal_indices (list): The list of original pool indices for the new "easy" calibration set.
            - new_test_indices (list): The list of original pool indices for the new "hard" test set.
    """
    train_classes = output_data.get('train_classes', [])
    cal_classes = output_data.get('cal_classes', [])
    test_classes = output_data.get('test_classes', [])
    
    # --- 1. Combine cal and test into a single pool with original indices ---
    combined_classes = cal_classes + test_classes
    pool_with_indices = list(enumerate(combined_classes))
    
    random.shuffle(pool_with_indices)

    # --- 2. Separate the shuffled pool by class type ---
    minority_pool_indexed = []
    majority_pool_indexed = []
    for index, label in pool_with_indices:
        if label in minority_classes:
            minority_pool_indexed.append((index, label))
        elif label in majority_classes:
            majority_pool_indexed.append((index, label))

    # # --- 3. Define the adversarial composition ---
    # cal_majority_frac = 0.8  # Calibration set gets x% of the majority samples
    # cal_minority_frac = 0.2 # Calibration set gets only x% of the minority samples
    
    # --- 4. Construct the "easy" calibration set indices ---
    n_cal_maj = int(len(majority_pool_indexed) * cal_majority_frac)
    n_cal_min = int(len(minority_pool_indexed) * cal_minority_frac)
    cal_set_indexed = majority_pool_indexed[:n_cal_maj] + minority_pool_indexed[:n_cal_min]
    
    # --- 5. Construct the "hard" test set indices with the remainder ---
    test_set_indexed = majority_pool_indexed[n_cal_maj:] + minority_pool_indexed[n_cal_min:]

    # Extract just the original indices from the pool
    new_cal_indices = [item[0] for item in cal_set_indexed]
    new_test_indices = [item[0] for item in test_set_indexed]
    
    random.shuffle(new_cal_indices)
    random.shuffle(new_test_indices)

    # --- 6. Verification and Printout ---
    all_unique_classes = set(train_classes + cal_classes + test_classes)
    new_cal_classes = [combined_classes[i] for i in new_cal_indices]
    new_test_classes = [combined_classes[i] for i in new_test_indices]
    
    print("\n--- ADVERSARIAL SHUFFLE COMPLETE ---")
    _print_dist_summary("Training Set (Fixed)", train_classes, all_unique_classes)
    _print_dist_summary("New 'Easy' Calibration Set", new_cal_classes, all_unique_classes)
    _print_dist_summary("New 'Hard' Test Set", new_test_classes, all_unique_classes)
    print("-" * 40)

    return new_cal_indices, new_test_indices

def _print_dist_summary(name, labels, all_class_names):
    """Prints a formatted summary of class distributions."""
    counts = Counter(labels)
    total = len(labels)
    print(f"\n{name} Distribution ({total} samples):")
    # Sort the class names to ensure consistent print order
    for cls in sorted(all_class_names):
        count = counts.get(cls, 0)
        percent = (count / total) * 100 if total > 0 else 0
        print(f"  - {cls:<20}: {count:>4} samples ({percent:5.1f}%)")

def precompute_wcp_weights(output_data, fnames, aux_model_type='lightgbm'):
    """
    Pre-computes and returns a dictionary of all applicable WCP weights for a given data split.

    Args:
        output_data (dict): Dictionary containing class labels for all splits.
        fnames (dict): fnames dictionary for either the main or the E2E model.
        aux_model_type (str): The type of auxiliary classifier to use ('lightgbm' or 'logistic_regression').
    """
    print(f"\n--- Pre-computing WCP weights using {aux_model_type.upper()} ---")
    all_weights = {}
    epsilon = 1e-9
    weight_clip_max = 50

    # --- 1. Label Shift Weights (Always Computed) ---
    print("  - Computing 'label_shift' weights...")
    cal_labels = output_data.get('cal_classes', [])
    ts_labels = output_data.get('test_classes', [])
    if cal_labels and ts_labels:
        cal_counts, test_counts = Counter(cal_labels), Counter(ts_labels)
        P_cal = {k: v / len(cal_labels) for k, v in cal_counts.items()}
        P_test = {k: v / len(ts_labels) for k, v in test_counts.items()}
        weights = np.array([P_test.get(lbl, 0) / P_cal.get(lbl, epsilon) for lbl in cal_labels])
        print('Min Weight',weights.min(),'Max Weight',weights.max())
        all_weights['label_shift'] = weights

    # --- Function to train the selected auxiliary classifier ---
    def train_aux_classifier(X_cal, X_test, shift_type):
        X_aux = np.vstack([X_cal, X_test])
        y_aux = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])
        
        classifier = None
        
        if aux_model_type == 'logistic_regression':
            print(f"    - Scaling features for Logistic Regression ({shift_type})...")
            scaler = StandardScaler()
            X_aux_scaled = scaler.fit_transform(X_aux)
            
            X_cal_scaled = scaler.transform(X_cal)
            X_test_scaled = scaler.transform(X_test)
            
            classifier = LogisticRegression(random_state=42,C=1.0,n_jobs=32,verbose=1)
            classifier.fit(X_aux_scaled, y_aux)
            
            # --- Sanity Check ---
            # test_acc = classifier.score(X_test_scaled, np.ones(len(X_test)))
            # print(f"    - Sanity Check: Aux classifier accuracy on TEST data ({shift_type}): {test_acc:.2%}")

            return classifier.predict_proba(X_cal_scaled)[:, 1]
        
        elif aux_model_type == 'lightgbm':
            classifier=LGBMClassifier(n_estimators=100,n_jobs=-1,verbosity=-1).fit(X_aux,y_aux)
            
            # --- Sanity Check ---
            # test_acc = classifier.score(X_test, np.ones(len(X_test)))
            # print(f"    - Sanity Check: Aux classifier accuracy on TEST data ({shift_type}): {test_acc:.2%}")
            
            return classifier.predict_proba(X_cal)[:, 1]
        
        else:
            raise ValueError(f"Unknown aux_model_type: '{aux_model_type}'")

    # --- 2. Latent Shift Weights ---
    if 'latent' in fnames.get('va', {}):
        print("  - Computing 'latent_shift' weights...")
        X_cal = load_var(fnames, 'va', 'latent').flatten(2).sum(2)
        X_test = load_var(fnames, 'ts', 'latent').flatten(2).sum(2)
        p_cal = train_aux_classifier(X_cal, X_test, shift_type='latent_shift')
        all_weights['latent_shift'] = p_cal / (1 - p_cal + epsilon)
        del X_cal,X_test

    # --- 3. Jacobian Shift Weights ---
    if 'J' in fnames.get('va', {}):
        print("  - Computing 'jacobian_shift' weights...")
        X_cal_jac = load_var(fnames, 'va', 'J')
        X_test_jac = load_var(fnames, 'ts', 'J')
        p_cal_jac = train_aux_classifier(X_cal_jac, X_test_jac, shift_type='jacobian_shift')
        all_weights['jacobian_shift'] = p_cal_jac / (1 - p_cal_jac + epsilon)
        del X_cal_jac,X_test_jac
    
    # --- 4. E2E Output Shift Weights ---
    if 'vhat' in fnames.get('va', {}):
        va_vhats = load_var(fnames, 'va', 'vhat')
        if va_vhats is not None and va_vhats.ndim == 2 and va_vhats.shape[1] == 3:
            print("  - Computing 'e2e_output_shift' weights...")
            ts_vhats = load_var(fnames, 'ts', 'vhat')
            
            va_lo, va_hi = va_vhats[:, 1].ravel(), va_vhats[:, 2].ravel()
            ts_lo, ts_hi = ts_vhats[:, 1].ravel(), ts_vhats[:, 2].ravel()
            
            X_cal_e2e = np.stack([va_lo, va_hi, va_hi - va_lo], axis=1)
            X_test_e2e = np.stack([ts_lo, ts_hi, ts_hi - ts_lo], axis=1)
            
            p_cal_e2e = train_aux_classifier(X_cal_e2e, X_test_e2e, shift_type='e2e_output_shift')
            all_weights['e2e_output_shift'] = p_cal_e2e / (1 - p_cal_e2e + epsilon)

    print("--- Weight computation complete. ---")
    return all_weights
