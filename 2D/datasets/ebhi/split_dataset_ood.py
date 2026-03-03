import os
import numpy as np
import shutil
import random
import pickle

# --- Path Setup ---
base_path='/scratch/yc130/EBHI-SEG/'
# The following two paths are not used in the provided logic but kept for context
# img_path=f'{base_path}imgs/'
# seg_path=f'{base_path}segs/'

imagesTr_path=f'{base_path}imagesTr/'
imagesVa_path=f'{base_path}imagesVa/'
imagesTs_path=f'{base_path}imagesTs/'
labelsTr_path=f'{base_path}labelsTr/'
labelsVa_path=f'{base_path}labelsVa/'
labelsTs_path=f'{base_path}labelsTs/'

# --- Directory Cleanup ---
dir_list=[imagesTr_path,imagesVa_path,imagesTs_path,
          labelsTr_path,labelsVa_path,labelsTs_path]
for dir_path in dir_list:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

# --- Data Splitting Logic ---
ood='standard'
# ood='test'

# Helper function to find paired files and return sorted lists of full paths
def find_paired_files(img_files, seg_files):
    """
    Identifies image and segmentation files that have a matching basename.
    Returns two sorted lists containing the full paths of the paired files.
    """
    # Create a set of basenames for faster lookup
    img_basenames = {os.path.basename(f) for f in img_files}
    seg_basenames = {os.path.basename(f) for f in seg_files}
    
    # Find the intersection of the basenames
    paired_basenames = img_basenames.intersection(seg_basenames)
    
    # Filter the original full-path lists to keep only paired files
    paired_img_files = sorted([f for f in img_files if os.path.basename(f) in paired_basenames])
    paired_seg_files = sorted([f for f in seg_files if os.path.basename(f) in paired_basenames])
    
    return paired_img_files, paired_seg_files

# --- Standard Split Configuration ---
if ood=='standard':
    tr_classes=['High-grade IN','Low-grade IN','Normal','Polyp','Serrated adenoma']
    va_ts_classes=['Adenocarcinoma']
    
    tr_img_files=[]
    tr_seg_files=[]
    for cl in tr_classes:
        # Construct full paths instead of just getting names
        img_dir = f'{base_path}{cl}/image/'
        seg_dir = f'{base_path}{cl}/label/'
        tr_img_files.extend([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        tr_seg_files.extend([os.path.join(seg_dir, f) for f in os.listdir(seg_dir)])
        
    va_ts_img_files=[]
    va_ts_seg_files=[]
    for cl in va_ts_classes:
        # Construct full paths
        img_dir = f'{base_path}{cl}/image/'
        seg_dir = f'{base_path}{cl}/label/'
        va_ts_img_files.extend([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        va_ts_seg_files.extend([os.path.join(seg_dir, f) for f in os.listdir(seg_dir)])

    # Get paired lists
    tr_img_files, tr_seg_files = find_paired_files(tr_img_files, tr_seg_files)
    va_ts_img_files, va_ts_seg_files = find_paired_files(va_ts_img_files, va_ts_seg_files)

    # Random split validation and test
    indices = list(range(len(va_ts_img_files)))
    random.shuffle(indices)
    
    split_point = len(indices) // 2
    va_indices = indices[:split_point]
    ts_indices = indices[split_point:]

    va_img_files = [va_ts_img_files[i] for i in va_indices]
    va_seg_files = [va_ts_seg_files[i] for i in va_indices]
    ts_img_files = [va_ts_img_files[i] for i in ts_indices]
    ts_seg_files = [va_ts_seg_files[i] for i in ts_indices]

# --- Test-Only Split Configuration ---
if ood=='test':
    tr_va_classes=['High-grade IN','Low-grade IN','Normal','Polyp','Serrated adenoma']
    ts_classes=['Adenocarcinoma']
    
    tr_va_img_files=[]
    tr_va_seg_files=[]
    for cl in tr_va_classes:
        # Construct full paths
        img_dir = f'{base_path}{cl}/image/'
        seg_dir = f'{base_path}{cl}/label/'
        tr_va_img_files.extend([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        tr_va_seg_files.extend([os.path.join(seg_dir, f) for f in os.listdir(seg_dir)])
        
    ts_img_files=[]
    ts_seg_files=[]
    for cl in ts_classes:
        # Construct full paths
        img_dir = f'{base_path}{cl}/image/'
        seg_dir = f'{base_path}{cl}/label/'
        ts_img_files.extend([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        ts_seg_files.extend([os.path.join(seg_dir, f) for f in os.listdir(seg_dir)])

    # Get paired lists for both sets
    tr_va_img_files, tr_va_seg_files = find_paired_files(tr_va_img_files, tr_va_seg_files)
    ts_img_files, ts_seg_files = find_paired_files(ts_img_files, ts_seg_files)

    # Randomly split the training/validation set
    indices = list(range(len(tr_va_img_files)))
    random.shuffle(indices)
    
    split_index = int(3 * len(indices) / 4)
    tr_indices = indices[:split_index]
    va_indices = indices[split_index:]
    
    tr_img_files = [tr_va_img_files[i] for i in tr_indices]
    tr_seg_files = [tr_va_seg_files[i] for i in tr_indices]
    va_img_files = [tr_va_img_files[i] for i in va_indices]
    va_seg_files = [tr_va_seg_files[i] for i in va_indices]

# Example of how to use the final lists (e.g., copying files)
# This part is for demonstration; you would add your own logic here.
print(f"Total training images: {len(tr_img_files)}")
print(f"Total validation images: {len(va_img_files)}")
print(f"Total test images: {len(ts_img_files)}")
with open(f'{ood}_fnames.pkl','wb') as f:
    pickle.dump([tr_img_files,tr_seg_files,va_img_files,va_seg_files,ts_img_files,ts_seg_files],f)
    
# Example: copy first training image and label to their destinations
# if tr_img_files:
#     shutil.copy(tr_img_files[0], os.path.join(imagesTr_path, os.path.basename(tr_img_files[0])))
#     shutil.copy(tr_seg_files[0], os.path.join(labelsTr_path, os.path.basename(tr_seg_files[0])))