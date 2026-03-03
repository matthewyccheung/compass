import os
import pandas as pd
import numpy as np
import shutil
import random
from collections import Counter
import pickle

# --- 1. Configuration ---
base_path = '/scratch/yc130/ham10k/'
img_source_path = os.path.join(base_path, 'HAM10000_imgs/')
seg_source_path = os.path.join(base_path, 'HAM10000_segs/')
csv_path = os.path.join(base_path, 'HAM10000_metadata.csv')

# Define all possible classes to ensure they are all collected
all_classes = ['akiec', 'bcc', 'bkl', 'df', 'vasc', 'mel', 'nv']

# --- 2. Data Collection from CSV ---

def collect_paired_files_from_csv(csv_path, img_dir, seg_dir):
    """
    Reads a CSV, finds corresponding image/segmentation pairs on disk,
    and returns a single list of all found samples.

    Returns:
        list: A list of sample dicts.
              Each dict is {'img': img_path, 'seg': seg_path, 'dx': class_label}
    """
    print("Reading CSV and collecting all paired files...")
    df = pd.read_csv(csv_path)
    all_files = []
    
    found_count = 0
    for _, row in df.iterrows():
        image_id = row['image_id']
        class_label = row['dx']
        
        img_path = os.path.join(img_dir, f"{image_id}.jpg")
        seg_path = os.path.join(seg_dir, f"{image_id}_segmentation.png")
        
        # Critical check: Ensure both image and segmentation files exist
        if os.path.exists(img_path) and os.path.exists(seg_path):
            sample_info = {'img': img_path, 'seg': seg_path, 'dx': class_label}
            all_files.append(sample_info)
            found_count += 1
        
    print(f"Found {found_count} total paired image/segmentation files.")
    return all_files

all_paired_files = collect_paired_files_from_csv(csv_path, img_source_path, seg_source_path)

# --- 3. Simple Random Splitting ---
print("\nPerforming a simple random split...")
random.shuffle(all_paired_files)

# Define proportions for the split
train_frac = 0.8
cal_frac = 0.1

n_total = len(all_paired_files)
n_train = int(n_total * train_frac)
n_cal = int(n_total * cal_frac)

train_files = all_paired_files[:n_train]
cal_files = all_paired_files[n_train : n_train + n_cal]
test_files = all_paired_files[n_train + n_cal:]

# --- 4. Final File Lists & Saving ---
# Unpack the lists of dictionaries into separate lists for images, segs, and classes
tr_img_files = [sample['img'] for sample in train_files]
tr_seg_files = [sample['seg'] for sample in train_files]
tr_labels = [sample['dx'] for sample in train_files]

va_img_files = [sample['img'] for sample in cal_files]
va_seg_files = [sample['seg'] for sample in cal_files]
va_labels = [sample['dx'] for sample in cal_files]

ts_img_files = [sample['img'] for sample in test_files]
ts_seg_files = [sample['seg'] for sample in test_files]
ts_labels = [sample['dx'] for sample in test_files]

# Save all file lists and labels for use in your main script
output_data = {
    "train_images": tr_img_files, "train_labels": tr_seg_files, "train_classes": tr_labels,
    "cal_images": va_img_files, "cal_labels": va_seg_files, "cal_classes": va_labels,
    "test_images": ts_img_files, "test_labels": ts_seg_files, "test_classes": ts_labels,
}

save_path = os.path.join(base_path, 'prevalence_fnames.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(output_data, f)
print(f"\nSaved all file lists and labels to {save_path}")


# --- 5. Verification ---
print("\n--- Data Split Verification ---")

def print_dist_summary(name, labels):
    counts = Counter(labels)
    total = len(labels)
    print(f"\n{name} Set ({total} samples):")
    for cls in all_classes:
        count = counts.get(cls, 0)
        percent = (count / total) * 100 if total > 0 else 0
        print(f"  - {cls:<10}: {count:>4} samples ({percent:5.1f}%)")

print_dist_summary("Training", tr_labels)
print_dist_summary("Calibration (Validation)", va_labels)
print_dist_summary("Test", ts_labels)

print("\n--- Overall Summary ---")
print(f"Total Files Used: {len(train_files) + len(cal_files) + len(test_files):>5} samples")
print("\nScript finished successfully.")
