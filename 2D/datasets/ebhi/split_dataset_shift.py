import os
import numpy as np
import shutil
import random
from collections import Counter
import pickle

# --- 1. Configuration ---
base_path = '/scratch/yc130/EBHI-SEG/'
# Destination directories are not used for copying in this version,
# but paths are kept for context.
imagesTr_path = os.path.join(base_path, 'imagesTr/')
imagesVa_path = os.path.join(base_path, 'imagesVa/')
imagesTs_path = os.path.join(base_path, 'imagesTs/')
labelsTr_path = os.path.join(base_path, 'labelsTr/')
labelsVa_path = os.path.join(base_path, 'labelsVa/')
labelsTs_path = os.path.join(base_path, 'labelsTs/')

# Define all possible classes to ensure all data is collected
all_classes = ['Normal', 'Serrated adenoma', 'Polyp', 'Low-grade IN', 'High-grade IN', 'Adenocarcinoma']

# --- 2. Data Collection ---

def find_paired_files(img_dir, seg_dir):
    """Finds image/segmentation pairs within a class directory."""
    try:
        img_files = {os.path.basename(f) for f in os.listdir(img_dir)}
        seg_files = {os.path.basename(f) for f in os.listdir(seg_dir)}
    except FileNotFoundError:
        return []
    paired_basenames = sorted(list(img_files.intersection(seg_files)))
    return [(os.path.join(img_dir, f), os.path.join(seg_dir, f)) for f in paired_basenames]

print("Collecting all paired files from class directories...")
all_paired_files = []
for class_name in all_classes:
    img_dir = os.path.join(base_path, class_name, 'image/')
    seg_dir = os.path.join(base_path, class_name, 'label/')
    
    paired_files = find_paired_files(img_dir, seg_dir)
    if not paired_files:
        print(f"  - Warning: No paired files found for class '{class_name}'. Skipping.")
        continue
    
    all_paired_files.extend(paired_files)
    print(f"  - Found {len(paired_files)} paired files for class '{class_name}'.")

# --- 3. Simple Random Splitting ---
print(f"\nTotal paired files found: {len(all_paired_files)}")
print("Performing a simple random split...")
random.shuffle(all_paired_files)

# Define proportions for the split
train_frac = 0.7
cal_frac = 0.15
# The rest (~0.15) will be the test set

n_total = len(all_paired_files)
n_train = int(n_total * train_frac)
n_cal = int(n_total * cal_frac)

train_files = all_paired_files[:n_train]
cal_files = all_paired_files[n_train : n_train + n_cal]
test_files = all_paired_files[n_train + n_cal:]


# --- 4. Final File Lists & Saving ---
def get_label_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(path)))

tr_img_files = [i[0] for i in train_files]
tr_seg_files = [i[1] for i in train_files]
tr_labels = [get_label_from_path(i[0]) for i in train_files]

va_img_files = [i[0] for i in cal_files]
va_seg_files = [i[1] for i in cal_files]
va_labels = [get_label_from_path(i[0]) for i in cal_files]

ts_img_files = [i[0] for i in test_files]
ts_seg_files = [i[1] for i in test_files]
ts_labels = [get_label_from_path(i[0]) for i in test_files]

# Save all file lists and labels for use in your main script
output_data = {
    "train_images": tr_img_files, "train_labels": tr_seg_files, "train_classes": tr_labels,
    "cal_images": va_img_files, "cal_labels": va_seg_files, "cal_classes": va_labels,
    "test_images": ts_img_files, "test_labels": ts_seg_files, "test_classes": ts_labels,
}

save_path = os.path.join(base_path, 'noshift_fnames.pkl')
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
        print(f"  - {cls:<20}: {count:>4} samples ({percent:5.1f}%)")

print_dist_summary("Training", tr_labels)
print_dist_summary("Calibration (Validation)", va_labels)
print_dist_summary("Test", ts_labels)

print("\n--- Overall Summary ---")
print(f"Total Files Used: {len(train_files) + len(cal_files) + len(test_files):>5} samples")
print("\nScript finished successfully.")


# import os
# import numpy as np
# import shutil
# import random
# from collections import Counter
# import pickle

# # --- 1. Configuration ---
# base_path = '/scratch/yc130/EBHI-SEG/'
# # Destination directories are not used for copying in this version,
# # but paths are kept for context.
# imagesTr_path = os.path.join(base_path, 'imagesTr/')
# imagesVa_path = os.path.join(base_path, 'imagesVa/')
# imagesTs_path = os.path.join(base_path, 'imagesTs/')
# labelsTr_path = os.path.join(base_path, 'labelsTr/')
# labelsVa_path = os.path.join(base_path, 'labelsVa/')
# labelsTs_path = os.path.join(base_path, 'labelsTs/')

# # Define classes for the prevalence shift
# minority_classes = ['Normal', 'Serrated adenoma']
# majority_classes = ['Polyp', 'Low-grade IN', 'High-grade IN', 'Adenocarcinoma']
# all_classes = minority_classes + majority_classes

# # --- 2. Data Collection ---

# def find_paired_files(img_dir, seg_dir):
#     """Finds image/segmentation pairs within a class directory."""
#     try:
#         img_files = {os.path.basename(f) for f in os.listdir(img_dir)}
#         seg_files = {os.path.basename(f) for f in os.listdir(seg_dir)}
#     except FileNotFoundError:
#         return []
#     paired_basenames = sorted(list(img_files.intersection(seg_files)))
#     return [(os.path.join(img_dir, f), os.path.join(seg_dir, f)) for f in paired_basenames]

# print("Collecting all paired files from class directories...")
# all_files_by_class = {cls: [] for cls in all_classes}
# for class_name in all_classes:
#     img_dir = os.path.join(base_path, class_name, 'image/')
#     seg_dir = os.path.join(base_path, class_name, 'label/')
    
#     paired_files = find_paired_files(img_dir, seg_dir)
#     if not paired_files:
#         print(f"  - Warning: No paired files found for class '{class_name}'. Skipping.")
#         continue
    
#     random.shuffle(paired_files)
#     all_files_by_class[class_name] = paired_files
#     print(f"  - Found {len(paired_files)} paired files for class '{class_name}'.")

# # --- 3. Splitting Logic for Cal/Test Shift ---

# # Define proportions for initial split
# train_frac = 0.5
# cal_frac = 0.25
# # The rest (0.2) will go into the test pool

# train_files = []
# cal_files = []
# test_pool_by_class = {cls: [] for cls in all_classes}

# print("\nSplitting data into Train, Calibration, and Test pools...")
# for class_name, files in all_files_by_class.items():
#     n_total = len(files)
#     n_train = int(n_total * train_frac)
#     n_cal = int(n_total * cal_frac)
    
#     train_files.extend(files[:n_train])
#     cal_files.extend(files[n_train : n_train + n_cal])
#     test_pool_by_class[class_name] = files[n_train + n_cal:]
    
#     print(f"  - Class '{class_name}': {n_train} Train, {n_cal} Cal, {len(test_pool_by_class[class_name])} in Test Pool.")

# # Shuffle the i.i.d. train and calibration sets
# random.shuffle(train_files)
# random.shuffle(cal_files)

# # --- 4. Constructing the Shifted Test Set ---
# print("\nConstructing the label-shifted test set...")
# test_files = []

# # Take all available samples from minority classes in the test pool
# for class_name in minority_classes:
#     files = test_pool_by_class[class_name]
#     test_files.extend(files)
#     print(f"  - Adding all {len(files)} remaining '{class_name}' samples to Test set.")

# # Take a small, fixed number of samples from majority classes to create the shift
# # This ensures the test set is dominated by the minority classes.
# num_majority_samples_per_class = 15
# for class_name in majority_classes:
#     files = test_pool_by_class[class_name]
#     num_to_take = min(num_majority_samples_per_class, len(files))
#     test_files.extend(files[:num_to_take])
#     print(f"  - Adding {num_to_take} of {len(files)} available '{class_name}' samples to Test set.")

# random.shuffle(test_files)

# # --- 5. Final File Lists & Saving ---
# # Note: You need to extract the class labels for the WCP functions.
# # This helper function does that from the file path.
# def get_label_from_path(path):
#     return os.path.basename(os.path.dirname(os.path.dirname(path)))

# tr_img_files = [i[0] for i in train_files]
# tr_seg_files = [i[1] for i in train_files]
# tr_labels = [get_label_from_path(i[0]) for i in train_files]

# va_img_files = [i[0] for i in cal_files]
# va_seg_files = [i[1] for i in cal_files]
# va_labels = [get_label_from_path(i[0]) for i in cal_files]

# ts_img_files = [i[0] for i in test_files]
# ts_seg_files = [i[1] for i in test_files]
# ts_labels = [get_label_from_path(i[0]) for i in test_files]

# # Save all file lists and labels for use in your main script
# output_data = {
#     "train_images": tr_img_files, "train_labels": tr_seg_files, "train_classes": tr_labels,
#     "cal_images": va_img_files, "cal_labels": va_seg_files, "cal_classes": va_labels,
#     "test_images": ts_img_files, "test_labels": ts_seg_files, "test_classes": ts_labels,
# }
# # output_data=[tr_img_files,tr_seg_files,va_img_files,va_seg_files,ts_img_files,ts_seg_files]

# # save_path = os.path.join(base_path, 'prevalence_fnames.pkl')
# # with open(save_path, 'wb') as f:
# #     pickle.dump(output_data, f)
# # print(f"\nSaved all file lists and labels to {save_path}")


# # --- 6. Verification ---
# print("\n--- Data Split Verification ---")

# def print_dist_summary(name, files, labels):
#     counts = Counter(labels)
#     total = len(files)
#     print(f"\n{name} Set ({total} samples):")
#     for cls in all_classes:
#         count = counts.get(cls, 0)
#         percent = (count / total) * 100 if total > 0 else 0
#         print(f"  - {cls:<20}: {count:>4} samples ({percent:5.1f}%)")

# print_dist_summary("Training", train_files, tr_labels)
# print_dist_summary("Calibration (Validation)", cal_files, va_labels)
# print_dist_summary("Test", test_files, ts_labels)

# print("\n--- Overall Summary ---")
# print(f"Total Files Used: {len(train_files) + len(cal_files) + len(test_files):>5} samples")
# print("\nScript finished successfully.")

