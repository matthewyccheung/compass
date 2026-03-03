import os
import numpy as np
import shutil

base_path='/scratch/yc130/ham10k/'
img_path=f'{base_path}HAM10000_imgs/'
seg_path=f'{base_path}HAM10000_segs/'
imagesTr_path=f'{base_path}imagesTr/'
imagesVa_path=f'{base_path}imagesVa/'
imagesTs_path=f'{base_path}imagesTs/'
labelsTr_path=f'{base_path}labelsTr/'
labelsVa_path=f'{base_path}labelsVa/'
labelsTs_path=f'{base_path}labelsTs/'
csv_path=f'{base_path}HAM10000_metadata.csv'

files=np.sort(os.listdir(img_path))
tr_files=files[:int(0.8*len(files))]
va_files=files[int(0.8*len(files)):int(0.9*len(files))]
ts_files=files[int(0.9*len(files)):]
for file in tr_files:
    seg_name=file.split('.jpg')[0]+'_segmentation.png'
    shutil.copy(f'{img_path}{file}',f'{imagesTr_path}{file}')
    shutil.copy(f'{seg_path}{seg_name}',f'{labelsTr_path}{seg_name}')
for file in va_files:
    seg_name=file.split('.jpg')[0]+'_segmentation.png'
    shutil.copy(f'{img_path}{file}',f'{imagesVa_path}{file}')
    shutil.copy(f'{seg_path}{seg_name}',f'{labelsVa_path}{seg_name}')
for file in ts_files:
    seg_name=file.split('.jpg')[0]+'_segmentation.png'
    shutil.copy(f'{img_path}{file}',f'{imagesTs_path}{file}')
    shutil.copy(f'{seg_path}{seg_name}',f'{labelsTs_path}{seg_name}')