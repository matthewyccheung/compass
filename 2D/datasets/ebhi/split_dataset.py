import os
import numpy as np
import shutil

base_path='/scratch/yc130/EBHI-SEG/'
img_path=f'{base_path}imgs/'
seg_path=f'{base_path}segs/'

imagesTr_path=f'{base_path}imagesTr/'
imagesVa_path=f'{base_path}imagesVa/'
imagesTs_path=f'{base_path}imagesTs/'
labelsTr_path=f'{base_path}labelsTr/'
labelsVa_path=f'{base_path}labelsVa/'
labelsTs_path=f'{base_path}labelsTs/'

dir_list=[imagesTr_path,imagesVa_path,imagesTs_path,
          labelsTr_path,labelsVa_path,labelsTs_path]
for dir_path in dir_list:
    # if os.path.exists(dir_path)==True:
    #     shutil.rmtree(dir_path)
    if os.path.exists(dir_path)==False:
        os.mkdir(dir_path)

img_files=set(np.sort(os.listdir(img_path)))
seg_files=set(np.sort(os.listdir(seg_path)))
files=list(img_files&seg_files)
tr_files=files[:int(0.8*len(files))]
va_files=files[int(0.8*len(files)):int(0.9*len(files))]
ts_files=files[int(0.9*len(files)):]
for file in tr_files:
    shutil.copy(f'{img_path}{file}',f'{imagesTr_path}{file}')
    shutil.copy(f'{seg_path}{file}',f'{labelsTr_path}{file}')
for file in va_files:
    shutil.copy(f'{img_path}{file}',f'{imagesVa_path}{file}')
    shutil.copy(f'{seg_path}{file}',f'{labelsVa_path}{file}')
for file in ts_files:
    shutil.copy(f'{img_path}{file}',f'{imagesTs_path}{file}')
    shutil.copy(f'{seg_path}{file}',f'{labelsTs_path}{file}')