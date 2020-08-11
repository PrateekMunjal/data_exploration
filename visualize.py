#%%
from glob import glob
import os
import SimpleITK as sitk
import numpy as np

datapath = '/meddat/backup/covid_challenge_eu/synthetic_data'

CT_paths = glob(os.path.join(datapath, "*_vol.nii.gz"))
Mask_paths = glob(os.path.join(datapath, "*_labels.nii.gz"))

CT_paths = sorted(CT_paths)
Mask_paths = sorted(Mask_paths)

print('ct_path: ', CT_paths[0])
print('Mask_path: ', Mask_paths[0])

assert len(Mask_paths) == len(CT_paths), "Unequal number of CTs and Masks"
#%%
cts = [sitk.ReadImage(ct_path) for ct_path in CT_paths]
masks = [sitk.ReadImage(mask_path) for mask_path in Mask_paths]

print('cts[0].shape: ', cts[0].shape)
print('masks[0].shape: ', masks[0].shape)
#%%


