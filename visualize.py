#%%
from glob import glob
import os
import SimpleITK as sitk
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)

datapath = '/meddat/backup/covid_challenge_eu/synthetic_data'

CT_paths = glob(os.path.join(datapath, "*_vol.nii.gz"))
Mask_paths = glob(os.path.join(datapath, "*_labels.nii.gz"))

CT_paths = sorted(CT_paths)
Mask_paths = sorted(Mask_paths)

print('ct_path: ', CT_paths[0])
print('Mask_path: ', Mask_paths[0])

assert len(Mask_paths) == len(CT_paths), "Unequal number of CTs and Masks"
# #%%
# cts = [sitk.ReadImage(ct_path) for ct_path in CT_paths]
# masks = [sitk.ReadImage(mask_path) for mask_path in Mask_paths]

# print('cts[0].shape: ', cts[0].shape)
# print('masks[0].shape: ', masks[0].shape)
#%%
from utils import get_dataset_stats, dump_data, pl_get_dataset_stats
from utils import plot_hist, collect_hu_values, get_nonzero_voxel_count
from multiprocessing import Pool
from functools import partial
from data import Data

img_size, img_res, img_area = get_dataset_stats(CT_paths=CT_paths, \
                            Mask_paths=Mask_paths)

# #%%

# pool = Pool()

# N = dt.num_scans
# partial_data_stats = partial(pl_get_dataset_stats, data=dt)
# _ = pool.map(partial_data_stats, range(N))

# pool.close()
# pool.join()
#%%
dt = Data(CT_paths, Mask_paths)
dt.img_size = img_size
dt.img_res = img_res
dt.img_area = img_area
#%%
dump_data(img_size, img_res, img_area, CT_paths,'./preprocess')

#%%
all_hu_vals, slice_count_all = collect_hu_values(dt)
#%%

temp_hu_vals = deepcopy(all_hu_vals)
temp_hu_vals[temp_hu_vals < -1500] = -1500
temp_hu_vals[temp_hu_vals > 700] = 700

print '==== Original HU Stats ===='
print 'len: ', len(all_hu_vals)
print 'min HU: ',np.min(all_hu_vals)
print 'max HU: ',np.max(all_hu_vals)
print 'median HU: ',np.median(all_hu_vals) #it can take time - as worst case is O(n^2) | remember qsort to find kth smallest elem
print '==========================='

plot_hist(temp_hu_vals, \
    title='Histogram of HU across subjects for lung abnormalities', \
        x_label='HU', y_label='Count', n_bins=100)

#%%
# len of counts: number of studies i.e 96
# slice_count_all[i] = Array of length j; 
# i is the index over patients | j is the number of slices in slice_count_all[i]
# and the value slice_count_all[i][j] is the voxel count of jth slice of ith patient
counts = get_nonzero_voxel_count(slice_count_all, thresh=0)
num_slices_per_scan = None
for temp_slice in slice_count_all:
    if num_slices_per_scan == None:
        num_slices_per_scan = [len(temp_slice)]
    else:
        num_slices_per_scan.append(len(temp_slice))

num_slices_per_scan = np.array(num_slices_per_scan)
print('Total Slices: ', np.sum(num_slices_per_scan))
print('Average number of slices: ', np.mean(num_slices_per_scan))
print('Median number of slices: ', np.median(num_slices_per_scan))

print('Mean image resolution: ', np.mean(dt.img_res, axis=0))
print('Median image resolution: ', np.median(dt.img_res, axis=0))
print('Mean grid size: ',np.mean(dt.img_size, axis=0))
print('Median grid size: ',np.median(dt.img_size, axis=0))

print('Mean FOV (in mm): ', np.multiply(np.mean(dt.img_size, axis=0), np.mean(dt.img_res, axis=0)))
print('Median FOV (in mm): ', np.multiply(np.median(dt.img_size, axis=0), np.median(dt.img_res, axis=0)))

#%%
# Plotting diff histograms
fig, ax = plt.subplots(1,3)
bin_edges = np.hstack((np.arange(0,2000,50), np.arange(2000,7000,200)))
hist = ax[0].hist(counts, bins=100)
ax[0].set_xlabel('Foreground Voxel Count (Full range)')
ax[0].set_ylabel('#Slices in each bin')
ax[0].set_xticks(np.arange(0,7000,1000))
ax[0].set_yticks(np.arange(0,60,5))
# At two limited levels of granularity - first half
ax[1].hist(counts, bins=np.arange(0,2000,50))
ax[1].set_xlabel('Foreground Voxel Count (0-2k). bin_width=50')
ax[1].set_ylabel('#Slices in each bin')
ax[1].set_title('Histogram of FG Voxel Count At Slice Level Across All Subjects (+ve slices only)')
ax[1].set_xticks(np.arange(0,2000,400))
ax[1].set_yticks(np.arange(0,40,5))
# At two limited levels of granularity - second half
ax[2].hist(counts, bins=np.arange(2000,7000,200))
ax[2].set_xlabel('Foreground Voxel Count (2k-7k). bin_width=200')
ax[2].set_ylabel('#Slices in each bin')
ax[2].set_xticks(np.arange(2000,7000,1000))
ax[2].set_yticks(np.arange(0,30,5))
# Now we look at the number of slices per subject above a certain count threshold
# fig = plt.gcf()
fig.set_size_inches(14,5)
#%%
# Plot cumulative count to help pick a threshold:
hist = np.histogram(counts, bins=1000, range=(0,10000))
cumsum_count = np.cumsum(hist[0])
edges = hist[1]
cumsum_count_thresh = cumsum_count[-1]-cumsum_count
fig, ax = plt.subplots(2,1)
ax[0].plot(edges[:-1], cumsum_count_thresh)
ax[0].set_xlabel('Threshold')
ax[0].set_ylabel('#Slices exceeding threshold')
ax[0].set_title('Use this plot to pick a threshold for chosing which slices to train on')
ax[0].set_xlim(0,7000)
ax[0].set_ylim(0,800)
ax[0].xaxis.set_major_locator(MultipleLocator(400))
ax[0].xaxis.set_minor_locator(MultipleLocator(100))
ax[0].yaxis.set_major_locator(MultipleLocator(100))
ax[0].yaxis.set_minor_locator(MultipleLocator(50))
ax[0].grid(True, 'both')
fig.set_size_inches(14,10)
ax[1].plot(edges[0:50], cumsum_count_thresh[0:50])
ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('#Slices exceeding threshold')
ax[1].set_title('Use this plot to pick a threshold for chosing which slices to train on')
ax[1].set_xlim(0,500)
ax[1].set_ylim(500,800)
ax[1].xaxis.set_major_locator(MultipleLocator(15))
ax[1].xaxis.set_minor_locator(MultipleLocator(5))

# ax[1].set_xticks(np.arange(0,500,10))
ax[1].set_yticks(np.arange(500,800,20))
ax[1].grid(True, 'both')

