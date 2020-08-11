#%%
import os
import SimpleITK as sitk
import subprocess as sp
import numpy as np
from openpyxl import Workbook
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)

def load_image(img_path): return sitk.ReadImage(img_path)
def get_size_resolution(sitk_img_object): return sitk_img_object.GetSize(), sitk_img_object.GetSpacing()

# Load all images
# Evaluate image sizes and resolutions
# Visualize quality of segmentation
ct_dir   = '/Users/shadab.khan/Desktop/Cogna/COVID19_1110/studies/CT-1'
mask_dir = '/Users/shadab.khan/Desktop/Cogna/COVID19_1110/masks'
# Grab path to all the CTs and masks
CTs = os.listdir(ct_dir)
CTs.sort()
CTs = [os.path.join(ct_dir, i) for i in CTs if i.endswith('.nii.gz')]
masks = os.listdir(mask_dir)
masks.sort()
masks = [os.path.join(mask_dir, i) for i in masks if i.endswith('.nii.gz')]
assert len(masks)==len(CTs), "Number of CTs and masks must match"

#%% Cycle through all images and collect stats
num_scan = len(CTs)
img_size = np.zeros((num_scan, 3)) # Collect image sizes
img_res  = np.zeros((num_scan, 3)) # Collect image resolutions
img_area = np.zeros((num_scan, 3)) # Collect imaged area in mm (x,y,z)
ind = 0
for ct_path, mask_path in zip(CTs, masks):
    ct = load_image(ct_path)
    mask = load_image(ct_path)
    ct_size, ct_res = get_size_resolution(ct)
    mask_size, mask_res = get_size_resolution(mask)
    assert ct_size==mask_size, "CT and mask image sizes must match\nDebugInfo:\nCT:{}\nMask:{}".format(ct_path, mask_path)
    img_size[ind] = ct_size
    img_res[ind]  = ct_res
    img_area[ind] = [i*j for i,j in zip(ct_size, ct_res)]
    ind += 1
#%% Dump stats to file
wb = Workbook()
ws = wb.active
columns = 'SubjID, szX, szY, szZ, resX, resY, resZ, areaX, areaY, areaZ\n'
fout = 'data_summary.csv'
with open(fout, 'w') as f:
    f.write(columns)
    for size, res, area, ct_path in zip(img_size, img_res, img_area, CTs):
        subjID = os.path.split(ct_path)[1].split('.')[0]
        writeOutArr = "{}, {:.0f}, {:.0f}, {:.0f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n".format(subjID, \
            size[0], size[1], size[2], res[0], res[1], res[2], area[0], area[1], area[2])
        f.write(writeOutArr)
#%% Collect stats on Housenfeld units of the masks for scaling and visualizations
locsAll = [] # collect locations where mask is 1
slice_count_all = [] # collect # non-0 voxels in each slice

for mask_path in masks:
    mask_arr = sitk.GetArrayFromImage( load_image(mask_path) )
    locsAll.append( np.where(mask_arr>0.5) )
    slice_count = [np.sum(i) for i in mask_arr]
    slice_count_all.append(slice_count)
# Now count number of voxels that are 1 in each mask
cum_sum = [0] # This we'll use later
numPoints = [] # Stores number of voxels for each mask
for i in locsAll: 
    numPoints.append( len(i[0]) )
    cum_sum.append( cum_sum[-1]+numPoints[-1] )
#%% Now we preallocate an array to dump HU values from the mask
all_hu_vals = np.zeros((cum_sum[-1], ))
# Get HU value from each mask and store
ind = 0
for ct_path, mask_path in zip(CTs, masks):
    ct_arr = sitk.GetArrayFromImage( load_image(ct_path) )
    mask_arr = sitk.GetArrayFromImage( load_image(mask_path) )
    all_hu_vals[ cum_sum[ind]:cum_sum[ind+1] ] = ct_arr[ mask_arr>0.5 ]
    ind+=1
#%% Create plots
# First plot the histogram of HUs
fig, ax = plt.subplots()
ax.hist(all_hu_vals, bins=200)
ax.set_xlabel('HU')
ax.set_ylabel('Count')
ax.set_title('Histogram of HU across subjects for lung abnormalities')
#%% Histogram of number of non-zero voxels in non-zero slices
def get_nonzero_voxel_count(slice_count_all, thresh=0):
    counts = []
    for subj_list in slice_count_all:
        subj_list = np.array(subj_list)
        subj_list = subj_list[ subj_list>thresh ]
        counts.extend( subj_list.tolist() )
    return counts
counts = get_nonzero_voxel_count(slice_count_all, thresh=0)
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
#%% Plot cumulative count to help pick a threshold:
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

#%%
i=45
ct = CTs[-1]
mask = masks[-1]
sp.call(( 'itksnap', '-g', ct, '-s', mask ))

# # # Cycle through all 50 images and make notes of any observations
# for ct, mask in zip(CTs, masks):
#     sp.call(( 'itksnap', '-g', ct, '-s', mask ))
#     i+=1
#     if i==1: break

# %%
