import SimpleITK as sitk
import os
import numpy as np
from openpyxl import Workbook
from tqdm import tqdm
from data import Data
import matplotlib.pyplot as plt

def load_image(img_path):
    """
    This function loads the image specified by img_path.
    
    INPUT:
    img_path: str, Absolute path of nifti images
    
    OUTPUT:
    sitk object corresponding to img_path specified
    """ 
    assert type(img_path) == str
    assert os.path.exists(img_path) 
    
    return sitk.ReadImage(img_path)

def get_size_resolution(sitk_img_object):
    """
    Returns the size and voxel spacing of sitk_image_object
    """
    assert sitk_img_object is not None

    return sitk_img_object.GetSize(), sitk_img_object.GetSpacing()

def collect_hu_values(data):
    """
    Collect stats on Housenfeld units of the masks for scaling and visualizations.

    INPUT:
    mask_paths: list, absolute paths of masks
    data: object of class Data

    OUTPUT:
    
    """
    assert isinstance(data, Data)

    mask_paths = data.Mask_paths
    assert type(mask_paths) == list

    locsAll = [] # collect locations where mask is 1
    slice_count_all = [] # collect # non-0 voxels in each slice

    for mask_path in tqdm(mask_paths, desc='Processing all masks'):
        assert os.path.exists(mask_path) == True, 'No mask exists at '+mask_path
        mask_arr = sitk.GetArrayFromImage(load_image(mask_path))
        locsAll.append(np.where(mask_arr>0.5))
        slice_count = [np.sum(i) for i in mask_arr]
        slice_count_all.append(slice_count)

    # Now count number of voxels that are 1 in each mask
    cum_sum = [0] # This we'll use later
    numPoints = [] # Stores number of voxels for each mask
    for i in locsAll: 
        numPoints.append(len(i[0]))
        cum_sum.append(cum_sum[-1]+numPoints[-1] )
    
    # Now we preallocate an array to dump HU values from the mask
    all_hu_vals = np.zeros((cum_sum[-1], ))
    # Get HU value from each mask and store
    ind = 0
    for ct_path, mask_path in tqdm(zip(data.CT_paths, data.Mask_paths), total=len(data.CT_paths), desc='Collecting HU values'):
        ct_arr = sitk.GetArrayFromImage( load_image(ct_path) )
        mask_arr = sitk.GetArrayFromImage( load_image(mask_path) )
        all_hu_vals[ cum_sum[ind]:cum_sum[ind+1] ] = ct_arr[ mask_arr>0.5 ]
        ind+=1
    
    return all_hu_vals, slice_count_all

def get_nonzero_voxel_count(slice_count_all, thresh=0):
    """
    get voxelcounts > thresh
    """
    counts = []
    for subj_list in slice_count_all:
        subj_list = np.array(subj_list)
        subj_list = subj_list[ subj_list>thresh ]
        counts.extend( subj_list.tolist() )
    return counts

def plot_hist(array, title, x_label, y_label, n_bins=200):
    """
    Plots the histogram where values are specified in array and groups
    the data using n_bins.

    INPUT:
    array: np.ndarray, The array values.
    title: str, the title of plot
    x_label: str, label shown on x-axis
    y_label: str, label shown on y-axis
    n_bins: int, the number of bins
    """
    assert type(array) == np.ndarray
    assert len(array)>0
    assert type(title) == str
    assert type(x_label) == str
    assert type(y_label) == str
    assert type(n_bins) == int
    assert n_bins > 0
    
    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)
    ax.hist(array, bins=n_bins)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

def dump_data(img_size, img_res, img_area, CT_paths,save_fpath=None):

    """
    Dumps the data in a csv file saved at path: save_fapth.

    INPUT:
    save_fpath: str, Path where we dump all the dataset statistics. Default: None

    OUTPUT:
    None

    """
    assert (save_fpath == None) or (type(save_fpath)==str)

    assert type(CT_paths) == list
    assert len(CT_paths) > 0

    assert type(img_size) == np.ndarray
    assert img_size.ndim == 2
    assert img_size.shape[0] == len(CT_paths)

    assert type(img_res) == np.ndarray
    assert img_res.ndim == 2
    assert img_res.shape[0] == len(CT_paths)

    assert type(img_area) == np.ndarray
    assert img_area.ndim == 2
    assert img_area.shape[0] == len(CT_paths)

    #construct save dir
    if save_fpath:
        if not os.path.exists(save_fpath):
            os.makedirs(save_fpath)

        #dump data
        wb = Workbook()
        ws = wb.active
        columns = 'SubjID, szX, szY, szZ, resX, resY, resZ, areaX, areaY, areaZ\n'
        fout = 'data_summary.csv'
        with open(os.path.join(save_fpath,fout), 'w') as f:
            f.write(columns)
            for size, res, area, wb_ct_path in tqdm(zip(img_size, img_res, img_area, CT_paths), total=len(CT_paths), desc='Dumping results'):
                subjID = os.path.split(wb_ct_path)[1].split('.')[0]
                writeOutArr = "{}, {:.0f}, {:.0f}, {:.0f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n".format(subjID, \
                    size[0], size[1], size[2], res[0], res[1], res[2], area[0], area[1], area[2])
                f.write(writeOutArr)
            
        print('Dataset stats written to ', os.path.join(save_fpath, fout))

def pl_get_dataset_stats(id, data):
    """
    Evaluate size, resolution and area of ct_scan
    """
    ct_path = data.CT_paths[id]
    mask_path = data.Mask_paths[id]

    assert os.path.exists(ct_path)
    assert os.path.exists(mask_path)

    ct = load_image(ct_path)
    mask = load_image(ct_path)

    ct_size, ct_res = get_size_resolution(ct)
    mask_size, mask_res = get_size_resolution(mask)
    
    assert ct_size==mask_size, "CT and mask image sizes must match\nDebugInfo:\nCT:{}\nMask:{}".format(ct_path, mask_path)
    
    data.img_size[id] = ct_size
    data.img_res[id]  = ct_res
    data.img_area[id] = [i*j for i,j in zip(ct_size, ct_res)]

def get_dataset_stats(CT_paths, Mask_paths):
    """
    INPUT:
    CT_paths: list, A list containing absolute paths to CTs
    Mask_paths: list, A list containing absolute paths to Masks
    
    OUTPUT:
    imageSizes, imageResolutions, imageAreas

    NOTE:
    If save_fpath is None, then we do not save the stats.
    """

    assert type(CT_paths) == list
    assert type(Mask_paths) == list

    assert len(CT_paths) == len(Mask_paths)
    
    num_scan = len(CT_paths)
    img_size = np.zeros((num_scan, 3)) # Collect image sizes
    img_res  = np.zeros((num_scan, 3)) # Collect image resolutions
    img_area = np.zeros((num_scan, 3)) # Collect imaged area in mm (x,y,z)
    ind = 0
    #loop over cts, masks
    for ct_path, mask_path in tqdm(zip(CT_paths, Mask_paths),total=len(CT_paths)):
        assert os.path.exists(ct_path)
        assert os.path.exists(mask_path)

        ct = load_image(ct_path)
        mask = load_image(ct_path)
        ct_size, ct_res = get_size_resolution(ct)
        mask_size, mask_res = get_size_resolution(mask)
        assert ct_size==mask_size, "CT and mask image sizes must match\nDebugInfo:\nCT:{}\nMask:{}".format(ct_path, mask_path)
        img_size[ind] = ct_size
        img_res[ind]  = ct_res
        img_area[ind] = [i*j for i,j in zip(ct_size, ct_res)]
        ind += 1

    return img_size, img_res, img_area