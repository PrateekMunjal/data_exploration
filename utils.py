import SimpleITK as sitk
import os
import numpy as np

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

def get_dataset_stats(CT_paths, Mask_paths):
    """

    INPUT:
    CT_paths: list, A list containing absolute paths to CTs
    Mask_paths: list, A list containing absolute paths to Masks
    
    OUTPUT:

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
    for ct_path, mask_path in zip(CT_paths, Mask_paths):
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
    
    