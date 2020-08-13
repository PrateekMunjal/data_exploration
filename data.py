import numpy as np

class Data:
    def __init__(self, CT_paths, Mask_paths):

        assert type(CT_paths) == list
        assert type(Mask_paths) == list

        assert len(Mask_paths) == len(CT_paths), "Unequal number of CTs and Masks"

        self.CT_paths = CT_paths
        self.Mask_paths = Mask_paths
        self.num_scans = len(CT_paths)

        self.img_size = np.zeros((self.num_scans, 3)) # Collect image sizes
        self.img_res  = np.zeros((self.num_scans, 3)) # Collect image resolutions
        self.img_area = np.zeros((self.num_scans, 3)) # Collect imaged area in mm (x,y,z)
