import numpy as np
import tifffile as tiff
import plotting_help_py37 as ph

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import os, random, math, copy, glob, nrrd, re, csv
from PIL import Image
from scipy.stats import pearsonr



def Pearson_correlation(X,Y):
    if len(X)==len(Y):
        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
        Sum_x_squared = sum((X-X.mean())**2)
        Sum_y_squared = sum((Y-Y.mean())**2)      
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr

def get_glomtypes_from_seqss(seqss, datatype_orn):
    if datatype_orn:
        types_ = np.array(seqss.obs.all_types)
        types_[seqss.obs.MARS_matched == 'MARS_2'] = 'DL1'
        types_[seqss.obs.MARS_matched == 'MARS_3'] = 'DC1'
        types_[seqss.obs.MARS_matched == 'MARS_5'] = 'VM6'
        types_[seqss.obs.all_types == 'DA41'] = 'DA4l'
        return types_
    else:
        types_ = np.array(seqss.obs.PN_type)
        types_[seqss.obs.PN_type == 'DA1_fru+'] = 'DA1'
        types_[seqss.obs.PN_type == 'DA1_fru-'] = 'DA1'
        types_[seqss.obs.PN_type == 'VM7v(1)'] = 'VM7v'
        # types_[seqss.obs.PN_type == 'MARS_5'] = 'DA3'
        # types_[seqss.obs.PN_type == 'MARS_27'] = 'DM5'
        # types_[seqss.obs.PN_type == 'MARS_7'] = 'VA3'
        types_[seqss.obs.PN_type == 'VM7 or VM5v #1'] = 'VM7'
        types_[seqss.obs.PN_type == 'VM7 or VM5v #2'] = 'VM5v'
        types_[seqss.obs.PN_type == 'DA41'] = 'DA4l'
        types_[seqss.obs.PN_type == 'MARS_2'] = 'DA2'
        types_[seqss.obs.PN_type == 'MARS_9'] = 'VA7m'
        types_[seqss.obs.PN_type == 'MARS_25'] = 'VM1'
        return types_

def get_recs_ORN_label(genotype, parent_folder='./', ch_pn=0, multifolders=False, **kwargs):
    recs = []
    if multifolders:
        fns = glob.glob(parent_folder + os.path.sep + genotype + os.path.sep + genotype + '*.tif')
    else:
        fns = glob.glob(parent_folder + os.path.sep + genotype + '*.tif')

    for fn in fns:
        rec = mistarget(folder='./', filename=fn[:-4],ch_pn=ch_pn, **kwargs)
        recs.append(rec)
    return recs

class mistarget():

    def __init__(self, folder='./', filename='', ch_pn = 0 ):

        self.intensity_DA1 = [0, 0]
        self.intensity_VA1d = [0, 0]
        self.intensity_VA1v = [0, 0]

        self.area_DA1 = [0, 0]
        self.area_VA1d = [0, 0]
        self.area_VA1v = [0, 0]
        self.area_DA4l = [0, 0]
        self.area_DC3 = [0, 0]
        self.area_total = [0, 0]

        # Initialize DA1 mask file
        DA1_mask_fns = (glob.glob(folder + os.path.sep + filename + '-DA1.nrrd'))
        if len(DA1_mask_fns):
            mask_, _ = nrrd.read(DA1_mask_fns[0])
            self.DA1mask = np.swapaxes(np.swapaxes(mask_.T, 0, 1), 1, 2)
        else:
            self.DA1mask = None

        # Initialize VA1d mask file
        VA1d_mask_fns = (glob.glob(folder + os.path.sep + filename + '-VA1d.nrrd'))
        if len(VA1d_mask_fns):
            self.fn = VA1d_mask_fns[0] #get file name in VA1d
            mask_, _ = nrrd.read(VA1d_mask_fns[0])
            self.VA1dmask = np.swapaxes(np.swapaxes(mask_.T, 0, 1), 1, 2)
        else:
            self.VA1dmask = None

        # Initialize VA1v mask file
        VA1v_mask_fns = (glob.glob(folder + os.path.sep + filename + '-VA1v.nrrd'))
        if len(VA1v_mask_fns):
            mask_, _ = nrrd.read(VA1v_mask_fns[0])
            self.VA1vmask = np.swapaxes(np.swapaxes(mask_.T, 0, 1), 1, 2)
        else:
            self.VA1vmask = None
            
        # Initialize DA4l mask file
        DA4l_mask_fns = (glob.glob(folder + os.path.sep + filename + '-DA4l.nrrd'))
        if len(DA4l_mask_fns):
            mask_, _ = nrrd.read(DA4l_mask_fns[0])
            self.DA4lmask = np.swapaxes(np.swapaxes(mask_.T, 0, 1), 1, 2)
        else:
            self.DA4lmask = None
            
        # Initialize DC3 mask file
        DC3_mask_fns = (glob.glob(folder + os.path.sep + filename + '-DC3.nrrd'))
        if len(DC3_mask_fns):
            mask_, _ = nrrd.read(DC3_mask_fns[0])
            self.DC3mask = np.swapaxes(np.swapaxes(mask_.T, 0, 1), 1, 2)
        else:
            self.DC3mask = None

        # Initialize thresholding file
        self.orndenmask = [None, None]
        orndenmask_fns = (glob.glob(folder + os.path.sep + filename + '-orndenmask.nrrd')
                      + glob.glob(folder + os.path.sep + filename + '-orndenmask-l.nrrd'))
        if len(orndenmask_fns):
            mask_, _ = nrrd.read(orndenmask_fns[0])
            self.orndenmask[0] = np.swapaxes(np.swapaxes(mask_.T, 0, 1), 1, 2)
        orndenmask_fns = (glob.glob(folder + os.path.sep + filename + '-orndenmask.nrrd')
                      + glob.glob(folder + os.path.sep + filename + '-orndenmask-r.nrrd'))
        if len(orndenmask_fns):
            mask_, _ = nrrd.read(orndenmask_fns[0])
            self.orndenmask[1] = np.swapaxes(np.swapaxes(mask_.T, 0, 1), 1, 2)

        # Initialize tiff files
        if ch_pn is not None:
            fn = (glob.glob(folder + os.path.sep + filename + '.tif'))[0]
            tif_ = tiff.imread(fn)
            tif = np.swapaxes(np.swapaxes(tif_.T, 0, 1), 2, 3)[:, :, :, ch_pn]
            self.tif = tif
            self.Y, self.X, self.Z = tif.shape
            self.pn = [0, 0]

            img = Image.open(fn)
            exifdata = img.getexif()
            self.xf = 1. / exifdata.get(282)    # 282 is a specific number for getting the XResolution value
            self.yf = self.xf
            self.zf = 1
            # Annotation: In case 282 gives the wrong info, check with the code below
            # for tag_id in exifdata:
            #     # get the tag name, instead of human unreadable tag id
            #     tag = TAGS.get(tag_id, tag_id)
            #     if tag == 'XResolution':
            #         data = exifdata.get(tag_id)
            #         break
        else:
            self.tif = None
        self.open()


    def open(self):
        if (self.DA1mask is not None) & (self.VA1dmask is not None) & (self.VA1vmask is not None) & (self.orndenmask is not None):
            #self.calculate_mistarget_intensity()
            self.calculate_mistarget_area()
        
        if (self.VA1dmask is not None) & (self.DA4lmask is not None):
            self.calculate_mistarget_area_DA4l()
            
        if (self.VA1dmask is not None) & (self.DC3mask is not None):
            self.calculate_mistarget_area_DC3()

    def calculate_mistarget_intensity(self):

        DA1mask = self.DA1mask
        VA1dmask = self.VA1dmask
        VA1vmask = self.VA1vmask
        for side_index in range(2):
            DA1mask_bool = (DA1mask == side_index + 1)
            VA1dmask_bool = (VA1dmask == side_index + 1)
            VA1vmask_bool = (VA1vmask == side_index + 1)

            # determine which side it is
            sig = self.tif * DA1mask_bool
            sig_sum = np.sum(sig)
            x_pnc_p = np.sum(np.sum(np.sum(sig, axis=0), axis=1) * np.arange(self.X)) / sig_sum  # sum axis 0, leave axis 1 as X
            side = 0 if (x_pnc_p < self.X / 2.) else 1 # left is 0

            if DA1mask_bool.sum() & VA1dmask_bool.sum() & VA1vmask_bool.sum():
                self.intensity_DA1[side] = np.sum(self.tif * DA1mask_bool)  # total intensity in DA1
                self.intensity_VA1d[side] = np.sum(self.tif * VA1dmask_bool)  # total intensity in VA1d
                self.intensity_VA1v[side] = np.sum(self.tif * VA1vmask_bool)  # total intensity in VA1v

    def calculate_mistarget_area(self):

        DA1mask = self.DA1mask
        VA1dmask = self.VA1dmask
        VA1vmask = self.VA1vmask
        for side_index in range(2): # when drawing mask, left then right
            DA1mask_bool = (DA1mask == side_index + 1)
            VA1dmask_bool = (VA1dmask == side_index + 1)
            VA1vmask_bool = (VA1vmask == side_index + 1)

            # determine which side it is
            sig = self.tif * VA1dmask_bool
            sig_sum = np.sum(sig)
            x_pnc_p = np.sum(np.sum(np.sum(sig, axis=0), axis=1) * np.arange(self.X)) / sig_sum  # sum axis 0, leave axis 1 as X
            side = 0 if (x_pnc_p < self.X / 2.) else 1 # left is 0

            if DA1mask_bool.sum() & VA1dmask_bool.sum() & VA1vmask_bool.sum():
                orndenmask_bool = (self.orndenmask[side] > 1)
                self.area_DA1[side] = np.sum(orndenmask_bool * DA1mask_bool) # total area in DA1
                self.area_VA1d[side] = np.sum(orndenmask_bool * VA1dmask_bool)  # total area in VA1d
                self.area_VA1v[side] = np.sum(orndenmask_bool * VA1vmask_bool)  # total area in VA1v
   
    def calculate_mistarget_area_DA4l(self):

        DA4lmask = self.DA4lmask
        VA1dmask = self.VA1dmask

        for side_index in range(2): # when drawing mask, left then right
            DA4lmask_bool = (DA4lmask == side_index + 1)
            VA1dmask_bool = (VA1dmask == side_index + 1)

            # determine which side it is
            sig = self.tif * DA4lmask_bool
            sig_sum = np.sum(sig)
            x_pnc_p = np.sum(np.sum(np.sum(sig, axis=0), axis=1) * np.arange(self.X)) / sig_sum  # sum axis 0, leave axis 1 as X
            side = 0 if (x_pnc_p < self.X / 2.) else 1 # left is 0

            if DA4lmask_bool.sum() & VA1dmask_bool.sum():
                orndenmask_bool = (self.orndenmask[side] > 1)
                self.area_DA4l[side] = np.sum(orndenmask_bool * DA4lmask_bool) # total area in DA4l
                self.area_VA1d[side] = np.sum(orndenmask_bool * VA1dmask_bool)  # total area in VA1d
                self.area_total[side] = np.sum(orndenmask_bool * DA4lmask_bool) + np.sum(orndenmask_bool * VA1dmask_bool)  # total area 

    def calculate_mistarget_area_DC3(self):

        DC3mask = self.DC3mask
        VA1dmask = self.VA1dmask

        for side_index in range(2): # when drawing mask, left then right
            DC3mask_bool = (DC3mask == side_index + 1)
            VA1dmask_bool = (VA1dmask == side_index + 1)

            # determine which side it is
            sig = self.tif * DC3mask_bool
            sig_sum = np.sum(sig)
            x_pnc_p = np.sum(np.sum(np.sum(sig, axis=0), axis=1) * np.arange(self.X)) / sig_sum  # sum axis 0, leave axis 1 as X
            side = 0 if (x_pnc_p < self.X / 2.) else 1 # left is 0

            if DC3mask_bool.sum() & VA1dmask_bool.sum():
                orndenmask_bool = (self.orndenmask[side] > 1)
                self.area_DC3[side] = np.sum(orndenmask_bool * DC3mask_bool) # total area in DC3
                self.area_VA1d[side] = np.sum(orndenmask_bool * VA1dmask_bool)  # total area in VA1d
                self.area_total[side] = np.sum(orndenmask_bool * DC3mask_bool) + np.sum(orndenmask_bool * VA1dmask_bool)  # total area 
