# this file includes a variety of transform configurations that could be applied as needed #

import torch
import monai
import monai.transforms as mt
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, binary_erosion
from skimage import measure 
from copy import deepcopy

class RoiCrop:
    '''
    this transform uses the center of the label to crop both image and label with fixed size
    expects NO CHANNEL input
    adds CHANNEL back
    '''

    def __init__(self, img_size):

        self.img_size = img_size

    def __call__(self,img_data,lbl_data):

        if not isinstance(img_data, np.ndarray):
            img_data = img_data.cpu().numpy()

        if not isinstance(lbl_data, np.ndarray):
            lbl_data = lbl_data.cpu().numpy()

        # get foreground points and corresponding values
        foreground_coords = np.argwhere(np.greater(lbl_data,0))

        # get distance from centroid (to see how large to crop)
        foreground_center = np.round(foreground_coords.mean(axis=0))
        x_start = int(max(foreground_center[0] - self.img_size[0] // 2,0))
        y_start = int(max(foreground_center[1] - self.img_size[1] // 2,0))
        z_start = int(max(foreground_center[2] - self.img_size[2] // 2,0))

        x_end = int(min(x_start + self.img_size[0],lbl_data.shape[0]))
        y_end = int(min(y_start + self.img_size[1],lbl_data.shape[1]))
        z_end = int(min(z_start + self.img_size[2],lbl_data.shape[2]))

        start_coords = [x_end-self.img_size[0],y_end-self.img_size[1],z_end-self.img_size[2]]
        end_coords = [x_end,y_end,z_end]

        # apply bounds
        out_img = np.expand_dims(img_data[start_coords[0]:end_coords[0],start_coords[1]:end_coords[1],start_coords[2]:end_coords[2]],axis=0)

        out_lbl = np.expand_dims(lbl_data[start_coords[0]:end_coords[0],start_coords[1]:end_coords[1],start_coords[2]:end_coords[2]],axis=0)

        # ensures that the label data contains the whole roi
        if out_lbl.sum() != lbl_data.sum():
            #print(lbl_data.sum()-out_lbl.sum(),lbl_data.sum())
            pass

        self.start_coords = start_coords
        self.end_coords = end_coords
        self.original_size = img_data.shape

        return torch.Tensor(out_img)

    def inverse(self,img):

        start_coords = np.asarray(self.start_coords)
        end_coords = np.asarray(self.end_coords)

        # first create original array
        original = torch.zeros(self.original_size)
        original[start_coords[0]:end_coords[0],start_coords[1]:end_coords[1],start_coords[2]:end_coords[2]] = img

        return original

def nnUnet_transform(img,fingerprint):
    '''
    this transform clips and normalizes images using fingerprint dictionary
    '''
    x = img.clone()
    out = torch.clip(x,min=fingerprint['0.5 percentile'],max=fingerprint['99.5 percentile'])
    out = (out - fingerprint['mean']) / fingerprint['std']

    return out

def resample_transform(pixdims=(2.3,2.3,3)):
    '''
    basic resampling
    '''

    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]), # load image
            mt.EnsureChannelFirstd(keys=["image", "label"]), # make channel first
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"), # adjust orientation
            mt.Spacingd(keys=["image", "label"], pixdim=pixdims, mode=("bilinear", "nearest")), # change spacing
            mt.EnsureTyped(keys=["image", "label"]),
        ]
    )

    return transform

def body_crop(input, crop_dict=None, reverse = False):
    # takes in input pytorch tensor and outputs cropped version (see: body crop)

    if not reverse:
        orig_size = input.shape

        img_data = input[0,:,:,:].cpu().numpy()

        proj = np.max(img_data,axis=2) # get projection

        x = proj > threshold_otsu(proj) # threshold
        x = binary_fill_holes(x).astype(int) # fill holes

        lbls = measure.label(x) # get label
        assert( lbls.max() != 0 ) # assume at least 1 CC
        x = lbls == np.argmax(np.bincount(lbls.flat)[1:])+1

        # get bounds
        nonzeros = np.argwhere(x>0) #get the coordinates of the image
        proj_mins = nonzeros.min(axis=0)
        proj_maxs = nonzeros.max(axis=0)

        # apply bounds
        out_img = np.expand_dims(img_data[proj_mins[0]:proj_maxs[0],proj_mins[1]:proj_maxs[1]],axis=[0])
        out_img = torch.from_numpy(out_img)

        crop_dict = {
            'orig_size': orig_size,
            'proj_mins': proj_mins,
            'proj_maxs': proj_maxs, 
        }

        return out_img, crop_dict
    
    else:

        orig_size = crop_dict['orig_size']
        proj_mins = crop_dict['proj_mins']
        proj_maxs = crop_dict['proj_maxs']

        out_img = torch.zeros(orig_size)
        out_img[0,proj_mins[0]:proj_maxs[0],proj_mins[1]:proj_maxs[1],:] = input
        
        return out_img

if __name__ == '__main__':

    from fingerprint import read_fingerprint
    from config import *

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('training on',device)

    img_path = 'D:\Rahul\Data\original_images\img\\001.nii'

    coarse_fingerprint = read_fingerprint('six_label')
    '''to_coarse = coarse_transform(coarse_fingerprint,COARSE_RES)

    coarse_img = to_coarse({'image':img_path})
    print(coarse_img['image'].shape)'''

