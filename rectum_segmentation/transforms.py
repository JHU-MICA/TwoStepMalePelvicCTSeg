# this file includes a variety of transform configurations that could be applied as needed

import torch
import monai
import monai.transforms as mt
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, binary_erosion
from skimage import measure 
from copy import deepcopy

import matplotlib.pyplot as plt

from paths import DATA_BASE

class RoiCrop(mt.MapTransform,mt.InvertibleTransform):
    '''
    this transform uses the center of the label to crop both image and label with fixed size
    '''

    def __init__(self, keys, img_size):

        self.keys = keys
        self.img_size = img_size

    def __call__(self,data):
        
        d = dict(data)

        img_data = data['image'][0,:,:,:]
        lbl_data = data['label'][0,:,:,:]

        if not isinstance(img_data, np.ndarray):
            img_data = img_data.numpy()

        if not isinstance(lbl_data, np.ndarray):
            lbl_data = lbl_data.numpy()

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
        out_img = np.expand_dims(img_data[x_start:x_end,y_start:y_end,z_start:z_end],axis=0)

        out_lbl = np.expand_dims(lbl_data[x_start:x_end,y_start:y_end,z_start:z_end],axis=0)

        # ensures that the label data contains the whole roi
        '''if out_lbl.sum() != lbl_data.sum():
            print(lbl_data.sum()-out_lbl.sum(),lbl_data.sum())
            raise ValueError'''

        self.push_transform(d, 'image', extra_info={"crop_start": start_coords, "crop_end": end_coords})
        d['image'] = torch.from_numpy(out_img)

        self.push_transform(d, 'label', extra_info={"crop_start": start_coords, "crop_end": end_coords})
        d['label'] = torch.from_numpy(out_lbl)
            

        return d

    def inverse(self,data):

        d = deepcopy(data)

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)

            # Create inverse transform
            orig_size = transform[monai.utils.enums.InverseKeys.ORIG_SIZE]
            extra_info = transform[monai.utils.enums.InverseKeys.EXTRA_INFO]
            start_coords = np.asarray(extra_info["crop_start"])
            end_coords = np.asarray(extra_info["crop_end"])

            # first create original array
            original = torch.zeros(orig_size)
            original[start_coords[0]:end_coords[0],start_coords[1]:end_coords[1],start_coords[2]:end_coords[2]] = torch.Tensor(d[key])
            original = torch.unsqueeze(original,0)

            d[key] = original

            # Remove the applied transform
            self.pop_transform(d, key)

        return d

        

class nnUNetTransform(mt.MapTransform):
    '''
    this transform clips and normalizes images using fingerprint dictionary
    '''

    def __init__(self, keys,fingerprint):

        self.keys = keys
        self.fingerprint = fingerprint

    def __call__(self,data):

        data['image'] = torch.clip(data['image'],min=self.fingerprint['0.5 percentile'],max=self.fingerprint['99.5 percentile'])

        data['image'] = (data['image'] - self.fingerprint['mean']) / self.fingerprint['std']

        return data



def nnUNet_transform(data_fingerprint,device='cuda:0',spatial_size=(64,64,64),pixdims=(1.7,1.7,1.7)):
    '''
    creates nnUNET transform pipeline for fixed size images
    '''

    train_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]), # load image
            mt.EnsureChannelFirstd(keys=["image", "label"]), # make channel first
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"), # adjust orientation
            mt.Spacingd(keys=["image", "label"], pixdim=pixdims, mode=("bilinear", "nearest")), # change spacing
            RoiCrop(keys=["image", "label"], img_size=spatial_size), # crop around roi
            mt.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0), # random flipping
            mt.RandAffined( # random affine
                keys=["image", "label"],
                mode=('bilinear', 'nearest'),
                prob=0.2, # increased
                spatial_size=spatial_size,
                rotate_range=(np.pi/18, np.pi/18, np.pi/18), # from 36 -> 18, added z
                scale_range=(0.05, 0.05, 0), 
                translate_range=(10, 10, 5), # changed to (10,10,5)
                device=device,
            ),
            nnUNetTransform(keys=["image"],fingerprint=data_fingerprint), # clip and normalize
            mt.EnsureTyped(keys=["image", "label"]),
        ]
    )

    test_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]), # load image
            mt.EnsureChannelFirstd(keys=["image", "label"]), # make channel first
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"), # adjust orientation
            mt.Spacingd(keys=["image", "label"], pixdim=pixdims, mode=("bilinear", "nearest")), # change spacing
            RoiCrop(keys=["image", "label"], img_size=spatial_size), # crop around roi
            nnUNetTransform(keys=["image"],fingerprint=data_fingerprint), # clip and normalize
            mt.EnsureTyped(keys=["image", "label"]),
        ]
    )

    return train_transform,test_transform