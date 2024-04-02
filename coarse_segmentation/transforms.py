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

class BodyCrop(mt.MapTransform,mt.InvertibleTransform):
    '''
    this transform uses the body coordinates from the image to crop both image and label
    method: max projection -> threshold -> fill any holes -> get coords -> get bounding rectangle
    '''

    def __call__(self,data):
        
        d = dict(data)

        img_data = data['image'][0,:,:,:]

        if not isinstance(img_data, np.ndarray):
            img_data = img_data.numpy()

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
        out_img = np.expand_dims(img_data[proj_mins[0]:proj_maxs[0],proj_mins[1]:proj_maxs[1]],axis=0)

        self.push_transform(d, 'image', extra_info={"body_start": proj_mins, "body_end": proj_maxs})
        d['image'] = torch.from_numpy(out_img)

        if 'label' in self.keys:
            lbl_data = data['label'][0,:,:,:]

            if not isinstance(lbl_data, np.ndarray):
                lbl_data = lbl_data.numpy()

            out_lbl = np.expand_dims(lbl_data[proj_mins[0]:proj_maxs[0],proj_mins[1]:proj_maxs[1]],axis=0)

            self.push_transform(d, 'label', extra_info={"body_start": proj_mins, "body_end": proj_maxs})
            d['label'] = torch.from_numpy(out_lbl)

        return d

    def inverse(self,data):

        d = deepcopy(data)

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)

            # Create inverse transform
            orig_size = transform[monai.utils.enums.InverseKeys.ORIG_SIZE]
            extra_info = transform[monai.utils.enums.InverseKeys.EXTRA_INFO]
            proj_mins = np.asarray(extra_info["body_start"])
            proj_maxs = np.asarray(extra_info["body_end"])

            # first create original array
            original = torch.zeros(orig_size)
            original[proj_mins[0]:proj_maxs[0],proj_mins[1]:proj_maxs[1],:] = torch.Tensor(d[key])
            original = torch.unsqueeze(original,0)

            d[key] = original

            # Remove the applied transform
            self.pop_transform(d, key)

        return d

        

class nnUNetTransform(mt.MapTransform):
    '''
    this transform clips and normalizes images using fingerprint dictionary
    '''

    def __init__(self, keys, fingerprint):

        self.keys = keys
        self.fingerprint = fingerprint
        self.allow_missing_keys = True

    def __call__(self,data):

        data['image'] = torch.clip(data['image'],min=self.fingerprint['0.5 percentile'],max=self.fingerprint['99.5 percentile'])

        data['image'] = (data['image'] - self.fingerprint['mean']) / self.fingerprint['std']

        return data



def nnUNet_transform(data_fingerprint,device='cuda:0',spatial_size=(112,112,48),pixdims=(2.3,2.3,3)):
    '''
    creates nnUNET transform pipeline 
    '''

    train_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]), # load image
            mt.EnsureChannelFirstd(keys=["image", "label"]), # make channel first
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"), # adjust orientation
            mt.Spacingd(keys=["image", "label"], pixdim=pixdims, mode=("bilinear", "nearest")), # change spacing
            mt.CropForegroundd(keys=["image", "label"], source_key="image", select_fn = lambda x: x>-1000), #  crop foreground
            BodyCrop(keys=["image", "label"]), # crop body
            mt.RandCropByPosNegLabeld( # random cropping for patch-based approach
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=2,neg=1,
            ),
            mt.RandFlipd(keys=["image", "label"],prob=0.2), # random flipping
            mt.RandAffined( # random affine
                keys=["image", "label"],
                mode=('bilinear', 'nearest'),
                prob=0.5,
                spatial_size=spatial_size,
                rotate_range=(np.pi/18, np.pi/18, np.pi/36),
                scale_range=(0.05, 0.05, 0.05),
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
            mt.CropForegroundd(keys=["image", "label"], source_key="image", select_fn = lambda x: x>-1000), #  crop foreground
            BodyCrop(keys=["image", "label"]), # crop body
            nnUNetTransform(keys=["image"],fingerprint=data_fingerprint), # clip and normalize
            mt.EnsureTyped(keys=["image", "label"]),
        ]
    )

    # remove nnunet and body crop transforms (do manually)
    eval_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]), # load image
            mt.EnsureChannelFirstd(keys=["image", "label"]), # make channel first
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"), # adjust orientation
            mt.Spacingd(keys=["image", "label"], pixdim=pixdims, mode=("bilinear", "nearest")), # change spacing
            mt.CropForegroundd(keys=["image", "label"], source_key="image", select_fn = lambda x: x>-1000), #  crop foreground
            mt.EnsureTyped(keys=["image", "label"]),
        ]
    )

    return train_transform,test_transform,eval_transform

def coarse_transform(data_fingerprint,device='cuda:0',img_dims=(112,112,48)):
    '''
    creates nnUNET transform pipeline 
    '''

    train_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]), # load image
            mt.EnsureChannelFirstd(keys=["image", "label"]), # make channel first
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"), # adjust orientation
            mt.CropForegroundd(keys=["image", "label"], source_key="image", select_fn = lambda x: x>-1000), #  crop foreground
            BodyCrop(keys=["image", "label"]), # crop body
            mt.Resized(keys=["image", "label"], spatial_size=img_dims, mode=("trilinear", "nearest")), # change spacing
            mt.RandFlipd(keys=["image", "label"],prob=0.2), # random flipping
            mt.RandAffined( # random affine
                keys=["image", "label"],
                mode=('bilinear', 'nearest'),
                prob=0.5,
                spatial_size=img_dims,
                rotate_range=(np.pi/18, np.pi/18, np.pi/36),
                scale_range=(0.05, 0.05, 0.05),
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
            mt.CropForegroundd(keys=["image", "label"], source_key="image", select_fn = lambda x: x>-1000), #  crop foreground
            BodyCrop(keys=["image", "label"]), # crop body
            mt.Resized(keys=["image", "label"], spatial_size=img_dims, mode=("trilinear", "nearest")), # change spacing
            nnUNetTransform(keys=["image"],fingerprint=data_fingerprint), # clip and normalize
            mt.EnsureTyped(keys=["image", "label"]),
        ]
    )

    return train_transform,test_transform

class nnUNetTransform(mt.MapTransform):
    '''
    this transform clips and normalizes images using fingerprint dictionary
    '''

    def __init__(self, keys,fingerprint):

        self.keys = keys
        self.fingerprint = fingerprint
        self.allow_missing_keys = True

    def __call__(self,data):

        data['image'] = torch.clip(data['image'],min=self.fingerprint['0.5 percentile'],max=self.fingerprint['99.5 percentile'])

        data['image'] = (data['image'] - self.fingerprint['mean']) / self.fingerprint['std']

        return data


def crop_transform(pixdims=(2.3,2.3,3)):
    '''
    creates transform pipeline for cropping: no nnUNet preprocessing 
    '''

    test_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]), # load image
            mt.EnsureChannelFirstd(keys=["image", "label"]), # make channel first
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"), # adjust orientation
            mt.Spacingd(keys=["image", "label"], pixdim=pixdims, mode=("bilinear", "nearest")), # change spacing
            BodyCrop(keys=["image", "label"]),
            mt.EnsureTyped(keys=["image", "label"]),
        ]
    )

    return test_transform

if __name__ == '__main__':

    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split

    from fingerprint import read_fingerprint
    from data_loader import create_datasets

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # set up transforms and datasets/dataloaders
    task_name = 'six_label'
    data_fingerprint = read_fingerprint(task_name)
    train_transform,test_transform = originalRes_transform(data_fingerprint,device=device)
    train_dataset, val_dataset, test_dataset = create_datasets(task_name,(train_transform,test_transform),cache_dir='six_label_highRes_patch')

    train_loader = monai.data.DataLoader(train_dataset,batch_size=1)

    data = monai.utils.first(train_loader)
    img = data['image']
    lbl = data['label']

    print(img[0].shape)

    saver = mt.SaveImaged(
        keys=['label'],
        meta_keys=['label_meta_dict'],
        separate_folder=False,
        output_postfix='pred',
        output_dir = os.path.join(DATA_BASE,'examples')
        )
    
    

    '''inverse_transform = mt.BatchInverseTransform(train_transform,test_loader)
    inverse_batch = inverse_transform(data)
    example = inverse_batch[0]
    saver(example)
    print(example.keys())'''

    # plot image
    monai.visualize.utils.matshow3d(
        volume=img[0],
        fig=None, title="input image",
        figsize=(5,5),
        every_n=10,
        frame_dim=-1,
        show=True,
        cmap="gray",
    )

    # plot label
    monai.visualize.utils.matshow3d(
        volume=lbl[0],
        fig=None, title="input image",
        figsize=(5,5),
        every_n=10,
        frame_dim=-1,
        show=True,
    )

    # plot overlay

    img_data = torch.squeeze(img[0],0).numpy()
    lbl_data = torch.squeeze(lbl[0],0).numpy()

    plt.imshow(np.max(img_data,axis=2),cmap='gray')
    plt.imshow(np.max(lbl_data,axis=2),alpha=0.5,cmap='jet')
    plt.show()

    post_label = mt.AsDiscrete(to_onehot=7)
    print(post_label(lbl[0]).shape)