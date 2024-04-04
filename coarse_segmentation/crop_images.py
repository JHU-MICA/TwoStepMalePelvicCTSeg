# use the coarse segmentation pipeline to crop images based on coarse model #

import torch
import monai
import monai.transforms as mt
import numpy as np
import os
import glob
from copy import deepcopy

from paths import MODEL_BASE, DATA_BASE
from fingerprint import read_fingerprint
from data_loader import create_datasets
from transforms import crop_transform
from models import GenericUnet


def roi_crop(img_data, lbl_data, pred_data, img_size):

    '''
    this transform uses the center of the predicted label to crop both image and label with fixed size
    expects NO CHANNEL input
    adds CHANNEL back

    img_data: image data to crop
    lbl_data: label data to crop
    pred_data: predicted label data to use to crop
    img_size: tuple, size of image
    '''

    if not isinstance(img_data, np.ndarray):
        img_data = img_data.cpu().numpy()

    if not isinstance(lbl_data, np.ndarray):
        lbl_data = lbl_data.cpu().numpy()

    if not isinstance(pred_data, np.ndarray):
        pred_data = pred_data.cpu().numpy()

    # get foreground points and corresponding values
    foreground_coords = np.argwhere(np.greater(pred_data,0))

    # get distance from centroid (to see how large to crop)
    foreground_center = np.round(foreground_coords.mean(axis=0))
    x_start = int(max(foreground_center[0] - img_size[0] // 2,0))
    y_start = int(max(foreground_center[1] - img_size[1] // 2,0))
    z_start = int(max(foreground_center[2] - img_size[2] // 2,0))

    x_end = int(min(x_start + img_size[0],pred_data.shape[0]))
    y_end = int(min(y_start + img_size[1],pred_data.shape[1]))
    z_end = int(min(z_start + img_size[2],pred_data.shape[2]))

    start_coords = [x_end-img_size[0],y_end-img_size[1],z_end-img_size[2]]
    end_coords = [x_end,y_end,z_end]

    # apply bounds
    out_img = np.expand_dims(img_data[start_coords[0]:end_coords[0],start_coords[1]:end_coords[1],start_coords[2]:end_coords[2]],axis=0)
    out_lbl = np.expand_dims(lbl_data[start_coords[0]:end_coords[0],start_coords[1]:end_coords[1],start_coords[2]:end_coords[2]],axis=0)

    return torch.Tensor(out_img), torch.Tensor(out_lbl)

def nnUnet_transform(img,fingerprint):
    '''
    this transform clips and normalizes images using fingerprint dictionary
    '''
    x = img.clone()
    out = torch.clip(x,min=fingerprint['0.5 percentile'],max=fingerprint['99.5 percentile'])
    out = (out - fingerprint['mean']) / fingerprint['std']

    return out

if __name__ == '__main__':

    # lbl_type: name, label number in coarse, crop size
    lbl_type = ('prostate',1,(64,64,48))
    lbl_base = [os.path.basename(x) for x in glob.glob(f'path/to/original_data/{lbl_type[0]}/*.nii')]

    img_paths = [os.path.join('path/to/original_data/img',x) for x in lbl_base]
    lbl_paths = [os.path.join(f'path/to/original_data/{lbl_type[0]}',x) for x in lbl_base]

    out_base = os.path.join(DATA_BASE,'cropped_data',f'{lbl_type[0]}_{lbl_type[2][0]}_{lbl_type[2][1]}_{lbl_type[2][2]}')

    data_dicts = [{'image':img_path,'label':lbl_path} for img_path,lbl_path in zip(img_paths,lbl_paths)]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('evaluating on',device)

    # set up transforms
    task_name = 'five_label'
    patch_size = (112,112,48)
    data_fingerprint = read_fingerprint(task_name)
    test_transform = crop_transform(pixdims=(1.7,1.7,3))

    model_name = 'five_lbl_axial_unet_weighted'
    model_type = 'patch_based_coarse'
    model_path = os.path.join(MODEL_BASE,'model_weights',model_type,model_name+'.pt')
    num_classes = 7

    # model -> can be found in runs.txt
    model = GenericUnet(1,
        num_classes,
        patch_size,
        norm_name='INSTANCE',
        num_layers=4,
        encoder_block='axial',
        encoder_units=3,
        )

    # load in the weights and push to device for evaluation
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # eval set up
    post_pred = mt.Compose([
        mt.EnsureTyped(keys=['label']), 
        mt.KeepLargestConnectedComponentd(keys=['label'],applied_labels=[1,2,3,4,5,6],),
        ])


    with torch.no_grad():
        for batch in data_dicts:

            img_name = os.path.basename(batch['image'])[:3]
            print(f'Image {img_name}',end='\r')

            # load in image
            batch = test_transform(batch)

            # push to device
            inputs, labels = (batch["image"].to(device),batch["label"].to(device))

            inputs = torch.unsqueeze(inputs,0)

            coarse_pred = torch.squeeze(monai.inferers.sliding_window_inference(
            nnUnet_transform(inputs,data_fingerprint),
            patch_size,
            1,
            model,
            overlap=0.5,
            ))

            coarse_pred = torch.argmax(coarse_pred,axis=0,keepdim=True)
            coarse_pred = mt.AsDiscrete(to_onehot=7)(coarse_pred)
            coarse_pred = mt.KeepLargestConnectedComponent(applied_labels=[1,3,4,5,6])(coarse_pred)

            cropped_input, cropped_lbl = roi_crop(torch.squeeze(inputs),torch.squeeze(labels),coarse_pred[lbl_type[1]], lbl_type[2])
            
            batch['image'] = cropped_input
            batch['label'] = cropped_lbl

            mt.SaveImaged(
                keys=['image'],
                meta_keys=['image_meta_dict'],
                separate_folder=False,
                output_postfix='',
                resample=False,
                output_dir = os.path.join(out_base,'image')
                )(batch)

            mt.SaveImaged(
                keys=['label'],
                meta_keys=['label_meta_dict'],
                separate_folder=False,
                output_postfix='',
                resample=False,
                output_dir = os.path.join(out_base,'label')
                )(batch)

