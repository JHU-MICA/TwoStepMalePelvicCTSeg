# applies coarse segmentation model to all images in dataset 

import torch
import monai
import pandas as pd
import monai.transforms as mt
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from copy import deepcopy

from paths import MODEL_BASE, DATA_BASE
from fingerprint import read_fingerprint
from data_loader import create_datasets
from transforms import nnUNet_transform, coarse_transform
from models import GenericUnet

def apply_model(
    data_dicts,
    out_dir = os.path.join(DATA_BASE,'coarse_preds'),
    lbls = ['prostate','bladder','rectum','sigmoid','seminal_vesicle','femoral_heads'],
    ):
    # pass dictionary of data and evaluate model
    # data_dicts: list of dictionaries to pass into model
    # out_dir: output directory to save images
    # lbls: name of labels

    [os.makedirs(os.path.join(out_dir,lbl)) for lbl in lbls if not os.path.exists(os.path.join(out_dir,lbl))]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('evaluating on',device)

    # set up transforms 
    num_classes = 7
    patch_size = (96,96,64)

    data_fingerprint = read_fingerprint('six_label')
    _, transform = nnUNet_transform(data_fingerprint,device=device,spatial_size=patch_size,pixdims=(1.7,1.7,3))


    # load model
    model_name = 'patch_coatUNet'
    model_type = 'patch_based_coarse'
    model_path = os.path.join(MODEL_BASE,'model_weights',model_type,model_name+'.pt')

    # model -> can be found in runs.txt
    model = GenericUnet(1,7,(96,96,64),norm_name='BATCH',num_layers=5,encoder_block='coat',encoder_units=3)

    # load in the weights and push to device for evaluation
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in data_dicts:

            img_name = os.path.basename(batch['image'])[:3]
            
            batch = transform(batch)

            print(f'Image {img_name}',end='\r')

            # push to device
            inputs = torch.unsqueeze(batch["image"].to(device),0)

            # evaluate model and apply softmax
            pred = torch.softmax(
                monai.inferers.sliding_window_inference(inputs,patch_size,1,model,overlap=0.5),1)

            for i,lbl in enumerate(lbls):
                save_path = os.path.join(out_dir,lbl)

                batch['label'] = pred[0,i+1]
                
                inv_batch = transform.inverse(batch)

                # save image
                mt.SaveImaged(
                    keys=['label'],
                    meta_keys=['label_meta_dict'],
                    separate_folder=False,
                    output_postfix='',
                    output_dir = save_path
                )(inv_batch)
            



if __name__ == '__main__':
    # get image data paths
    img_paths = glob.glob('D:\Rahul\Data\original_images\img\*.nii')
    
    data_dicts = [{'image':path,'label':path} for path in img_paths]

    apply_model(data_dicts)