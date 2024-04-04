# test model performance

import torch
import monai
import pandas as pd
import monai.transforms as mt
from monai.metrics import compute_meandice
import numpy as np
import os
from paths import MODEL_BASE, DATA_BASE

from fingerprint import read_fingerprint
from data_loader import create_datasets
from transforms import nnUNet_transform
from models import GenericUnet
from swinmlp_unet import SwinMLPUNet 
from utils import crop_bladder_segmentation, multi_organ_dice, body_crop, normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('evaluating on',device)

# set up transforms and datasets/dataloaders
task_name = 'coarse'
patch_size = (112,112,48)
data_fingerprint = read_fingerprint(task_name)
train_transform,_,test_transform = nnUNet_transform(data_fingerprint,device=device,spatial_size=patch_size,pixdims=(1.7,1.7,3))
_, _, test_set = create_datasets(task_name,(train_transform,test_transform),cache_dir='coarse')
batch_size = 1
test_loader = monai.data.DataLoader(test_set,batch_size=1)

model_name = 'ACA_UNet'
model_type = 'coarse_segmentation'
model_path = os.path.join(MODEL_BASE,'model_weights',model_type,model_name+'.pt')
num_classes = 7
save = False # whether or not to save image

# model -> can be found in runs.txt
model = GenericUnet(
        1,
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

post_label = mt.Compose([
    mt.EnsureType(), 
    mt.AsDiscrete(to_onehot=num_classes)])

inverse_transform = mt.BatchInverseTransform(test_transform,test_loader)

# saver object
saver = mt.Compose([
    mt.SaveImaged(
        keys=['label'],
        meta_keys=['label_meta_dict'],
        separate_folder=False,
        output_postfix='pred',
        output_dir = os.path.join(DATA_BASE,'coarse_to_fine','ACAUNet_COARSE')
    )
])

dice_df = pd.DataFrame(columns=['prostate','bladder','rectum','seminal','left fh','right fh'])
haus_df = pd.DataFrame(columns=['prostate','bladder','rectum','seminal','left fh','right fh'])

with torch.no_grad():
    for i, batch in enumerate(test_loader):

        print(f'Image {i+1}/{len(test_loader)}',end='\r')

        # push to device
        inputs, labels = (batch["image"].to(device),batch["label"].to(device))
        img_name = os.path.basename(batch['image_meta_dict']['filename_or_obj'][0])

        cropped_inputs, cropping_dict = body_crop(inputs)
        inputs.set_array(cropped_inputs)
        inputs = inputs.to(device)
        inputs = normalize(inputs, data_fingerprint)

        gt_path = batch['label_meta_dict']['filename_or_obj'][0]
        ground_truth = np.expand_dims(mt.LoadImage()(gt_path)[0],0)
        ground_truth = torch.unsqueeze(post_label(ground_truth),0).to(device)

        prediction = torch.argmax(monai.inferers.sliding_window_inference(inputs,patch_size,3,model,overlap=0.5,mode='gaussian'),axis=1,keepdims=True)
        prediction.applied_operations = batch['label'].applied_operations

        prediction.set_array(body_crop(prediction.get_array(), cropping_dict, True))

        pred_dict = {'label':prediction}

        with monai.transforms.utils.allow_missing_keys_mode(test_transform):
            prediction = inverse_transform(pred_dict)[0]

        prediction = mt.AsDiscreted(keys=['label'],to_onehot=num_classes)(prediction)
        prediction = post_pred(prediction)

        prediction['label'] = crop_bladder_segmentation(prediction['label'])
        dice, haus = multi_organ_dice(prediction['label'],ground_truth[0])
        print(dice)
        dice_df.loc[img_name] = dice
        haus_df.loc[img_name] = haus

        if save:
            prediction['label'] = torch.unsqueeze(prediction['label'],0)
            prediction['label'] = torch.argmax(prediction['label'],1)
            saver(prediction)
        
    
dice_df.loc['mean'] = dice_df.mean()
dice_df.loc['std'] = dice_df.iloc[:-1,:].std()

haus_df.loc['mean'] = haus_df.mean()
haus_df.loc['std'] = haus_df.iloc[:-1,:].std()

print(dice_df)
print(haus_df)