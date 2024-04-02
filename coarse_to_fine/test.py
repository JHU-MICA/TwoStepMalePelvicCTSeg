# test model performance

import torch
import monai
import pandas as pd
import monai.transforms as mt
from monai.metrics import compute_meandice
import numpy as np
import os

from main import coarse_to_fine
import config
from test_utils import multi_organ_dice

run_name = 'TestOut'

test_imgs = ['001','002','003','004','005','006','009','010','011','012','013','014','015']
img_paths = ['D:\Rahul\Data\original_images\img\\{}.nii'.format(x) for x in test_imgs]
lbl_paths = ['D:\Rahul\AugmentedData\PelvicData\\5_organ_labels\\{}-label.nii'.format(x) for x in test_imgs]
pred_paths = [os.path.join(config.DATA_BASE,'coarse_to_fine',run_name)+'\\{}_pred.nii.gz'.format(x) for x in test_imgs]

create_lbls = True # whether or not to rerun the label making

if create_lbls:
    for img in img_paths:
        print(os.path.basename(img))
        coarse_to_fine(img,True,os.path.join(config.DATA_BASE,'coarse_to_fine',run_name))

label_transform = mt.Compose(
        [
            mt.LoadImaged(keys=['label']), # load image
            mt.EnsureChannelFirstd(keys=["label"]), # make channel first
            mt.EnsureTyped(keys=['label']), # make sure type is okay
            mt.AsDiscreted(keys=['label'],to_onehot=7)
        ]
    )

saver = mt.Compose([
    mt.SaveImaged(
        keys=['label'],
        meta_keys=['label_meta_dict'],
        separate_folder=False,
        output_postfix='pred',
        output_dir = os.path.join(config.DATA_BASE,'coarse_to_fine',run_name)
    )
])

dice_df = pd.DataFrame(columns=['prostate','bladder','rectum','seminal','left fh','right fh'])
haus_df = pd.DataFrame(columns=['prostate','bladder','rectum','seminal','left fh','right fh'])

for (y_true,y_pred) in zip(lbl_paths,pred_paths):

    print(f'Image {os.path.basename(y_true)}',end='\r')

    true = label_transform({'label': y_true})
    prediction = label_transform({'label': y_pred})
    
    dice, haus = multi_organ_dice(prediction,true)

    print(dice)

    dice_df.loc[os.path.basename(y_true)] = dice
    haus_df.loc[os.path.basename(y_true)] = haus

    '''prediction['label'] = torch.unsqueeze(prediction['label'],0)
    prediction['label'] = torch.argmax(prediction['label'],1)
    saver(prediction)'''


dice_df.loc['mean'] = dice_df.mean()
dice_df.loc['std'] = dice_df.iloc[:-1,:].std()

haus_df.loc['mean'] = haus_df.mean()
haus_df.loc['std'] = haus_df.iloc[:-1,:].std()

print(dice_df)
print(haus_df)
    
