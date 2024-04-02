# gets segmentation metrics for images

import os
import glob

import pandas as pd
import numpy as np
from medpy import metric
import nibabel as nib

def multi_organ_metrics(pred,true,num_rois,pixdims):

    pred_onehot = [pred==i for i in range(num_rois+1)]
    gt_onehot = [true==i for i in range(num_rois+1)]

    # rectum #
    z_crop = np.argwhere(np.equal(gt_onehot[3],1)).max(axis=0)[-1]
    pred_onehot[3][:,:,z_crop+1:] = 0

    # left fh #
    z_crop = np.argwhere(np.equal(gt_onehot[5],1)).min(axis=0)[-1]
    pred_onehot[5][:,:,:z_crop] = 0

    # right fh #
    z_crop = np.argwhere(np.equal(gt_onehot[6],1)).min(axis=0)[-1]
    pred_onehot[6][:,:,:z_crop] = 0

    dice = []
    hd95 = []

    for cls in range(1,num_rois+1):
        dice.append(metric.binary.dc(pred_onehot[cls],gt_onehot[cls]))
        hd95.append(metric.binary.hd95(pred_onehot[cls],gt_onehot[cls],pixdims))
    
    return dice, hd95


model_names = [
    'Enhancement_RESUNET',
    'Enhancement_DenseUNET',
    'Enhancement_UNET_MT',
    'Enhancement_ResUNET_MT',
    'Enhancement_DenseUNET_MT',
]

test_imgs = ['001','002','003','004','005','006','009','010','011','012','013','014','015']
lbl_paths = ['D:\Rahul\AugmentedData\PelvicData\\5_organ_labels\\{}-label.nii'.format(x) for x in test_imgs]

pred_base = 'D:\Rahul\Program\Torch_PelvicCTSegmentation\\data_files\\coarse_to_fine' 
csv_base = 'D:\Rahul\Program\Torch_PelvicCTSegmentation\\data_files\\coarse_to_fine_analysis' 

for model_name in model_names:

    print('Model: {}'.format(model_name))

    pred_paths = [os.path.join(pred_base,model_name)+'\\{}_pred.nii.gz'.format(x) for x in test_imgs]

    csv_path = os.path.join(csv_base,'{}.xlsx'.format(model_name))

    dice_df = pd.DataFrame(columns=['prostate','bladder','rectum','seminal vesicles','left femoral heads', 'right femoral heads'])
    haus_df = pd.DataFrame(columns=['prostate','bladder','rectum','seminal vesicles','left femoral heads', 'right femoral heads'])

    for i in range(len(pred_paths)):

        gt_label = nib.load(lbl_paths[i])
        pred_label = nib.load(pred_paths[i])

        gt_array = gt_label.get_fdata()
        pred_array = pred_label.get_fdata()

        img_spacing = gt_label.header['pixdim'][1:4]

        dice, hd95 = multi_organ_metrics(pred_array, gt_array, 6, img_spacing) 

        print(dice)
        dice_df.loc[test_imgs[i]] = dice
        haus_df.loc[test_imgs[i]] = hd95

    with pd.ExcelWriter(csv_path) as writer:
        dice_df.to_excel(writer, sheet_name='Dice')
        haus_df.to_excel(writer, sheet_name='95HD')