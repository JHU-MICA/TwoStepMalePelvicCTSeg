# main driver file for the coarse to fine segmentation #

import os

import torch
import monai
import monai.transforms as mt
from monai.transforms.utils import allow_missing_keys_mode
import numpy as np

import config
from fingerprint import read_fingerprint
from transforms import resample_transform, nnUnet_transform, RoiCrop, body_crop
from enhance import enhance_image
from models import GenericUnet
from swinmlp_unet import SwinMLPUNet

def coarse_to_fine(img_path,save=False,save_path=None,seminal_fine=True,device='cuda:0'):

    '''
    img_path: individual img to be segmented
    save: whether or not to save the image
    save_path: path to save it at
    seminal_fine: whether or not to use fine segmentation for final model output
    takes in either a list of img paths or an individual img path and segments it
    '''

    print('evaluating on',device)

    if save:
        if save_path is None:
            save_path = os.path.join(config.DATA_BASE,'coarse_to_fine')
            print('No save path found, will default to {}'.format(save_path))
        else:
            print('Will save to {}'.format(save_path))
    
    ## READ IN FINGERPRINTS ##
    coarse_fingerprint = read_fingerprint('five_label')
    prostate_fingerprint = read_fingerprint('prostate_enhanced')
    rectum_fingerprint = read_fingerprint('rectum_enhanced')
    seminal_fingerprint = read_fingerprint('seminal_enhanced')
    femur_fingerprint = read_fingerprint('femur_enhanced')

    # ASSUME ALL PIXDIMS ARE THE SAME
    pixdims = config.COARSE_RES

    # create resampled images
    resampler = resample_transform(pixdims)
    
    resampled_img = resampler({'image':[img_path],'label':[img_path]})

    cropped_resampled_img, crop_dict = body_crop(resampled_img['image'])

    resampled_img['image'].set_array(cropped_resampled_img)

    ## COARSE SEGMENTATION ##
    model = GenericUnet(**config.COARSE_CONFIG)
    model.load_state_dict(torch.load(config.COARSE_WEIGHTS))
    model.to(device)
    model.eval()

    inputs = torch.unsqueeze(nnUnet_transform(resampled_img['image'],coarse_fingerprint),0).to(device)

    with torch.no_grad():
        coarse_lbl = torch.squeeze(monai.inferers.sliding_window_inference(
            inputs,
            config.COARSE_DIMS,
            1,
            model,
            overlap=0.5,
            mode='gaussian'
        ))

    enhancer = torch.softmax(coarse_lbl, 0).cpu().numpy()

    # output from coarse segmentation: one-hot format resampled image
    coarse_lbl = torch.argmax(coarse_lbl,axis=0,keepdim=True)
    coarse_lbl = mt.AsDiscrete(to_onehot=7)(coarse_lbl)
    coarse_lbl = mt.KeepLargestConnectedComponent(applied_labels=[1,3,4,5,6])(coarse_lbl)

    ## PROSTATE SEGMENTATION ##
    model = GenericUnet(**config.PROSTATE_CONFIG)
    model.load_state_dict(torch.load(config.PROSTATE_WEIGHTS))
    model.to(device)
    model.eval()

    cropper = RoiCrop(config.PROSTATE_DIMS)

    inputs = resampled_img['image'][0]

    inputs = enhance_image(inputs, enhancer[1])
    inputs = cropper(inputs,coarse_lbl[1])
    inputs = torch.unsqueeze(nnUnet_transform(inputs,prostate_fingerprint),0).to(device)
    with torch.no_grad():
        prostate = torch.squeeze(model(inputs))

    prostate = torch.argmax(prostate,0).cpu()

    prostate = cropper.inverse(prostate)

    print('PROSTATE SEGMENTATION DONE')

    ## RECTUM SEGMENTATION ##
    model = GenericUnet(**config.RECTUM_CONFIG)
    model.load_state_dict(torch.load(config.RECTUM_WEIGHTS))
    model.to(device)
    model.eval()

    cropper = RoiCrop(config.RECTUM_DIMS)

    inputs = resampled_img['image'][0]
    inputs = enhance_image(inputs, enhancer[3])
    inputs = cropper(inputs,coarse_lbl[3])
    inputs = torch.unsqueeze(nnUnet_transform(inputs,rectum_fingerprint),0).to(device)
    with torch.no_grad():
        rectum = torch.squeeze(model(inputs))

    rectum = torch.argmax(rectum,0).cpu()

    rectum = cropper.inverse(rectum)

    print('RECTUM SEGMENTATION DONE')

    ## SEMINAL VESICLE SEGMENTATION ##
    if seminal_fine == True:
        model = GenericUnet(**config.SEMINAL_CONFIG)
        model.load_state_dict(torch.load(config.SEMINAL_WEIGHTS))
        model.to(device)
        model.eval()

        cropper = RoiCrop(config.SEMINAL_DIMS)

        inputs = resampled_img['image'][0]
        inputs = enhance_image(inputs, enhancer[4])
        inputs = cropper(inputs,coarse_lbl[4])
        inputs = torch.unsqueeze(nnUnet_transform(inputs,seminal_fingerprint),0).to(device)
        with torch.no_grad():
            seminal = torch.squeeze(model(inputs))

        seminal = torch.argmax(seminal,0).cpu()

        seminal = cropper.inverse(seminal)
    
    else:
        seminal = coarse_lbl[4].cpu()

    print('SEMINAL VESICLE SEGMENTATION DONE')

    ## LEFT FEMORAL HEAD SEGMENTATION ##
    model = GenericUnet(**config.FEMUR_CONFIG)
    model.load_state_dict(torch.load(config.FEMUR_WEIGHTS))
    model.to(device)
    model.eval()

    cropper = RoiCrop(config.FEMUR_DIMS)

    inputs = resampled_img['image'][0]
    inputs = enhance_image(inputs, enhancer[5])
    inputs = cropper(inputs,coarse_lbl[5])
    inputs = torch.unsqueeze(nnUnet_transform(inputs,femur_fingerprint),0).to(device)
    with torch.no_grad():
        lt_femur = torch.squeeze(model(inputs))

    lt_femur = torch.argmax(lt_femur,0).cpu()

    lt_femur = cropper.inverse(lt_femur)

    inputs = resampled_img['image'][0]
    inputs = enhance_image(inputs, enhancer[6])
    inputs = cropper(inputs,coarse_lbl[6])
    inputs = torch.unsqueeze(nnUnet_transform(inputs,femur_fingerprint),0).to(device)
    with torch.no_grad():
        rt_femur = torch.squeeze(model(inputs))

    rt_femur = torch.argmax(rt_femur,0).cpu()

    rt_femur = cropper.inverse(rt_femur)

    print('FEMORAL HEAD SEGMENTATION DONE')

    # add individual segmentations to coarse label
    out_lbl = prostate
    out_lbl += 2 * torch.clip((coarse_lbl[2].cpu() - out_lbl),0,1) # add bladder
    out_lbl += 3 * torch.clip((rectum - out_lbl),0,1) # add rectum
    out_lbl += 4 * torch.clip((seminal - out_lbl),0,1) # add seminal vesicles
    out_lbl += 5 * torch.clip((lt_femur - out_lbl),0,1) # add femoral heads
    out_lbl += 6 * torch.clip((rt_femur - out_lbl),0,1) # add femoral heads

    out_lbl = body_crop(out_lbl, crop_dict, True)

    resampled_img['label'].set_array(out_lbl)
    resampled_img.pop('image', None)

    with monai.transforms.utils.allow_missing_keys_mode(resampler):
        output = resampler.inverse(resampled_img)

    output = mt.AsDiscreted(keys=['label'],to_onehot=7)(output)
    output['label'][2,:,:,:10] = 0; output['label'][2,:,:,-10:] = 0 # MAKE SURE BLADDER IS REASONABLE
    output = mt.KeepLargestConnectedComponentd(keys=['label'],applied_labels=[1,2,3,4,5,6])(output)
    output['label'] = torch.argmax(output['label'],0,keepdims=True)
    
    if save:
        mt.SaveImaged(
        keys=['label'],
        meta_keys=['label_meta_dict'],
        separate_folder=False,
        output_postfix='pred',
        output_dir = save_path
        )(output)

    return np.squeeze(output['label'].cpu().numpy())
