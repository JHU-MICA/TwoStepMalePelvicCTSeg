import torch
import monai.transforms as mt
from monai.metrics import compute_meandice, compute_hausdorff_distance
import numpy as np
    
def multi_organ_dice(pred,truth):
    '''
    takes in prediction and ground truth and returns dice score
    cuts at appropriate height
    '''
    
    y_pred = pred['label'].cpu().numpy()
    y_true = truth['label'].cpu().numpy()

    # rectum #
    z_crop = np.argwhere(np.equal(y_true[3],1)).max(axis=0)[-1]
    y_pred[3][:,:,z_crop+1:] = 0

    # left fh #
    z_crop = np.argwhere(np.equal(y_true[5],1)).min(axis=0)[-1]
    y_pred[5][:,:,:z_crop] = 0

    # right fh #
    z_crop = np.argwhere(np.equal(y_true[6],1)).min(axis=0)[-1]
    y_pred[6][:,:,:z_crop] = 0

    y_pred = torch.unsqueeze(torch.Tensor(y_pred),0)
    y_true = torch.unsqueeze(torch.Tensor(y_true),0)

    dices = compute_meandice(y_pred,y_true,include_background=False)[0].cpu().numpy()
    hauses = compute_hausdorff_distance(y_pred,y_true,include_background=False,percentile=95)[0].cpu().numpy()

    return dices, hauses