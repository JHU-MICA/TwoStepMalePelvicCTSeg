import torch
import monai.transforms as mt
from monai.metrics import compute_meandice, compute_hausdorff_distance
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, binary_erosion
from skimage import measure 


def crop_bladder_segmentation(pred):
    # takes in prediction (one hot encoded) and crops bladder to be within limits of femurs

    pred[2,:,:,:10] = 0; pred[2,:,:,-10:] = 0
    
    return pred

def normalize(input, fingerprint):
    # nnunet normalization using fingerprint file
    output = torch.clip(input,min=fingerprint['0.5 percentile'],max=fingerprint['99.5 percentile'])

    output = (output - fingerprint['mean']) / fingerprint['std']

    return output

def body_crop(input, crop_dict=None, reverse = False):
    # takes in input pytorch tensor and outputs cropped version (see: body crop)

    if not reverse:
        orig_size = input.shape

        img_data = input[0,0,:,:,:].cpu().numpy()

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
        out_img = np.expand_dims(img_data[proj_mins[0]:proj_maxs[0],proj_mins[1]:proj_maxs[1]],axis=[0,1])
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
        out_img[0,0,proj_mins[0]:proj_maxs[0],proj_mins[1]:proj_maxs[1],:] = torch.from_numpy(input[0,0])
        
        return out_img

    
def multi_organ_dice(pred,truth):
    '''
    takes in prediction and ground truth and returns dice score
    cuts at appropriate height
    '''
    
    y_pred = pred.cpu().numpy()
    y_true = truth.cpu().numpy()

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