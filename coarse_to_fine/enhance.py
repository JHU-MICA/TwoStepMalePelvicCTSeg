import torch
from scipy.ndimage import gaussian_filter
import numpy as np

def enhance_image(img,softmax,a=0.5,b=2,sigma=5):
    # enhance torch tensor using numpy softmax values
    # img: input image array
    # softmax: input softmax array
    # a: minimum value of mask
    # b: maximum value of mask
    # sigma: sigma for gaussian
    
    img_np = img.cpu().numpy()
    enhancer = gaussian_filter(softmax,5)
    enhancer = 0.5 - np.power(np.power(enhancer-0.5,2),0.5).round(3)
    enhancer = (b - a) * (enhancer - enhancer.min()) / (enhancer.max()) + a
    
    return torch.from_numpy(img_np * enhancer)
