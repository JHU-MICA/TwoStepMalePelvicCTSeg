# this file will be used to create a data fingerprint file (nnUNet style) from a given dataset #

import nibabel as nib
import numpy as np
import pandas as pd
import os

from paths import DATA_BASE

def read_fingerprint(task_name):
    '''
    returns dictionary of values from fingerprint file
    task_name: name of task for fingerprint file
    returns: dict of values
    '''

    output_dict = dict()

    filename = os.path.join(DATA_BASE,'data_fingerprints',task_name+'_fingerprint.txt')

    with open(filename) as file:
        for line in file:
            split_line = line.rstrip().split(':')
            
            output_dict[split_line[0]] = float(split_line[1])


    return output_dict

def get_fingerprint(data_loc,task_name):
    '''
    computes foreground mean, std, 0.5 and 0.95 percentiles and saves in data 
    data_loc: pandas dataframe of image and label
    task_name: name of segmentation task
    returns: None, saves: fingerprint txt file
    '''

    # get data paths
    img_paths = list(data_loc['image'])
    lbl_paths = list(data_loc['label'])

    foreground_vals = [] # list of numpy arrays (initially)

    for i,(img,lbl) in enumerate(zip(img_paths,lbl_paths)):
        print('Image: {}, {}/{}'.format(os.path.basename(img),i+1,len(img_paths)),end='\r')

        # load in data
        img_data = nib.load(img).get_fdata()
        lbl_data = nib.load(lbl).get_fdata()

        # get foreground points and corresponding values
        foreground_coords = np.argwhere(np.greater(lbl_data,0))
        foreground_coords = np.ravel_multi_index(foreground_coords.T, img_data.shape)
        foreground = img_data.take(foreground_coords)

        foreground_vals.append(foreground)

    print('\n')

    # get statistics
    foreground_vals = np.concatenate(foreground_vals)
    foreground_mean = foreground_vals.mean()
    foreground_std = foreground_vals.std()
    foreground_005 = np.percentile(foreground_vals,0.5)
    foreground_995 = np.percentile(foreground_vals,99.5)

    # write to output file
    output_file = os.path.join(DATA_BASE,'data_fingerprints',task_name+'_fingerprint.txt')
    
    f = open(output_file,'w')
    f.write('mean: {}\n'.format(foreground_mean))
    f.write('std: {}\n'.format(foreground_std))
    f.write('0.5 percentile: {}\n'.format(foreground_005))
    f.write('99.5 percentile: {}\n'.format(foreground_995))
    f.close()

    print('Finished')

    return

if __name__ == '__main__':

    data_loc = pd.read_csv(os.path.join(DATA_BASE,'data_locs','five_label_data_loc.csv'))

    test_loc = data_loc.loc[:13]
    train_loc = data_loc.loc[14:]

    get_fingerprint(train_loc,'five_label')