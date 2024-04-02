# this file will have functions to load in data using monai api

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import monai

from fingerprint import read_fingerprint
from paths import DATA_BASE

def create_datasets(task_name,transforms,random_state=0,cache_dir=None,val_size=0.2,*args,**kwargs):
    '''
    this function will be used to create data loaders for a given task
    task_name: name of task (data to use)
    transforms: tuple (train_transforms,test_transforms)
    batch_size: tuple (train_batchsize,test_batchsize)
    random_state: state f or train/val split (change for CV)
    val_size: proportion for validation split
    cache_dir: directory to save cached images (if None -> use task name)
    args, kwargs: any other parameters that might need to be passed in
    '''

    test_names = ['001','002','003','004','005','006','009','010','011','012','013','014','015']

    # get the cache directory for the data (persistent storage dataloader is default)
    if cache_dir == None:
        cache_dir = os.path.join(DATA_BASE,'cached_data',task_name)
    else:
        cache_dir = os.path.join(DATA_BASE,'cached_data',cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # read in the data and split
    data_loc = pd.read_csv(os.path.join(DATA_BASE,'data_locs',task_name+'_data_loc.csv'))
    train_loc = data_loc.loc[15:] 
    train_loc, val_loc = train_test_split(train_loc,test_size=val_size,random_state=random_state)
    test_loc = data_loc.loc[:14]

    train_dicts = [
        {'image': img_name,'label': lbl_name}
        for img_name, lbl_name in zip(train_loc.image,train_loc.label)
    ]

    val_dicts = [
        {'image': img_name,'label': lbl_name}
        for img_name, lbl_name in zip(val_loc.image,val_loc.label)
    ]

    test_dicts = [
        {'image': img_name,'label': lbl_name}
        for img_name, lbl_name in zip(test_loc.image,test_loc.label)
    ]

    train_dataset = monai.data.PersistentDataset(data=train_dicts,transform=transforms[0],cache_dir=cache_dir)
    val_dataset = monai.data.PersistentDataset(data=val_dicts,transform=transforms[1],cache_dir=cache_dir)
    test_dataset = monai.data.PersistentDataset(data=test_dicts,transform=transforms[1],cache_dir=cache_dir)

    return train_dataset, val_dataset, test_dataset