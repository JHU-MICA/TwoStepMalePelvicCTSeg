# this file contains paths to models #

import os

# base directory for all data (ex. data loc, fingerprints, cached data)
DATA_BASE = 'path/to/data/folder' 
MODEL_BASE = 'path/to/model/folder' 

COARSE_WEIGHTS = 'path/to/model/file' # ex) os.path.join(MODEL_BASE,'model_weights','model_type','model_coarse.pt')
COARSE_DIMS = (112,112,48)
COARSE_RES = (1.7,1.7,3)
COARSE_CONFIG = {
    'in_channels': 1,
    'out_channels': 7,
    'img_size': (112,112,48),
    'norm_name': "INSTANCE",
    'num_layers' : 4,
    'encoder_block' : 'axial',
    'encoder_units' : 3,
}

PROSTATE_WEIGHTS = 'path/to/model/file' # ex) os.path.join(MODEL_BASE,'model_weights','model_type','model_fine_prostate.pt')
PROSTATE_DIMS = (64,64,48)
PROSTATE_RES = (1.7,1.7,3)
PROSTATE_CONFIG = {
    'in_channels': 1,
    'out_channels': 2,
    'img_size': (64,64,48),
    'norm_name': "INSTANCE",
    'num_layers' : 5,
    'encoder_block' : 'conv',
    'encoder_units' : 2,
    'decoder_block' : 'conv',
    'decoder_units' : 2,
}

RECTUM_WEIGHTS = 'path/to/model/file' # ex) os.path.join(MODEL_BASE,'model_weights','model_type','model_fine_rectum.pt')
RECTUM_DIMS = (64,96,64)
RECTUM_RES = (1.7,1.7,3)
RECTUM_CONFIG = {
    'in_channels': 1,
    'out_channels': 2,
    'img_size': (64,96,64),
    'num_layers' : 5,
    'encoder_block' : 'dense',
    'encoder_units' : 4,
    'decoder_block' : 'conv',
    'decoder_units' : 2,
}

SEMINAL_WEIGHTS = 'path/to/model/file' # ex) os.path.join(MODEL_BASE,'model_weights','model_type','model_fine_seminal.pt')
SEMINAL_DIMS = (64,64,48)
SEMINAL_RES = (1.7,1.7,3)
SEMINAL_CONFIG = {
    'in_channels': 1,
    'out_channels': 2,
    'img_size': (64,64,48),
    'norm_name': "instance",
    'num_layers' : 5,
    'encoder_block' : 'conv',
    'encoder_units' : 2,
    'decoder_block' : 'conv',
    'decoder_units' : 2,
}

FEMUR_WEIGHTS = 'path/to/model/file' # ex) os.path.join(MODEL_BASE,'model_weights','model_type','model_fine_femur.pt')
FEMUR_DIMS = (96,96,64)
FEMUR_RES = (1.7,1.7,3)
FEMUR_CONFIG = {
    'in_channels': 1,
    'out_channels': 2,
    'img_size': (96,96,64),
    'norm_name': "instance",
    'num_layers' : 5,
    'encoder_block' : 'dense',
    'encoder_units' : 4,
    'decoder_block' : 'conv',
    'decoder_units' : 2,
}
