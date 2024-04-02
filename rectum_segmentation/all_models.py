# create fingerprint, train and test all the models

import pandas as pd
import os

from paths import DATA_BASE
from task_configs import TASK_NAME
from model_configs import MODEL_CONFIGS
from fingerprint import get_fingerprint
import mean_teacher
import train
from test import test_model

TASK_NAME = 'rectum'
PATCH_SIZE = (64,96,64)
PIXDIMS = (1.7,1.7,3)
MODEL_TYPE = 'rectum'

TRAIN_STATE = 0
VAL_SIZE = 0.1

for model_dict in MODEL_CONFIGS:
    train.train(model_dict['name'],model_dict['model_settings'], TASK_NAME, PATCH_SIZE, MODEL_TYPE, TRAIN_STATE, VAL_SIZE, PIXDIMS)
    test_model(model_dict['name'],model_dict['model_settings'],'plain', TASK_NAME, PATCH_SIZE, MODEL_TYPE, PIXDIMS)


TASK_NAME = 'rectum_enhanced'
MODEL_TYPE = 'rectum_enhanced'

for model_dict in MODEL_CONFIGS:
    train.train(model_dict['name'],model_dict['model_settings'], TASK_NAME, PATCH_SIZE, MODEL_TYPE, TRAIN_STATE, VAL_SIZE, PIXDIMS)
    test_model(model_dict['name'],model_dict['model_settings'],'enhanced', TASK_NAME, PATCH_SIZE, MODEL_TYPE, PIXDIMS)

MODEL_TYPE = 'rectum_enhanced_mt'

for model_dict in MODEL_CONFIGS:
    mean_teacher.train(model_dict['name'],model_dict['model_settings'], TASK_NAME, PATCH_SIZE, MODEL_TYPE, TRAIN_STATE, VAL_SIZE, PIXDIMS)
    test_model(model_dict['name']+'-student',model_dict['model_settings'],'enhanced_mt', TASK_NAME, PATCH_SIZE, MODEL_TYPE, PIXDIMS)
    test_model(model_dict['name']+'-teacher',model_dict['model_settings'],'enhanced_mt', TASK_NAME, PATCH_SIZE, MODEL_TYPE, PIXDIMS)
