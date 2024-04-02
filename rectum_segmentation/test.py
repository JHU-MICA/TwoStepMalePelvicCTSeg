# test model performance

import torch
import monai
import pandas as pd
import monai.transforms as mt
from monai.metrics import compute_meandice
import numpy as np
import os
from paths import MODEL_BASE

from fingerprint import read_fingerprint
from data_loader import create_datasets
from transforms import nnUNet_transform
from models import GenericUnet

def test_model(model_name, model_settings, out_name, TASK_NAME, PATCH_SIZE, MODEL_TYPE, PIXDIMS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('evaluating on',device)

    # set up transforms and datasets/dataloaders
    task_name = TASK_NAME
    patch_size = PATCH_SIZE
    data_fingerprint = read_fingerprint(task_name)
    train_transform,test_transform = nnUNet_transform(data_fingerprint,device=device,spatial_size=patch_size,pixdims=PIXDIMS)
    _, _, test_set = create_datasets(task_name,(train_transform,test_transform),cache_dir=TASK_NAME)
    batch_size = 1
    test_loader = monai.data.DataLoader(test_set,batch_size=1)

    model_name = model_name
    model_type = MODEL_TYPE
    model_path = os.path.join(MODEL_BASE,'model_weights',model_type,model_name+'.pt')
    num_classes = 2

    # model -> can be found in runs.txt
    model = GenericUnet(**model_settings)

    # load in the weights and push to device for evaluation
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    metrics_df = pd.DataFrame(columns=['organ'])

    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            print(f'Image {i+1}/{len(test_loader)}',end='\r')

            # push to device
            inputs, labels = (batch["image"].to(device),batch["label"].to(device))

            img_name = os.path.basename(batch['image_meta_dict']['filename_or_obj'][0])

            prediction = torch.argmax(model(inputs),dim=1)
            prediction = mt.AsDiscrete(to_onehot=num_classes)(prediction)
            prediction = mt.KeepLargestConnectedComponent(1,True)(prediction)

            labels = mt.AsDiscrete(to_onehot=num_classes)(labels[0])

            prediction = torch.unsqueeze(prediction,0)
            labels = torch.unsqueeze(labels,0)

            dice = compute_meandice(prediction,labels,include_background=False)[0]

            metrics_df.loc[img_name] = dice.cpu().detach().numpy()
            
        
    metrics_df.loc['mean'] = metrics_df.mean()
    metrics_df.loc['std'] = metrics_df.iloc[:-1,:].std()

    with open(out_name, "a") as myfile:
        myfile.write(model_name+'\n')
        myfile.write('{}+-{}'.format(metrics_df.organ['mean'],metrics_df.organ['std'])+'\n')