# this file contains a variety of training functions for training different models

import torch
import monai
import monai.transforms as mt
import numpy as np
import os

from paths import MODEL_BASE

from fingerprint import read_fingerprint
from data_loader import create_datasets
from transforms import nnUNet_transform
from models import GenericUnet


def train(model_name, model_settings, TASK_NAME, PATCH_SIZE, MODEL_TYPE, TRAIN_STATE, VAL_SIZE, PIXDIMS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('training on',device)

    # set up transforms and datasets/dataloaders
    task_name = TASK_NAME
    patch_size = PATCH_SIZE
    inference_size = PATCH_SIZE # for patch-based inference (usually the same as spatial_size if 3d)
    data_fingerprint = read_fingerprint(task_name)
    train_transform,test_transform = nnUNet_transform(data_fingerprint,device=device,spatial_size=patch_size,pixdims=PIXDIMS)
    train_set, val_set, _ = create_datasets(task_name,(train_transform,test_transform),val_size=VAL_SIZE,cache_dir=TASK_NAME,random_state=TRAIN_STATE)
    batch_size = 2
    train_loader = monai.data.DataLoader(train_set,batch_size=batch_size)
    val_loader = monai.data.DataLoader(val_set,batch_size=1)

    # basic training loop
    model_name = model_name
    model_type = MODEL_TYPE
    model_path = os.path.join(MODEL_BASE,'model_weights',model_type)
    num_classes = 2
    print_step = 10

    model = GenericUnet(**model_settings)

    #model.load_state_dict(torch.load(os.path.join(model_path,model_name+'.pt')))
    start_epoch = 0

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # tensorboard set up
    writer_dir = os.path.join(MODEL_BASE,'logs',model_type,model_name)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=writer_dir)
    rois = ['prostate']

    # eval set up
    post_pred = mt.Compose([mt.EnsureType(), mt.AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = mt.Compose([mt.EnsureType(), mt.AsDiscrete(to_onehot=num_classes)])

    # hyperparameters
    num_epochs = 100
    dice_calculator = monai.metrics.DiceMetric(include_background=False,reduction='sum_batch')
    criterion = monai.losses.DiceLoss(include_background=False,to_onehot_y=True,softmax=True,reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    lr_scheduler = None
    best_loss = np.inf

    model = model.to(device)

    for epoch in range(num_epochs):

        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")

        model.train()
        epoch_loss = 0
        val_loss = 0

        # training steps
        for i,batch in enumerate(train_loader):
            optimizer.zero_grad()

            # push to device
            inputs, labels = (batch["image"].to(device),batch["label"].to(device))

            # run model and compute loss 
            outputs = model(inputs)
            
            loss = criterion(outputs, labels) / len(inputs)
            loss.backward()
            epoch_loss += loss.item() * len(inputs)

            # apply loss
            
            optimizer.step()

            if i % print_step == print_step-1:

                # compute dice score
                outputs = [post_pred(x) for x in monai.data.decollate_batch(outputs)]
                labels = [post_label(x) for x in monai.data.decollate_batch(labels)]
                dice_calculator(y_pred=outputs,y=labels)
                dice_vals = dice_calculator.aggregate().cpu().numpy()/len(inputs)
                dice_vals = dice_vals.round(decimals=3)
                dice_calculator.reset()
                print(f"batch {i+1}/{len(train_loader)}: loss {loss.item()}, dice {dice_vals}")
        

        # validation steps
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # push to device
                inputs, labels = (batch["image"].to(device),batch["label"].to(device))
                print(f"val {i+1}/{len(val_loader)}",end="\r")

                # run model and compute loss 
                outputs = monai.inferers.sliding_window_inference(inputs,inference_size,batch_size,model,overlap=0.5)
                #outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss # add loss to val loss

                # compute dice score
                outputs = [post_pred(x) for x in monai.data.decollate_batch(outputs)]
                labels = [post_label(x) for x in monai.data.decollate_batch(labels)]
                dice_calculator(y_pred=outputs,y=labels)
            
        # aggregate dice values
        dice_vals = dice_calculator.aggregate().cpu().numpy() / len(val_set)
        dice_vals = dice_vals.round(decimals=3)
        dice_calculator.reset()

        # compute aggregate losses
        val_loss /= len(val_set)
        epoch_loss /= len(train_set)
        
        # print out epoch stats
        print(f'Epoch Loss: {epoch_loss}, Val Dice: {dice_vals}')

        # save model if appropriate
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),os.path.join(model_path,model_name+'.pt'))
            print('weights saved')

        # write to summary
        loss_dict = {'train':epoch_loss,'val':val_loss}
        writer.add_scalars('Loss', loss_dict, start_epoch+epoch)

        dice_dict = dict()
        for j,r in enumerate(rois): dice_dict[r] = dice_vals[j]
        
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        writer.add_scalars('Dice', dice_dict, start_epoch+epoch)

    print('finished training {}'.format(model_name))