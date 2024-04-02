# this file contains a training script for mean teacher training #
# adapted from: "A Noise-Robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions From CT Images" #

import torch
import monai
import monai.transforms as mt
import numpy as np
import copy
import os


from fingerprint import read_fingerprint
from data_loader import create_datasets
from transforms import nnUNet_transform
from models import GenericUnet
from paths import MODEL_BASE

# apply moving average from student to teacher
def update_ema(student,teacher,a):
    '''
    student: student model
    teacher: teacher model
    a: alpha
    step: current training step -> needed to calculate alpha
    '''

    for s_param, t_param in zip(student.parameters(),teacher.parameters()):
        t_param.data.mul_(a).add_(s_param.data,alpha=1-a)

def get_alpha(seg_loss,beta,step,a_max=1.0):
    '''
    modulate alpha based on threshold (moving average of loss vals) and step
    returns: alpha
    '''

    a_min = 0.999 if step > 10000 else 0.99

    # if segmentation loss is small enough -> incorporate it more
    if seg_loss < beta:
        a = a_min
    else:
        a = a_max

    # Use the true average until the exponential average is more correct
    a = min(1 - 1 / (step + 1), a)

    return a

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
    ema_model = GenericUnet(**model_settings)

    # no need for gradients for teacher model (ema_model)
    for param in ema_model.parameters():
            param.detach_()
    

    start_epoch = 0

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # tensorboard set up
    writer_dir = os.path.join(MODEL_BASE,'logs',model_type,model_name)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=writer_dir)

    # eval set up
    post_pred = mt.Compose([mt.EnsureType(), mt.AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = mt.Compose([mt.EnsureType(), mt.AsDiscrete(to_onehot=num_classes)])

    # hyperparameters
    num_epochs = 100
    dice_calculator = monai.metrics.DiceMetric(include_background=False,reduction='sum_batch')
    consistency_criterion = torch.nn.L1Loss(reduction='none')
    criterion = monai.losses.DiceLoss(include_background=False,to_onehot_y=True,softmax=True,reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    lr_scheduler = None
    best_loss = np.inf
    global_step = 1
    noise = 0.1 # amount to scale noise added to image
    loss_vals = [1000] * len(train_loader) # store loss values
    l1 = 0.1 # lambda for adaptive student

    # push models to gpu
    model = model.to(device)
    ema_model = ema_model.to(device) 

    for epoch in range(num_epochs):

        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")

        model.train()
        ema_model.train()

        epoch_loss = 0
        val_loss = 0

        beta = np.percentile(loss_vals,90) # calculate dynamic threshold for loss
        loss_vals = []

        # training steps
        for i,batch in enumerate(train_loader):
            optimizer.zero_grad()

            # push to device
            inputs, labels = (batch["image"].to(device),batch["label"].to(device))

            # run model and compute loss 
            outputs = model(inputs + torch.randn(inputs.shape,device=device) * noise)
            ema_outputs = ema_model(inputs + torch.randn(inputs.shape,device=device) * noise)

            seg_loss = criterion(outputs, labels)
            ema_loss = criterion(ema_outputs,labels)
            consistency_loss = consistency_criterion(outputs,ema_outputs).mean(dim=[1,2,3,4]).sum()

            l_val = l1 if seg_loss > ema_loss else l1 * 0.1
            a_val = get_alpha(seg_loss,beta,global_step)
            
            loss = seg_loss + l_val * consistency_loss # / len(inputs)
            loss.backward()
            epoch_loss += loss.item()
            loss_vals.append(seg_loss.item()) # * len(inputs))

            # update ema
            update_ema(model,ema_model,a_val)

            # apply loss
            
            optimizer.step()
            global_step += 1

            if i % print_step == print_step-1:

                # compute dice score
                outputs = [post_pred(x) for x in monai.data.decollate_batch(outputs)]
                ema_outputs = [post_pred(x) for x in monai.data.decollate_batch(ema_outputs)]
                labels = [post_label(x) for x in monai.data.decollate_batch(labels)]
                dice_calculator(y_pred=outputs,y=labels)
                dice_vals = dice_calculator.aggregate().cpu().numpy()/len(inputs)
                dice_vals = dice_vals.round(decimals=3)
                dice_calculator.reset()
                dice_calculator(y_pred=ema_outputs,y=labels)
                ema_dice_vals = dice_calculator.aggregate().cpu().numpy()/len(inputs)
                ema_dice_vals = ema_dice_vals.round(decimals=3)
                dice_calculator.reset()
                print(f"batch {i+1}/{len(train_loader)}: loss {loss.item()/ len(inputs)}, student dice {dice_vals}, teacher dice {ema_dice_vals}")
        

        # validation steps
        model.eval()
        ema_model.eval()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # push to device
                inputs, labels = (batch["image"].to(device),batch["label"].to(device))
                print(f"val {i+1}/{len(val_loader)}",end="\r")

                # run model and compute loss 
                outputs = monai.inferers.sliding_window_inference(inputs,inference_size,batch_size,model,overlap=0.5)
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

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # push to device
                inputs, labels = (batch["image"].to(device),batch["label"].to(device))
                print(f"val {i+1}/{len(val_loader)}",end="\r")

                # run model and compute loss 
                outputs = monai.inferers.sliding_window_inference(inputs,inference_size,batch_size,ema_model,overlap=0.5)

                # compute dice score
                outputs = [post_pred(x) for x in monai.data.decollate_batch(outputs)]
                labels = [post_label(x) for x in monai.data.decollate_batch(labels)]
                dice_calculator(y_pred=outputs,y=labels)
            
        # aggregate dice values
        ema_dice_vals = dice_calculator.aggregate().cpu().numpy() / len(val_set)
        ema_dice_vals = ema_dice_vals.round(decimals=3)
        dice_calculator.reset()

        # compute aggregate losses
        val_loss /= len(val_set)
        epoch_loss /= len(train_set)
        
        # print out epoch stats
        print(f'Epoch Loss: {epoch_loss}, Val Student Dice: {dice_vals}, Val Teacher Dice: {ema_dice_vals}')

        # save model if appropriate
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),os.path.join(model_path,model_name+'-student.pt'))
            torch.save(ema_model.state_dict(),os.path.join(model_path,model_name+'-teacher.pt'))
            print('weights saved')

        # write to summary
        loss_dict = {'train':epoch_loss,'val':val_loss}
        writer.add_scalars('Loss', loss_dict, start_epoch+epoch)

        dice_dict = {'student':dice_vals,'teacher':ema_dice_vals}
        
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        writer.add_scalars('Dice', dice_dict, start_epoch+epoch)

    print('finished training {}'.format(model_name))