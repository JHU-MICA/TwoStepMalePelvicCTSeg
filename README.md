# Pelvic CT Segmentation Repository

This repository acts as the codebase for the paper "Cascaded cross-attention transformers and convolutional neural networks for multi-organ segmentation in male pelvic computed tomography".


## Getting Started
In order to train on your own data, you will have to create a separate "Data" and "Model" folder. The "Data" folder will be used to store "data_loc" and "data_fingerprint" files, as well as any other data that gets generated (ex. test data segmentations, cached data). 

- data_loc files: These will be stored in a "data_locs" subfolder of the "Data" folder. These files will have to be manually created. I do have a jupyter notebook that will do this automatically and would be happy to share. The data_loc file must be a csv file with at least two columns, "image" and "label", that correspond to full paths to images and corresponding labels. It is important to name the file as "{task name}_data_loc.csv". This "task name" is important, as it is used to identify data_locs and fingerprint files.

- fingerprint files: These will be stored in a "data_fingerprints" subfolder of the "Data" folder. Files are created automatically. They are used to process data via nnUNet preprocessing strategies, where the population values are used to clip and normalize individual data. It is important to make sure that files are saved as "{task name}_fingerprint.txt".

Make sure to install all the required packages via `pip install -r /path/to/requirements.txt`.

## File Directory
All files for each segmentation task are stored in one folder. For each task, there will be the following main files (some additional files may also exist):

1. data_loader.py: Contains function needed to load in the training/testing/validation data. The testing data used is set to be the first ~ 15 images, so it can be changed as needed.

2. fingerprint.py: Contains functions to create and read fingerprint files.

3. models.py: Contains functions to create segmentation models (UNet and DeepLabv3)

4. paths.py: Stores paths to the "Model" and "Data" folders. These paths must be changed to point to the appropriate folders.

5. test.py: Script for evaluating models on test data

6. transforms.py: Contains transformation (preprocessing) chains for model training/testing

7. backbones: Folder containing code for creating the model "backbones"

8. transformer: Folder containing code for creating transformer modules

For Fine Segmentation Tasks

9. task_configs.py: Describes the image patch size and folder for segmentation task

10. all_models.py: Trains all models whose configurations are described in `model_configs.py`. 

For Coarse Segmentation

11. trainers.py: Script for training a coarse segmentation model



coarse_to_fine folder: Contains code for running the full two-step segmentation pipeline. Models can be changed using the config.py file. To run the model on nifti images, run the `coarse_to_fine` function in `main.py` file. To run the model on dicom images, run the `run_pipeline` function in `pipeline.py`.

## Trained Models and Sample Data
Trained models (coarse multi-organ segmentation & each organ-specific fine segmentation models) can be found in Releases-Models

A sample CT to test the pipeline can be found in Releases-Sample Data

## Reference

Please cite the following paper.

Rahul Pemmaraju, Gayoung Kim, Lina Mekki, Daniel Y. Song, and Junghoon Lee, "Cascaded cross-attention transformers and convolutional neural networks for multi-organ segmentation in male pelvic computed tomography," Journal of Medical Imaging, 2024
