import numpy as np
from tqdm import tqdm
import SimpleITK as sitk

from main import coarse_to_fine
from DicomRTTool import DicomReaderWriter

# Run the full segmentation pipeline using dicom images

def run_pipeline(dcm_path, out_dir, device='cuda:0', **kwargs):
    '''
        dcm_path: input dicom path
        out_dir: output directory
        device: device on which to run segmentation code (cuda and cpu supported)
        -- kwargs --
        roi_names: names of the rois
        color_dict: dict of colors to use for each roi
    '''

    roi_names = kwargs.pop('roi_names', ['prostate', 'bladder', 'rectum', 'seminal vesicles', 'left femoral head', 'right femoral head'])

    color_dict = kwargs.pop(
            'color_dict', 
            {
                'prostate': [51, 255, 51], # prostate green
                'bladder': [255, 255, 0], # bladder yellow
                'rectum': [102, 51, 0], # rectum brown
                'seminal vesicles': [0, 255, 255], # sv blue
                'left femoral head': [255, 0, 0], # lt femur red 
                'right femoral head': [255, 119, 0], # rt femur yellow
            }
        )

    # initialize dicom reader
    Dicom_reader = DicomReaderWriter(arg_max=True)

    # read images
    Dicom_reader.walk_through_folders(dcm_path)
    Dicom_reader.get_images()

    # save as temp nii file
    dicom_sitk_handle = Dicom_reader.dicom_handle # SimpleITK image handle
    sitk.WriteImage(dicom_sitk_handle, 'temp.nii')

    # run coarse_to_fine pipeline and reshape to x,y,z,#channels
    seg_np = np.squeeze(coarse_to_fine('temp.nii', device=device))
    seg_out = np.moveaxis(np.array([seg_np == i for i in range(len(roi_names)+1)]), 0, -1).astype(int)
    seg_out = np.transpose(seg_out, (2,1,0,3)) 
    seg_out = np.expand_dims(seg_out, 0)

    Dicom_reader.prediction_array_to_RT(prediction_array=seg_out, output_dir=out_dir, ROI_Names=roi_names, ROI_Colors=color_dict)

if __name__ == '__main__':
    dcm_path = 'D:\Rahul\Data\ProstateCT_DCM\\002'
    out_dir = 'test'

    run_pipeline(dcm_path, out_dir, device='cpu')