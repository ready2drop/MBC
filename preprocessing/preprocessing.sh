#!/bin/bash
## get tabular
python preprocessing/get_tabular.py 

## get json : necessary file for prediction
#python preprocessing/get_json.py --data_dir /home/irteam/rkdtjdals97-dcloud-dir/datasets/Part3_nifti/ 

## prediction : Segmentation
# python preprocessing/test.py --data_dir /home/irteam/rkdtjdals97-dcloud-dir/datasets/Part3_nifti/ --exp_name test2 

## crop slice 
# python preprocessing/crop_slice.py --image_dir /home/irteam/rkdtjdals97-dcloud-dir/datasets/Part3_nifti/ --label_dir /home/irteam/rkdtjdals97-dcloud-dir/MBC/outputs/test2/ --output_dir /home/irteam/rkdtjdals97-dcloud-dir/datasets/Part3_nifti_crop/