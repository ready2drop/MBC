#!/bin/bash
## get json
python auto_labeling/get_json.py --data_dir /home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti/ 

## pred
python auto_labeling/test.py --data_dir /home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti/ --exp_name test2 --pretrained_model_name best_metric_model.pth

## crop slice
python auto_labeling/crop_slice.py --image_dir datasets/Part2_nifti/ --label_dir /outputs/test2/ --output_dir datasets/Part2_nifti_crop/