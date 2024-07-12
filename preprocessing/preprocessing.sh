#!/bin/bash
# ## get tabular
# python preprocessing/get_tabular.py  

# ## get json : necessary file for prediction
# python preprocessing/get_json.py --data_dir /home/irteam/rkdtjdals97-dcloud-dir/datasets/Part4_nifti/ 

# ## prediction : Segmentation
# python preprocessing/test.py --data_dir /home/irteam/rkdtjdals97-dcloud-dir/datasets/Part4_nifti/ --exp_name test1

## crop slice 
python preprocessing/crop_slice.py --image_dir /home/irteam/rkdtjdals97-dcloud-dir/datasets/Part4_nifti/ --label_dir /home/irteam/rkdtjdals97-dcloud-dir/MBC/outputs/test1/ --output_dir /home/irteam/rkdtjdals97-dcloud-dir/datasets/Part4_nifti_crop_ver2/