import os
from pytz import timezone
from datetime import datetime

def logdir(str):
    seoul_timezone = timezone('Asia/Seoul')
    today_seoul = datetime.now(seoul_timezone)
    
    directory_name = today_seoul.strftime("%Y-%m-%d-%H-%M")
    
    log_dir = str+directory_name
    
    if os.path.exists(log_dir):
        pass
    else:
        os.makedirs(log_dir)
        
    return log_dir

def get_model_parameters(dict):
    
    if dict['model_architecture'] == 'efficientnet_v2_l':
        dict['model_parameters'] = {'weights':'EfficientNet_V2_L_Weights.IMAGENET1K_V1'}
        dict['num_features'] = 1280
        
    elif dict['model_architecture'] == 'convnext_large':
        dict['model_parameters'] = {'weights':'ConvNeXt_Large_Weights.IMAGENET1K_V1'}
        dict['num_features'] = 1536
        
    elif dict['model_architecture'] == 'regnet_y_32gf':
        dict['model_parameters'] = {'weights':'RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1'}
        dict['num_features'] = 3712
        
    elif dict['model_architecture'] == 'resnext101_64x4d':
        dict['model_parameters'] = {'weights':'ResNeXt101_64X4D_Weights.IMAGENET1K_V1'}
        dict['num_features'] = 2048
        
    elif dict['model_architecture'] == 'vit_l_16':
        dict['model_parameters'] = {'weights':'ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1'}
        dict['num_features'] = 1024
        pass
    
    elif dict['model_architecture'] == 'SwinUNETR':
        dict['model_parameters'] = {'img_size' : (96, 96, 96), 'in_channels' : 1, 'out_channels' : 1,'feature_size' : 48}
        dict['num_features'] = 768
        pass
    
   
    return dict