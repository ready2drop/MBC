import os
import numpy as np
from pytz import timezone
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import scipy.ndimage as ndimage

def logdir(str, mode):
    seoul_timezone = timezone('Asia/Seoul')
    today_seoul = datetime.now(seoul_timezone)
    
    directory_name = today_seoul.strftime("%Y-%m-%d-%H-%M")
    
    log_dir = str+directory_name + '-' + mode
    
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
        dict['num_features'] = 192
        pass
    
   
    return dict

        
def save_confusion_matrix_roc_curve(targets_all, predicted_all, log_dir):
        # Save confusion matrix
        cm = confusion_matrix(targets_all, predicted_all)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['real_stone_0', 'real_stone_1'])
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax)
        plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'))

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(targets_all, predicted_all)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(log_dir, 'roc_curve.png'))    
        