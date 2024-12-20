import os
import numpy as np
from pytz import timezone
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
import seaborn as sns
import pandas as pd


def logdir(str, mode, modality, model):
    seoul_timezone = timezone('Asia/Seoul')
    today_seoul = datetime.now(seoul_timezone)
    
    directory_name = today_seoul.strftime("%Y-%m-%d-%H-%M")
    
    log_dir = str+directory_name + '-' + mode + '-' + modality + '-' + model
    
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
    
    elif dict['model_architecture'] == 'ViT':
        # dict['model_parameters'] = {'img_size' : (96, 96, 96), 'in_channels' : 1, 'patch_size' : (16, 16, 16), 'hidden_size' : 768, 'num_layers' : 12, 'num_heads' : 12, 'mlp_dim' : 3072, 'dropout_rate' : 0.1}
        dict['model_parameters'] = {'img_size' : (256, 256, 32), 'in_channels' : 1, 'patch_size' : (16, 16, 4), 'hidden_size' : 768, 'num_layers' : 12, 'num_heads' : 12, 'mlp_dim' : 3072, 'dropout_rate' : 0.1}
        dict['num_features'] = 768
        pass
    
    elif dict['model_architecture'] == 'ResNet':
        dict['model_parameters'] = {'block' : 'bottleneck', 'layers' : (3, 4, 6, 3), 'block_inplanes' : [64, 128, 256, 512], 'spatial_dims' : 3, 'n_input_channels' : 1, 'no_max_pool' : False, 'shortcut_type' : 'B', 'widen_factor' : 1.0}
        dict['num_features'] = 400
        pass
   
    return dict

        
def save_confusion_matrix_roc_curve(targets_all, predicted_all, log_dir, model_name):
        # Save confusion matrix
        cm = confusion_matrix(targets_all, predicted_all)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['real_stone_0', 'real_stone_1'])
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax)
        plt.savefig(os.path.join(log_dir, f'{model_name}_confusion_matrix.png'))

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
        plt.title(f'Receiver Operating Characteristic of {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(log_dir, f'{model_name}_roc_curve.png'))    

def decision_curve_analysis(y_true, y_prob, thresholds=None):
    """
    Perform Decision Curve Analysis (DCA) for a single model, showing only net benefit >= 0.

    Parameters:
    - y_true (array-like): True binary labels.
    - y_prob (array-like): Predicted probabilities for the positive class.
    - thresholds (array-like): Array of threshold probabilities. Defaults to 0.01 to 0.99 in steps of 0.01.

    Returns:
    - results (dict): Dictionary containing thresholds and net benefits for model, treat-all, and treat-none strategies.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.7, 0.1)

    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = 0  # Net benefit of treat-none is always 0

    for threshold in thresholds:
        treat_model = (y_prob >= threshold).astype(int)
        tp_model = np.sum((y_true == 1) & (treat_model == 1))
        fp_model = np.sum((y_true == 0) & (treat_model == 1))
        prob_tp = tp_model / len(y_true)
        prob_fp = fp_model / len(y_true)
        net_benefit = prob_tp - (prob_fp * threshold / (1 - threshold))

        # Only include net benefit >= 0
        if net_benefit >= 0:
            net_benefit_model.append(net_benefit)
            net_benefit_all.append((np.sum(y_true == 1) / len(y_true)) - (np.sum(y_true == 0) / len(y_true)) * (threshold / (1 - threshold)))
        else:
            net_benefit_model.append(None)
            net_benefit_all.append(None)

    results = {
        'thresholds': thresholds,
        'net_benefit_model': net_benefit_model,
        'net_benefit_all': net_benefit_all,
        'net_benefit_none': [net_benefit_none] * len(thresholds)
    }
    return results
    
def plot_roc_and_calibration_test(models, X_test, y_test, log_dir, model_names):
    """
    Plot ROC and Calibration curves for multiple models on the test set.
    
    Parameters:
    models (list): List of trained models to evaluate.
    X_test (array-like): Test feature data.
    y_test (array-like): Test target data.
    model_names (list): List of model names corresponding to the models list.
    """
    
    plt.figure(figsize=(18, 6))

    # Subplot for ROC Curve
    plt.subplot(1, 3, 1)
    for model, name in zip(models, model_names):
        try:
            y_test_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
            
            # Highlight Stacking and Calibrated Stacking Models
            if name == 'Stacking Model':
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})', color='yellow', linewidth=2)
            elif name == 'Calibrated Stacking Model':
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})', color='orange', linewidth=2)
            elif "Calibrated" in name:
                plt.plot(fpr, tpr, color='darkgray', linestyle='--', alpha=0.7)  # Dark gray for calibrated models
            else:
                plt.plot(fpr, tpr, color='lightgray', linestyle='--', alpha=0.5)  # Light gray for uncalibrated models
        except Exception as e:
            print(f"Failed to calculate ROC for {name}: {e}")
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Chance')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # Subplot for Calibration Curve
    plt.subplot(1, 3, 2)
    for model, name in zip(models, model_names):
        try:
            y_test_pred_proba = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_test_pred_proba, n_bins=10)
            
            # Highlight Stacking and Calibrated Stacking Models
            if name == 'Stacking Model':
                plt.plot(prob_pred, prob_true, label=name, color='yellow', linewidth=2)
            elif name == 'Calibrated Stacking Model':
                plt.plot(prob_pred, prob_true, label=name, color='orange', linewidth=2)
            elif "Calibrated" in name:
                plt.plot(prob_pred, prob_true, color='darkgray', linestyle='--', alpha=0.7)  # Dark gray for calibrated models
            else:
                plt.plot(prob_pred, prob_true, color='lightgray', linestyle='--', alpha=0.5)  # Light gray for uncalibrated models
        except Exception as e:
            print(f"Failed to calculate Calibration Curve for {name}: {e}")

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    plt.title('Calibration Curve')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend(loc="lower right")

    # Subplot for Decision Curve Analysis (DCA)
    plt.subplot(1, 3, 3)
    for model, name in zip(models, model_names):
        try:
            y_test_pred_proba = model.predict_proba(X_test)[:, 1]
            dca_results = decision_curve_analysis(y_test, y_test_pred_proba)

            # Extract valid net benefits and thresholds
            filtered_thresholds = [t for t, nb in zip(dca_results['thresholds'], dca_results['net_benefit_model']) if nb is not None]
            filtered_net_benefit_model = [nb for nb in dca_results['net_benefit_model'] if nb is not None]
            filtered_net_benefit_all = [nb for nb in dca_results['net_benefit_all'] if nb is not None]

            # Highlight Stacking and Calibrated Stacking Models
            if name == 'Stacking Model':
                plt.plot(filtered_thresholds, filtered_net_benefit_model, label=name, color='yellow', linewidth=2)
            elif name == 'Calibrated Stacking Model':
                plt.plot(filtered_thresholds, filtered_net_benefit_model, label=name, color='orange', linewidth=2)
            elif "Calibrated" in name:
                plt.plot(filtered_thresholds, filtered_net_benefit_model, color='darkgray', linestyle='--', alpha=0.7)  # Dark gray for calibrated models
            else:
                plt.plot(filtered_thresholds, filtered_net_benefit_model, color='lightgray', linestyle='--', alpha=0.5)  # Light gray for uncalibrated models
        except Exception as e:
            print(f"Failed to calculate DCA for {name}: {e}")

    # Plot reference lines
    plt.axhline(0, color='red', linestyle='--', label='Treat None')
    plt.plot(filtered_thresholds, filtered_net_benefit_all, linestyle='--', color='green', label='Treat All')
    plt.title('Decision Curve Analysis (Net Benefit ≥ 0)')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.legend(loc="lower right")

    # Final adjustments and save
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'model_evaluation_with_stacking_calibration_highlight.png'))
    plt.show()
            
def plot_tsne(features, labels, epoch, log_dir):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=labels, palette='viridis')
    plt.title(f't-SNE of Combined Features at Epoch {epoch}')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.legend()
    plt.savefig(os.path.join(log_dir,f'tsne_epoch_{epoch}.png'))
    plt.show()        
    
    
# Define a function to save feature importance
def save_feature_importance(importance, feature_names, model_name, log_dir):
    """
    Save the feature importance as a CSV file and optionally plot it.
    
    :param importance: Feature importance values (list or array).
    :param feature_names: Corresponding feature names.
    :param model_name: Name of the model for which we are saving feature importance.
    :param log_dir: Directory to save the CSV and plots.
    """
    # Create a DataFrame to hold the feature names and their corresponding importance
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    
    # Sort by importance
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    
    # Save as CSV
    csv_path = os.path.join(log_dir, f'{model_name}_feature_importance.csv')
    feature_df.to_csv(csv_path, index=False)
    
    print(f"Feature importance saved for {model_name} at {csv_path}")
    
    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df)
    plt.title(f'{model_name} Feature Importance')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(log_dir, f'{model_name}_feature_importance.png')
    plt.savefig(plot_path)
    print(f"Feature importance plot saved for {model_name} at {plot_path}")
    plt.close()    