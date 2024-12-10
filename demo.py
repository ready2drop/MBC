import argparse
import os
import io
import base64
import matplotlib.pyplot as plt
import sys
import bleach
import gradio as gr
import torch
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from src.dataset.bc_feature_engineering import load_data
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.storage")

def parse_args(args):
    parser = argparse.ArgumentParser(description="M3D-LaMed chat")
    parser.add_argument('--data_dir', type=str, default="/home/rkdtjdals97/datasets/DUMC_nifti_crop/")
    parser.add_argument('--excel_file', type=str, default="dumc_1024a.csv")
    parser.add_argument('--modality', type=str, default="tabular")
    parser.add_argument('--phase', type=str, default="combine")
    parser.add_argument('--smote', type=bool, default=True)
    parser.add_argument('--model_name_or_path', type=str, default="/home/rkdtjdals97/MBC/logs/2024-12-10-14-41-test-tabular/calibrated_stacking_model.pkl", choices=[])
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    return parser.parse_args(args)


def load_data_and_prepare(data_dir, excel_file, modality, phase, smote):
    # Load train, validation, and test data
    train_df,val_df = load_data(data_dir, excel_file, 'train', modality, phase, smote)
    # test_df = load_data(data_dir, excel_file, 'test', modality, phase, smote)
    
    # Prepare training, validation, and test features and targets
    x_train = train_df.drop(columns=['patient_id', 'target'])
    y_train = train_df['target']
    
    # x_val = val_df.drop(columns=['patient_id', 'target'])
    # y_val = val_df['target']
    
    # x_test = test_df.drop(columns=['patient_id', 'target'])
    # y_test = test_df['target']
    
    # return x_train, y_train, x_val, y_val, x_test, y_test
    return x_train, y_train

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Inference function
def classify(tabular_data):
    """
    Perform classification on tabular data.
    Args:
        tabular_data (list): List of input data points (e.g., rows in a dataframe)
    Returns:
        str: Classification result and probabilities
    """
    # scaler = StandardScaler()
    # scaler.fit(x_train)  # Fit the scaler to training data

    input_data = np.array(tabular_data, dtype=float)
    print(input_data)
    
    # Normalize input data using the scaler
    # normalized_data = scaler.transform(input_data)
    
    # Get predicted class and probabilities
    prob = model.predict_proba(input_data)
    predicted_class = model.predict(input_data)

    # Get the probability for the predicted class
    class_index = int(predicted_class[0])  # The predicted class index (0 or 1, etc.)
    class_probability = prob[0][class_index]  # Probability corresponding to the predicted class

    # Format output
    result = f"Predicted Class: {predicted_class[0]}, Probability: {class_probability:.2f}"
    
    return result

args = parse_args(sys.argv[1:])
# x_train, y_train, x_val, y_val, x_test, y_test = load_data_and_prepare(args.data_dir, args.excel_file, args.modality, args.phase, args.smote)
x_train, y_train = load_data_and_prepare(args.data_dir, args.excel_file, args.modality, args.phase, args.smote)
model = load_model(args.model_name_or_path)

# Create model
device = torch.device(args.device)


# Gradio
examples = [
    [
        [['1', '1', '10.6', '171', '14.54', '236', '548', '182', '12.33', '3.2', '72']],
        "PT_NO = 10001862, VISIBLE_STONE_CT = True, REAL_STONE = True",
    ],
    [
        [['1','1','13.6','388','21.13','196','70','118','1.87','2.7', '58']],
        "PT_NO = 10007376, VISIBLE_STONE_CT = True, REAL_STONE = True",
    ],
    [
        [['1','1','12','170','6.7','71','11','18','3.95','0.6', '91']],
        "PT_NO = 10040285, VISIBLE_STONE_CT = False, REAL_STONE = True",
    ],
    [
        [['1','1','12.1','192','8.63','47','32','59','0.02','0.4', '57']],
        "PT_NO = 10005545, VISIBLE_STONE_CT = False, REAL_STONE = False",
    ],
]
tabular_header = ['DUCT_DILIATATION_8MM','DUCT_DILIATATION_10MM','Hb','PLT','WBC','ALP','ALT','AST','CRP','BILIRUBIN', 'AGE']

description = """
GPU 리소스 제약으로 인해, 온라인 데모에서는 NVIDIA RTX 3090 24GB를 사용하고 있습니다. \n

**Note**: 현재 저희 모델은 **총담관결석증**의 분석 및 진단을 중심으로 최적화되어 있으며, 정확하고 신뢰할 수 있는 결과를 제공합니다. \n 
"""

title_markdown = ("""
# 임상 데이터 기반 딥러닝을 이용한 총담관석 예측 모델  
## Development of a Common Bile Duct Stone Prediction Model Using Deep Learning Based on Clinical Data
[📖[Learn more about Common Bile Duct Stones (총담관결석증)](https://namu.wiki/w/%EC%B4%9D%EB%8B%B4%EA%B4%80%EA%B2%B0%EC%84%9D%EC%A6%9D)] 
""")


def explain_with_lime(tabular_data):
    """
    Apply LIME to explain predictions.
    Args:
        tabular_data (list): List of input data points (e.g., rows in a dataframe)
    Returns:
        str: HTML or image showing LIME explanation
    """
    input_data = np.array(tabular_data, dtype=float)
    explainer = LimeTabularExplainer(
        training_data=x_train.values,  # Replace with your training data
        feature_names=tabular_header,
        class_names=['intermediate', 'High'],  # Replace with actual class names
        mode='classification'
    )

    explanation = explainer.explain_instance(
        input_data[0],  # Single instance to explain
        model.predict_proba,  # Probability prediction function
        num_features=len(tabular_header)
    )
    
    # Plot LIME explanation
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(25, 8) 
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return f"<img src='data:image/png;base64,{encoded_image}'/>"


# tabular_header = ['SEX','FIRST_SBP','FIRST_DBP','FIRST_HR','FIRST_RR','FIRST_BT','AGE','VISIBLE_STONE_CT','PANCREATITIS','DUCT_DILIATATION_10MM','DUCT_DILIATATION_8MM','Hb','PLT','WBC','ALP','ALT','AST','CRP','BILIRUBIN']
tabular_header = ['DUCT_DILIATATION_10MM','DUCT_DILIATATION_8MM','Hb','PLT','WBC','ALP','ALT','AST','CRP','BILIRUBIN', 'AGE']
tabular_dtype = ['number'] * len(tabular_header)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(title_markdown)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            tabular_input = gr.Dataframe(headers= tabular_header, datatype= tabular_dtype, label="Tabular Input", type="array", interactive=True, row_count=1, col_count=11)
            info = gr.Textbox(lines=1, label="Patient info", visible = False)

            with gr.Accordion("Parameters", open=False) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                        label="Temperature", )
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.1, interactive=True, label="Top P", )
                
            with gr.Row():
                # btn_c = gr.ClearButton([tabular_input])
                btn_c = gr.Button("Clear")
                btn = gr.Button("Run")
                
 

            
    result_output = gr.Textbox(lines=2, label="Classification Result")
    lime_output = gr.HTML(label="LIME Explanation")
    gr.Examples(examples=examples, inputs=[tabular_input, info])
    btn.click(fn=classify, inputs=tabular_input, outputs=result_output)
    btn.click(fn=explain_with_lime, inputs=tabular_input, outputs=lime_output)  # Add LIME button    
   
    # Clear functionality: resets inputs and outputs
    def clear_fields():
        return None, None, [[None] * len(tabular_header)]

    btn_c.click(fn=clear_fields, inputs=[], outputs=[result_output, lime_output, tabular_input])


demo.queue()
demo.launch(share=True)


