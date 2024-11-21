import argparse
import os
import re
import sys
import bleach
import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import nibabel as nib
import numpy as np
from PIL import Image
from monai.transforms import Resize
# from LaMed.src.model.language_model import *


def parse_args(args):
    parser = argparse.ArgumentParser(description="M3D-LaMed chat")
    parser.add_argument('--model_name_or_path', type=str, default="./LaMed/output/LaMed-Phi3-4B-finetune-0000/hf/", choices=[])
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--seg_enable', type=bool, default=True)
    parser.add_argument('--proj_out_num', type=int, default=256)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def image_process(file_path):
    if file_path.endswith('.nii.gz'):
        nifti_img = nib.load(file_path)
        img_array = nifti_img.get_fdata()
    elif file_path.endswith(('.png', '.jpg', '.bmp')):
        img = Image.open(file_path).convert('L')
        img_array = np.array(img)
        img_array = img_array[np.newaxis, :, :]
    elif file_path.endswith('.npy'):
        img_array = np.load(file_path)
    else:
        raise ValueError("Unsupported file type")

    resize = Resize(spatial_size=(32, 256, 256), mode="bilinear")
    img_meta = resize(img_array)
    img_array, img_affine = img_meta.array, img_meta.affine

    return img_array, img_affine

args = parse_args(sys.argv[1:])
os.makedirs(args.vis_save_path, exist_ok=True)

# Create model
# device = torch.device(args.device)

# dtype = torch.float32
# if args.precision == "bf16":
#     dtype = torch.bfloat16
# elif args.precision == "fp16":
#     dtype = torch.half

# kwargs = {"torch_dtype": dtype}
# if args.load_in_4bit:
#     kwargs.update(
#         {
#             "torch_dtype": torch.half,
#             "load_in_4bit": True,
#             "quantization_config": BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_quant_type="nf4",
#                 llm_int8_skip_modules=["visual_model"],
#             ),
#         }
#     )
# elif args.load_in_8bit:
#     kwargs.update(
#         {
#             "torch_dtype": torch.half,
#             "quantization_config": BitsAndBytesConfig(
#                 llm_int8_skip_modules=["visual_model"],
#                 load_in_8bit=True,
#             ),
#         }
#     )


# tokenizer = AutoTokenizer.from_pretrained(
#     args.model_name_or_path,
#     model_max_length=args.max_length,
#     padding_side="right",
#     use_fast=False,
#     trust_remote_code=True
# )
# model = AutoModelForCausalLM.from_pretrained(
#     args.model_name_or_path,
#     device_map='auto',
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     **kwargs
# )
# model = model.to(device=device)

# model.eval()

# Gradio
# Gradio
examples = [
    [
        "/home/rkdtjdals97/MBC/data/DUMC/10001862_Pre_enhance_5mm.npy",
        [['1', '104', '59', '100', '24', '38.2', '72', '1', '0', '1', '1', '10.6', '171', '14.54', '236', '548', '182', '12.33', '3.2']],
        "Given the 3D CT image, <region of interest>, and the associated <condition> information, determine if a common bile duct stone (CBD stone) is present.",
        "VISIBLE_STONE_CT = True, REAL_STONE = True",
    ],
    [
        "/home/rkdtjdals97/MBC/data/DUMC/10007376_Pre_enhance_3mm.npy",
        [['1','106','74','81','18','36.1','58','1','0','1','1','13.6','388','21.13','196','70','118','1.87','2.7']],
        "Given the 3D CT image, <region of interest>, and the associated <condition> information, determine if a common bile duct stone (CBD stone) is present.",
        "VISIBLE_STONE_CT = True, REAL_STONE = True",
    ],
    [
        "/home/rkdtjdals97/MBC/data/DUMC/10040285_Pre_enhance_3mm.npy",
        [['1','121','72','76','18','37','91','0','0','1','1','12','170','6.7','71','11','18','3.95','0.6']],
        "Given the 3D CT image, <region of interest>, and the associated <condition> information, determine if a common bile duct stone (CBD stone) is present.",
        "VISIBLE_STONE_CT = False, REAL_STONE = True",
    ],
    [
        "/home/rkdtjdals97/MBC/data/DUMC/10005545_Pre_enhance_3mm.npy",
        [['0','130','77','83','20','36.7','57','0','1','1','1','12.1','192','8.63','47','32','59','0.02','0.4']],
        "Given the 3D CT image, <region of interest>, and the associated <condition> information, determine if a common bile duct stone (CBD stone) is present.",
        "VISIBLE_STONE_CT = False, REAL_STONE = False",
    ],
]

description = """
GPU 리소스 제약으로 인해, 온라인 데모에서는 NVIDIA RTX 3090 24GB를 사용하고 있습니다. \n

**Note**: Half-precision (FP16)으로 동작하므로, 추론 성능에 일부 저하가 발생할 수 있습니다. \n
**Note**: 질문의 모호함을 피하기 위해 입력 텍스트는 간결한 영어로 작성해주시고, 마침표와 문법이 올바르게 사용되었는지 확인해주세요.  \n

현재 저희 모델은 **총담관결석증**에 특화되어 있으며, 향후 더 다양한 국소부위 질환에 대한 데이터를 반영해 지속적으로 성능을 개선할 계획입니다. \n 
"""

title_markdown = ("""
# 미세 국소부위 질환 임상 추론을 위한 3D Image-Tabular Fusion 모델  
### 🐀 Team MAGAWA
[📖[Origin of Name](https://ko.wikipedia.org/wiki/%EB%A7%88%EA%B0%80%EC%99%80)] 
""")


def extract_box_from_text(text):
    match = re.search(r'\[([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+)\]', text)
    if match:
        box_coordinates = [float(coord) for coord in match.groups()]
        return box_coordinates
    else:
        return None

# to be implemented
def inference(input_image, input_tabular, input_str, temperature, top_p, x1, y1, x2, y2):
    global vis_box
    global seg_mask
    image_xy = 256
    image_z = 32
    
    vis_box = [0, 0, 0, 0, 0, 0]
    seg_mask = np.zeros((image_z, image_xy, image_xy), dtype=np.uint8)

    ## filter out special chars
    input_str = bleach.clean(input_str)
    print("input_str: ", input_str, "input_image: ", input_image, "input_tabular: ", input_tabular)
    
    conditions = input_tabular  # Convert your tabular data to a suitable string format
    region = f"{(x1/image_xy):.4f},{(y1/image_xy):.4f},{(0/image_z):.4f},{(x2/image_xy):.4f},{(y2/image_xy):.4f},{(image_z/image_z):.4f}"  # Format bounding box coordinates (x1,y1,z1,x2,y2,z2)
    
    # Model Inference
    # prompt = "<im_patch>" * args.proj_out_num + input_str 
    prompt = (
    "<im_patch>" * args.proj_out_num + 
    f"{input_str} "
    f"<condition>: {', '.join(map(str, input_tabular[0]))} "
    f"<region of interest>: {region}"
    f"\n\n"
    )
    print(prompt)
    
    input_id = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device=device)
    image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

    generation, seg_logit = model.generate(image_pt, input_id, seg_enable=args.seg_enable, max_new_tokens=args.max_new_tokens,
                                        do_sample=args.do_sample, top_p=top_p, temperature=temperature)

    output_str = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    print("output_str", output_str)
    box = extract_box_from_text(output_str)
    if box is not None:
        vis_box = [box[0]*32, box[1]*256, box[2]*256, box[3]*32, box[4]*256, box[5]*256]
        vis_box = [int(b) for b in vis_box]
        return output_str, (image_rgb[0], [((0,0,0,0), 'target')])

    seg_mask = (torch.sigmoid(seg_logit) > 0.5).squeeze().detach().cpu().numpy()
    if seg_mask.sum() == 0:
        return output_str, None
    else:
        return output_str, (image_rgb[0], [(seg_mask[0], 'target')])


def select_slice(selected_slice):
    min_s = min(vis_box[0], vis_box[3])
    max_s = max(vis_box[0], vis_box[3])

    if min_s <= selected_slice <= max_s:
        return (image_rgb[selected_slice], [(seg_mask[selected_slice], 'target_mask'), ((vis_box[2],vis_box[1], vis_box[5],vis_box[4]), 'target_box')])
    else:
        return (image_rgb[selected_slice], [(seg_mask[selected_slice], 'target_mask'), ((0,0,0,0), 'target_box')])


def load_image(load_image):
    global image_np
    global image_rgb
    global vis_box
    global seg_mask
    vis_box = [0, 0, 0, 0, 0, 0]
    seg_mask = np.zeros((32, 256, 256), dtype=np.uint8)

    image_np, image_affine = image_process(load_image)
    image_rgb = (np.stack((image_np[0],) * 3, axis=-1) * 255).astype(np.uint8)

    return (image_rgb[0], [((0,0,0,0), 'target_box')])


def draw_bbox(image_np, x1, y1, x2, y2, slice_index=0):
    """
    Draw a bounding box on a specific slice of the image.
    
    Parameters:
    - image_np: 3D or 4D numpy array (multi-channel or multi-slice image)
    - x1, y1, x2, y2: Coordinates of the bounding box
    - slice_index: Index of the slice to draw the bounding box on (default is 0)
    """
    # Check the number of dimensions in the image
    if len(image_np.shape) == 4:  # if the image has multiple channels (e.g., (C, D, H, W))
        # Select a single slice based on the slice_index
        image_slice = image_np[0, slice_index, :, :]  # Assuming channel-first format (C, D, H, W)
    elif len(image_np.shape) == 3:  # for a simple 3D image (D, H, W)
        image_slice = image_np[slice_index, :, :]
    else:
        raise ValueError("Unsupported image format!")

    # Convert grayscale image to RGB for visualization
    if image_slice.dtype != np.uint8:
        image_slice = (image_slice * 255).astype(np.uint8)  # Scale if necessary

    image_rgb = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2RGB)

    # Draw the bounding box on the image using OpenCV
    if slice_index >= vis_box[0] and slice_index <= vis_box[3]:
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box with thickness 3

    # Convert back to PIL image for Gradio
    return Image.fromarray(image_rgb.astype(np.uint8))  # Ensure type is uint8


def load_image_with_bbox(load_image, x1, y1, x2, y2, slice_index):
    global image_np
    global vis_box
    vis_box = [x1, y1, x2, y2]

    image_np, image_affine = image_process(load_image)

    # Draw the bounding box on the image
    image_with_bbox = draw_bbox(image_np, x1, y1, x2, y2, slice_index)

    return (image_with_bbox, [((x1, y1, x2, y2), 'annotated_box')])



tabular_header = ['SEX','FIRST_SBP','FIRST_DBP','FIRST_HR','FIRST_RR','FIRST_BT','AGE','VISIBLE_STONE_CT','PANCREATITIS','DUCT_DILIATATION_10MM','DUCT_DILIATATION_8MM','Hb','PLT','WBC','ALP','ALT','AST','CRP','BILIRUBIN']
tabular_dtype = ['number'] * len(tabular_header)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(title_markdown)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            image = gr.File(type="filepath", label="Input File")
            tabular_input = gr.Dataframe(headers= tabular_header, datatype= tabular_dtype, label="Tabular Input", type="array", interactive=True, row_count=1, col_count=19)
            text = gr.Textbox(lines=1, placeholder="Please ask a question", label="Text Instruction")
            info = gr.Textbox(lines=1, label="Patient info", visible = False)

            with gr.Accordion("Parameters", open=False) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                        label="Temperature", )
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.1, interactive=True, label="Top P", )
                
            with gr.Row():
                btn_c = gr.ClearButton([image, tabular_input, text])
                btn = gr.Button("Run")
            text_out = gr.Textbox(lines=5, placeholder=None, label="Text Output", autofocus=True)
            
        with gr.Column():
            image_out = gr.AnnotatedImage(color_map={"target_mask": "#a89a00", "target_box": "#ffae00",'annotated_box': "#17cf7c"}, height=768,width=768)
            slice_slider = gr.Slider(minimum=0, maximum=31, step=1, interactive=True, scale=1, label="Selected Slice", min_width=50)
 
            with gr.Accordion("Region of Interest", open=False) as parameter_row:
                # Add inputs for bounding box coordinates
                x1 = gr.Slider(minimum=0, maximum=255, step=1, value = 0, label="x1")
                x2 = gr.Slider(minimum=0, maximum=255, step=1, value = 255, label="x2")
                y1 = gr.Slider(minimum=0, maximum=255, step=1, value = 0, label="y1")
                y2 = gr.Slider(minimum=0, maximum=255, step=1, value = 255, label="y2")
                
                # Button to trigger the update
                btn_a = gr.Button("Annotate Image")

            

    gr.Examples(examples=examples, inputs=[image, tabular_input, text, info])
    image.change(fn=load_image, inputs=[image], outputs=[image_out])
    btn.click(fn=inference, inputs=[image, tabular_input, text, temperature, top_p, x1, y1, x2, y2], outputs=[text_out, image_out])
    btn_a.click(fn=load_image_with_bbox, inputs=[image, x1, y1, x2, y2, slice_slider], outputs=[image_out])
    btn_c.click()
    slice_slider.change(fn=select_slice, inputs=slice_slider, outputs=[image_out])

demo.queue()
demo.launch(share=True)


