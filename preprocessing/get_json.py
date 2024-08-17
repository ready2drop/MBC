import os
import json
from glob import glob
import nibabel as nib
import argparse

def generate_json(data_dir):
    # Dictionary to store the final JSON structure
    json_structure = {
        "description": "DUMC",
        "labels": {},
        "licence": "ksm",
        "modality": {
            "0": "CT"
        },
        "name": "btcv",
        "numTest": 0,
        "tensorImageSize": "3D",
        "test": []
    }

    # Traverse the data directory to find .nii.gz files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, data_dir)
                # Add to the test list
                json_structure["test"].append({
                    "image": relative_path.replace("\\", "/"),
                    "label": relative_path.replace("\\", "/")
                })

    # Update the number of test cases
    json_structure["numTest"] = len(json_structure["test"])

    return json_structure

def save_json(json_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)

def main(data_dir, output_file):
    # NIfTI 파일 경로 패턴
    image_list = glob(os.path.join(data_dir, '*.nii.gz'))

    # Flag to check if any errors occurred
    error_occurred = False
    
    # 파일 검사
    for file_path in image_list:
        try:
            # NIfTI 파일 로드 시도
            our_img = nib.load(file_path)
            # NIfTI 데이터를 numpy 배열로 변환 시도
            our_data = our_img.get_fdata()
            print(f"File loaded and data read successfully: {file_path}")
        except nib.filebasedimages.ImageFileError as img_err:
            # NIfTI 파일 로드 오류
            print(f"ImageFileError loading file: {file_path}, error: {img_err}")
            error_occurred = True
        except EOFError as eof_err:
            # 압축 파일 읽기 오류
            print(f"EOFError reading file: {file_path}, error: {eof_err}")
            error_occurred = True
        except Exception as e:
            # 기타 예외 처리
            print(f"Error processing file: {file_path}, error: {e}")
            error_occurred = True

    # Only generate and save JSON if no errors occurred
    if not error_occurred:
        json_data = generate_json(data_dir)
        save_json(json_data, output_file)
        print(f"JSON file has been created and saved to {output_file}")
    else:
        print("Errors occurred during file processing. JSON file not created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get json")
    parser.add_argument("--data_dir", default='/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti/', type=str, help="image directory")
    args = parser.parse_args()
    
    output_file = os.path.join(args.data_dir, 'dataset.json')
    main(args.data_dir, output_file)