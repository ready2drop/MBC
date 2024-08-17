import numpy as np
import nibabel as nib
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt

def find_liver_bounding_box(label, liver_label):
    """
    Find the bounding box (x, y coordinates) of the liver label in the 2D slice with the most instances.

    Args:
        label: 3D numpy array representing the label image.
        liver_label: Label value for the liver.

    Returns:
        Tuple of (start_x, end_x, start_y, end_y) representing the bounding box coordinates.
    """
    liver_slices = np.where(label == liver_label)[-1]
    if len(liver_slices) == 0:
        print(f"Liver label {liver_label} not found in the label image")
        return None

    liver_counts = np.bincount(liver_slices)
    most_visible_slice_index = np.argmax(liver_counts)
    liver_slice = label[..., most_visible_slice_index]
    liver_coordinates = np.where(liver_slice == liver_label)

    start_x = np.min(liver_coordinates[0])
    end_x = np.max(liver_coordinates[0])
    start_y = np.min(liver_coordinates[1])
    end_y = np.max(liver_coordinates[1])
    
    return (start_x, end_x, start_y, end_y)
  
def extract_and_crop_slices(image_path, label_path, start_label, end_label, output_dir, patient_num):
  """
  CT 이미지 슬라이스를 추출하고 Crop합니다.

  Args:
    image_path: CT 이미지 파일 경로
    label_path: 레이블 이미지 파일 경로
    start_label: 추출 시작 레이블
    end_label: 추출 종료 레이블
    output_dir: 추출된 슬라이스 저장 디렉토리

  Returns:
    추출된 슬라이스 이미지 (3D NumPy array)
  """

  # 이미지 불러오기
  image_data = nib.load(image_path).get_fdata()
  label_data = nib.load(label_path).get_fdata()

  print(f"전체 슬라이스 범위: {image_data.shape[2]}")

  # 슬라이스 범위 계산
  start_z = np.where(label_data == start_label)[-1]
  if len(start_z) == 0:
      print(f"Start label {start_label} not found in {label_path}, skipping...")
      return None, None

  start_value = np.bincount(start_z)

  end_z = np.where(label_data == end_label)[-1]
  if len(end_z) == 0:
      print(f"End label {end_label} not found in {label_path}, skipping...")
      return None, None

  end_value = np.bincount(end_z)

  
  start_slice  = np.argmax(start_value)
  end_slice  = np.argmax(end_value)

  # 추출 슬라이스 확인
  print(f"추출 슬라이스 범위: {start_slice} ~ {end_slice}")

  # # Find the bounding box coordinates of the liver label
  # bbox = find_liver_bounding_box(label_data, end_label)
  # if bbox is None:
  #     return None, None

  # start_x, end_x, start_y, end_y = bbox
  # # Print bounding box coordinates
  # print(f"Start X: {start_x}, End X: {end_x}")
  # print(f"Start Y: {start_y}, End Y: {end_y}")
  # print(f"Strat Z: {start_slice}, End Z: {end_slice}")

  # # Perform extraction based on bounding box coordinates
  # extracted_slices = image_data[start_x:300, start_y:end_y, start_slice-5:end_slice+2]
  # Perform extraction based on bounding box coordinates
  extracted_slices = image_data[..., start_slice-5:end_slice+2]

  # 첫 번째 이미지로부터 정보 추출
  header, affine = nib.load(image_path).header, nib.load(image_path).affine

  # Nifti 이미지 생성
  cropped_image = nib.Nifti1Image(extracted_slices, affine, header)
  
  print(f'추출된 슬라이스 {cropped_image.shape[2]}')
  
  # 저장
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  nib.save(cropped_image, os.path.join(output_dir, f"{patient_num}.nii.gz"))

  return image_data, extracted_slices

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Crop slice")
  parser.add_argument("--image_dir", default='/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti/', type=str, help="image directory")
  parser.add_argument("--label_dir", default='/home/irteam/rkdtjdals97-dcloud-dir/research-contributions/SwinUNETR/BTCV/outputs/test2/', type=str, help="Prediction of image")
  parser.add_argument("--output_dir", default="/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti_crop/", type=str, help="output directory")
  parser.add_argument("--start_label", type=int, default=11, help="Start label for extraction")
  parser.add_argument("--end_label", type=int, default=6, help="End label for extraction")
  args = parser.parse_args()

  # 예시
  images_dir = args.image_dir
  labels_dir = args.label_dir
  output_dir = args.output_dir
  start_label = args.start_label
  end_label = args.end_label

  for label_path in glob(os.path.join(labels_dir, '*.nii.gz')):
        image_path = os.path.join(images_dir, os.path.basename(label_path))
        patient_num = os.path.basename(image_path).split('.')[0]

        if os.path.exists(image_path):
            print(f"Processing image: {image_path}")
            print(f"Processing label: {label_path}")
            extract_and_crop_slices(image_path, label_path, start_label, end_label, output_dir, patient_num)
        else:
            print(f"Image not found for label: {label_path}")



