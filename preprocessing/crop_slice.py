import numpy as np
import nibabel as nib
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt

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
  image = nib.load(image_path).get_fdata()
  label = nib.load(label_path).get_fdata()

  print(f"전체 슬라이스 범위: {image.shape[2]}")

  # 슬라이스 범위 계산
  start_z = np.where(label == start_label)[-1]
  if len(start_z) == 0:
      print(f"Start label {start_label} not found in {label_path}, skipping...")
      return None, None

  start_value = np.bincount(start_z)

  end_z = np.where(label == end_label)[-1]
  if len(end_z) == 0:
      print(f"End label {end_label} not found in {label_path}, skipping...")
      return None, None

  end_value = np.bincount(end_z)

  
  start_slice  = np.argmax(start_value)
  end_slice  = np.argmax(end_value)

  # 추출 슬라이스 확인
  print(f"추출 슬라이스 범위: {start_slice} ~ {end_slice}")

  # 추출 수행
  extracted_slices = image[...,start_slice-10:end_slice]

  # 첫 번째 이미지로부터 정보 추출
  header, affine = nib.load(image_path).header, nib.load(image_path).affine

  # Nifti 이미지 생성
  cropped_image = nib.Nifti1Image(extracted_slices, affine, header)
  
  print(f'추출된 슬라이스 {cropped_image.shape[2]}')
  # 저장
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  nib.save(cropped_image, os.path.join(output_dir, f"{patient_num}.nii.gz"))

  return image, extracted_slices

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Crop slice")
  parser.add_argument("--image_dir", default='/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti/', type=str, help="image directory")
  parser.add_argument("--label_dir", default='/home/irteam/rkdtjdals97-dcloud-dir/research-contributions/SwinUNETR/BTCV/outputs/test2/', type=str, help="Prediction of image")
  parser.add_argument("--output_dir", default="/home/irteam/rkdtjdals97-dcloud-dir/datasets/Part2_nifti_crop/", type=str, help="output directory")
  args = parser.parse_args()

  # 예시
  images = args.image_dir
  labels = args.label_dir
  start_label = 11 # pancreas
  end_label = 6 # liver
  output_dir = args.output_dir

  for i in glob(os.path.join(labels,'*.nii.gz')):
        image_path = images + i.split('/')[-1]
        print(image_path)
        label_path = labels+i.split('/')[-1]
        print(label_path)
        patient_num = image_path.split('/')[-1].split('.')[0]
        # 슬라이스 추출 및 Crop
        extract_and_crop_slices(image_path, label_path, start_label, end_label, output_dir, patient_num)



