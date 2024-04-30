import numpy as np
import nibabel as nib
import os
import cv2
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
  start_slice = np.where(label == start_label)[0][0]
  end_slice = np.where(label == end_label)[-1][0] + 1

  # 추출 슬라이스 확인
  print(f"추출 슬라이스 범위: {start_slice} ~ {end_slice}")

  # 추출 수행
  extracted_slices = image[...,start_slice:end_slice]

  # 첫 번째 이미지로부터 정보 추출
  header, affine = nib.load(image_path).header, nib.load(image_path).affine

  # Nifti 이미지 생성
  cropped_image = nib.Nifti1Image(extracted_slices, affine, header)
  
  print(f'추출된 슬라이스 {cropped_image.shape[2]}')
  # 저장
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  nib.save(cropped_image, os.path.join(output_dir, f"cropped_image{patient_num}.nii.gz"))

  return image, extracted_slices

# 예시
patient_num = '0004'
image_path = f"/home/irteam/rkdtjdals97-dcloud-dir/datasets/BTCV/imagesTr/img{patient_num}.nii.gz"
label_path = f"/home/irteam/rkdtjdals97-dcloud-dir/datasets/BTCV/labelsTr/label{patient_num}.nii.gz"
start_label = 6
end_label = 11
output_dir = "./output"


# 슬라이스 추출 및 Crop
image, cropped_slices = extract_and_crop_slices(image_path, label_path, start_label, end_label, output_dir, patient_num)

# Plot the slice
plt.figure("check", (10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 48], cmap='gray')
plt.title('Origin')
plt.subplot(1, 2, 2)
plt.imshow(cropped_slices[...,0], cmap='gray')
plt.title('Crop')
plt.show()

