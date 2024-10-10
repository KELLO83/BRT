import os
import cv2
from natsort import natsorted
import pdb
from glob import glob
def replace_images_in_tree(source_folder, target_folder):
    source_files = os.listdir(source_folder)
    for root, dirs, files in os.walk(target_folder):
        for file_name in files:
            if file_name in source_files:
                source_file_path = os.path.join(source_folder, file_name)
                target_file_path = os.path.join(root, file_name)
                
                source_image = cv2.imread(source_file_path)
                cv2.imwrite(target_file_path, source_image)
                
                print(f"Replaced {file_name} in {root}")
                #pdb.set_trace()
                

# 경로 설정
source_folder = 'detect_1-1/_camera8_image_raw'  # 복사할 소스 폴더
target_folder = 'inspect/detect_scen1-1/_camera8_image_raw'  # 복사받을 대상 트리 구조의 폴더

# 이미지 교체 함수 호출
replace_images_in_tree(source_folder, target_folder)
