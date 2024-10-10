import os
import shutil
from tqdm import tqdm
import file_raw_get

# 경로 설정 



number = 11
camera_back = 6
base_path = 'inspect'  # 'inspect' 폴더의 경로
target_dir = f'group/{number}'  # 이미지를 저장할 새로운 폴더 경로


if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for detect_folder in os.listdir(base_path):
    detect_path = os.path.join(base_path,    detect_folder)
    
    if os.path.isdir(detect_path) and detect_folder.startswith('detect_'):
        detect_prefix = detect_folder
        
        for camera_folder in ['_camera6_image_raw', '_camera8_image_raw']:
            camera_path = os.path.join(detect_path, camera_folder)
            
            if os.path.exists(camera_path):
                eleven_path = os.path.join(camera_path, f'{number}')
                
                if os.path.exists(eleven_path):
                    for image_name in tqdm(os.listdir(eleven_path)):
                        source_image_path = os.path.join(eleven_path, image_name)
                        
                        if os.path.isfile(source_image_path):
                            new_image_name = f"{detect_prefix}_{image_name}"
                            destination_image_path = os.path.join(target_dir, new_image_name)
                            
                            shutil.copy2(source_image_path, destination_image_path)
                            print(f"Copied {source_image_path} to {destination_image_path}")

print(f"=========================== len : {len(os.listdir(target_dir))} ================================")

file_raw_get.call(number , camera_back)