import cv2
import os
from glob import glob
import natsort
import shutil
from tqdm import tqdm

sublist= ['1','1-1','4','5','6','8']

camera = '_camera8_image_raw'

file_list = []


for sub in sublist:
    sub_folder_path = os.path.join(os.getcwd() , sub)

    if os.path.isdir(sub_folder_path):
        camera_path = os.path.join(sub_folder_path , camera)

        if os.path.isdir(camera_path):
            for path in natsort.natsorted(glob(os.path.join(camera_path , "*.jpg"))):
                if path.endswith(".jpg"):
                    file_list.append(path)



print(file_list)
print(len(file_list))


os.makedirs('back_8',exist_ok=True)
for i in tqdm(file_list):
    name = i.split('/')[5:]
    name = name[0] +'_'+ name[2]
    print(name)
    save_name = os.path.join(f'back_8/{name}')
    shutil.copy(i , save_name)