from glob import glob
import os
import re
from natsort import natsorted
import cv2
from tqdm import tqdm
from typing import List


def call(number_ : int , camera_number : int):
    number = number_
    file_list = natsorted(os.listdir(f'group/{number}'))

    camera_name = f'_camera{camera_number}_image_raw'

    save_dir = f'group/raw/{number}'
    os.makedirs(save_dir , exist_ok=True)
    for i in tqdm(file_list):
        string = i.split('_')
        file_name = string[2] + '_' +  string[3]
        folder_name = re.sub(r'[^0-9\-]' , '',string[1])
        
        target_path = os.path.join(folder_name , camera_name , file_name)


        image = cv2.imread(target_path)
        try:
            cv2.imwrite(os.path.join(save_dir , i) , image)
        except:
            print("ERROR : ", os.path.join(save_dir , i))
