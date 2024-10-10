import os
from glob import glob
import cv2
import re
import shutil
import natsort 
from tqdm import tqdm
import pdb

base_dir = 'group/raw'
seat_number = 16
SAVE_DIR = f'cv2_diff_test/raw_seat/{seat_number}'
os.makedirs(SAVE_DIR , exist_ok=True)
main = os.getcwd()

base_dir = os.path.join(main , base_dir)
file_list = []
for root , dirs , files in os.walk(base_dir):
    if re.search(rf'\b{seat_number}\b', os.path.basename(root)):
        for file in files:
            if file.endswith(".jpg"):
                absolute_path = os.path.join(root , file)
                relative_path = os.path.relpath(absolute_path , main)
                file_list.append(relative_path)


file_list = natsort.natsorted(file_list)
if not file_list :
    raise FileNotFoundError

for i in tqdm(file_list):
    src_path = i
    i = i.split('/')
    i[1] =  str(i[1]).replace('detect_scen','').replace('detect_','')
    full_path = '/'.join([i[1] , i[2]  , i[4]])
    image = cv2.imread(full_path , cv2.IMREAD_COLOR)
    if image is None:
        print(src_path)
        print(full_path)
        pdb.set_trace()
    save_name = f"{i[1]}_{os.path.basename(full_path)}"
    save_name = os.path.join(SAVE_DIR , f"{save_name}")
    cv2.imwrite(f"{save_name}" , image)


