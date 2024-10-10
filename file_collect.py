import os
import shutil
from tqdm import tqdm
sub_list = ['1.1', '4.1', 'krri1']
file_list = []

for sub_folder in sub_list:
    sub_f = os.path.join(os.getcwd(), sub_folder)
    
    target = os.path.join(sub_f, '_camera6_image_raw')
    
    if os.path.exists(target):
        f_list = os.listdir(target)
        for file_name in f_list:
            if file_name.endswith(".jpg"):
                file_list.append(os.path.join(target, file_name))

print(len(file_list))

SAVE_DIR = 'front2_all'
os.makedirs(SAVE_DIR , exist_ok=True)
for i in tqdm(file_list):
    T = i.split('/')
    t = T[-3] + '_' + T[-1]
    New_name = os.path.join(SAVE_DIR , t)
    shutil.copy(i , New_name)    
    

