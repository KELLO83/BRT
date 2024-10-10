import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import pdb
import shutil
import re

def string__(string):
    match = re.search(r'\d',string)
    name = string[match.start() : ]

    return name

Number = 24
full_image = natsorted(glob(os.path.join(f"back_6", "*.jpg")))
seat_image = natsorted(glob(os.path.join(f"cv2_diff_test/raw_seat/{Number}" , "*.jpg")))

full_image = natsorted(os.listdir('back_6'))
seat_image = natsorted(os.listdir(f"cv2_diff_test/raw_seat/{Number}"))
seat_image = list(map(string__ , seat_image))


file_list = []
in_file_list = []
for i in full_image:
    if not i in seat_image:
        file_list.append(i)
    else:
        in_file_list.append(i)

print(len(full_image))
print(len(seat_image))
print(len(file_list))
print(len(in_file_list))
print(file_list[-1])

SAVE_DIR = f'cv2_diff_test/removed_raw_seat/{Number}'
os.makedirs(SAVE_DIR ,exist_ok=True)


for i in tqdm(file_list):
    src_name = os.path.join('back_6',i)
    save_name = os.path.join(SAVE_DIR , i)
    shutil.copy(src_name , save_name)


    