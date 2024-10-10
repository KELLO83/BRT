import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import pdb
import shutil

Number = 12
seated_file = natsorted(glob(os.path.join(f"cv2_diff_test/raw_seat/{Number}", "*.jpg")))
seated_file = list(map(lambda x: os.path.basename(x), seated_file))

sub_folder = ["1.1", "4.1", "krri1"]
camera_name = "_camera6_image_raw"


file_list = []
for i in tqdm(sub_folder):
    sub_f = os.path.join(i, camera_name)
    if os.path.isdir(sub_f):
        for fi in os.listdir(sub_f):
            if fi.endswith(".jpg"):
                file_list.append(os.path.join(sub_f, fi))

print(len(file_list))
print(file_list[-1])


import pdb

SAVE_DIR = f"cv2_diff_test/removed_raw_seat/{Number}"
SAVE_DIR = f"Rm_Noise_front2"
os.makedirs(SAVE_DIR, exist_ok=True)

for i in tqdm(file_list):
    split = i.split("/")
    full_name = split[0] + "_" + split[-1]
    filename = os.path.join(SAVE_DIR, full_name)
    if not full_name in seated_file:
        shutil.copy(i, filename)
