import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import pdb
import shutil

Number = 13
seated_file = natsorted(glob(os.path.join(f"cv2_diff_test/raw_seat/{Number}", "*.jpg")))
seated_file = list(map(lambda x: os.path.basename(x), seated_file))
all_file = natsorted(glob(os.path.join("cv2_diff_test/front", "**", "*.jpg"), recursive=True))

refine_file = []
for i in tqdm(all_file):
    split = i.split("/")
    folder_name = split[-2]
    file_name = split[-1]
    file_name = folder_name+'_'+file_name
    if not file_name in seated_file:
        refine_file.append(i)

print(len(refine_file))
SAVE_DIR = f"cv2_diff_test/removed_raw_seat/{Number}"
os.makedirs(SAVE_DIR, exist_ok=True)

for i in tqdm(refine_file):
    split = i.split("/")
    SCEN = split[-2]
    full_name = SCEN + '_' + split[-1]
    file_name = os.path.join(SAVE_DIR, full_name)
    shutil.copy(i, file_name)
