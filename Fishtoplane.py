import cv2
import os
from tqdm.auto import  tqdm
import fish_map
import numpy as np

sub_list = ['1','1-1','1.1','4','4.1','5','6','8','krri1']
sub_list = ['cv2_diff_test/raw_seat/24']
root_path = os.getcwd()

file_list = []


SAVE_DIR = '8_undistort'

for f in tqdm(sub_list):
    f_p = os.path.join(root_path , f)

    for dirpath , dirname , filenames in os.walk(f_p):
        for file in tqdm(filenames):
            if file.endswith('.jpg'):
                file_path = os.path.join(dirpath , file)
                frame = cv2.imread(file_path , cv2.IMREAD_COLOR)
                h, w, _ = list(map(int, frame.shape))

                black = np.zeros(((int(w - h) // 2), w, 3), dtype=np.uint8)

                frame_new = cv2.vconcat([black, frame])
                frame_new = cv2.vconcat([frame_new, black])

                h, w, _ = list(map(int, frame_new.shape))

                undistorted = fish_map.fisheye_to_plane_info(frame_new , h , w , 180 , 90 , 600 , 0 , 0)
                
                file_path = file_path.split('/')
                base_folder_name = file_path[-3]
                camera_name = file_path[-2]
                image_name = file_path[-1]

                full_path = os.path.join(SAVE_DIR , base_folder_name , camera_name , image_name)
                os.makedirs(os.path.dirname(full_path) , exist_ok=True)

                # cv2.namedWindow("t",cv2.WINDOW_NORMAL)
                # cv2.imshow("t", undistorted)
                # cv2.waitKey(0)
                cv2.imwrite(full_path , undistorted)





