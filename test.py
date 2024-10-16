import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm.auto import  tqdm
import os
import fish_map
import box_module
import logging
import pandas as pd
import natsort
import glob

f_non_sig_rewnew = { # 좌석에 맞추어진 박스
    1: [410, 410, 519, 639],
    2: [520, 411, 639, 639],
    3: [373, 297, 452, 467],
    4: [453, 297, 526, 467],
    5: [364, 237, 420, 351],
    6: [421, 237, 475, 351],
    7: [352, 205, 394, 295],
    8: [395, 205, 443, 295],
    9: [170, 242, 232, 371],  
    10: [220, 214, 265, 310],
    11: [346, 184, 383, 247],  
    12: [384, 184, 417, 247],
    13: [237, 198, 277, 257],
}


model = YOLO('yolo8x.pt')
image_folder = natsort.natsorted(glob.glob(os.path.join('empty_front','**','*.jpg'),recursive=True))
image_folder = natsort.natsorted(glob.glob(os.path.join('cv2_diff_test/front/krri1' ,'**','*.jpg'), recursive=True))
image_folder = image_folder[270 : ]
image_folder = natsort.natsorted(glob.glob(os.path.join('empty_front','*.jpg')))
def apply_nms(boxes, scores, iou_threshold=0.6):
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)
    if isinstance(indices, list) and len(indices) > 0:
        return [boxes[i[0]] for i in indices]
    elif isinstance(indices, np.ndarray) and indices.size > 0:
        return [boxes[i] for i in indices.flatten()]
    else:
        return []
    
                

def nmx_box_to_cv2_loc(boxes):
    x1 , y1 , w, h = boxes
    x2 = x1 + w
    y2 = y1 + h

    return [x1 , y1 , x2 , y2]

for index , i in tqdm(enumerate(image_folder),total=len(image_folder)):
    image_name = i
    distort_image   = cv2.imread(i , cv2.IMREAD_COLOR)
    undisort_image = distort_image.copy()
   # undisort_image = cv2.resize(undisort_image , (640, 640) , interpolation=cv2.INTER_LANCZOS4)
    if distort_image is not None:
        h, w, _ = list(map(int, distort_image.shape))

        # Add black padding around the distorted image
        black = np.zeros(((int(w - h) // 2), w, 3), dtype=np.uint8)
        frame_new = cv2.vconcat([black, distort_image])
        frame_new = cv2.vconcat([frame_new, black])

        # Recalculate the height and width after padding
        h, w, _ = list(map(int, frame_new.shape))

        # Apply fisheye_to_plane_info mapping on the newly padded image
        undisort_image = np.array(fish_map.fisheye_to_plane_info(frame_new, h, w, 180, 90, 640, 0, 0))
        undisort_image = cv2.resize(undisort_image, (640  , 640), interpolation=cv2.INTER_LANCZOS4)
    
    # undisort_image = distort_image.copy()
    img = undisort_image.copy()
    result = model(undisort_image , verbose=True , classes = 0)

    boxes = []
    scores = []


    for res in result:
        for box in res.boxes:
            if int(box.cls) == 0:
                x1 , y1 , w , h = box.xyxy[0].tolist()
                score = box.conf
                boxes.append([int(x1), int(y1), int(w - x1), int(h - y1)])
                scores.append(float(score.detach().cpu().numpy()))

    nmx_boxes = apply_nms(boxes , scores)
    nmx_boxes = list(map(nmx_box_to_cv2_loc , nmx_boxes))
    
    for value in f_non_sig_rewnew.values():
        #value = [v * 2 for v in value]  # 각 원소에 2를 곱함
        x1 , y1 ,x2 , y2 = value
        cv2.rectangle(img , (x1 , y1) , (x2 , y2) , (0,255,0),2)
        

    # for idx , i in enumerate(nmx_boxes):
    #     x1, y1, x2, y2 = i
    #     center_x , center_y = (x1 + x2) // 2 , (y1 + y2) // 2
    #     if idx <= 99999:
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #         cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  
    #         cv2.putText(img, f'({center_x}, {center_y})', 
    #                     (center_x, center_y), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 
    #                     0.3, 
    #                     (255, 255, 255), 
    #                     1)
    #         cv2.putText(img , f"{idx}" ,(x1 , y1) , cv2.FONT_HERSHEY_SCRIPT_COMPLEX , 0.3 ,(0,0,255),1)
    # img_up = cv2.resize(img, (640 * 3, 640 * 3), interpolation=cv2.INTER_LANCZOS4)


    # x1 , y1 , x2 , y2 = [660,338,975,727]
    # img_up = img[y1 : y2 , x1 : x2]
    cv2.namedWindow("t",cv2.WINDOW_NORMAL)
    #cv2.namedWindow("up",cv2.WINDOW_NORMAL)
    cv2.imshow("t",img)
    
    #cv2.imshow("up",img_up)
    cv2.waitKey(0)
