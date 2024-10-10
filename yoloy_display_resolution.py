import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm.auto import  tqdm
from natsort import  natsort
import matplotlib.pyplot as plt
import matplotlib
import os
import fish_map
import pandas as pd
import openpyxl
import matplotlib
import re
from super_gradients.training import models
from glob import glob
from natsort import natsorted

def apply_nms(boxes , scores , iou = 0.4):
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou)
    if isinstance(indices, list) and len(indices) > 0:
        return [boxes[i[0]] for i in indices]
    elif isinstance(indices, np.ndarray) and indices.size > 0:
        return [boxes[i] for i in indices.flatten()]
    else:
        return []


def generate_box(results , img , f):
    boxes = []
    scores = []
    colors = {
        'red' : (0,0,255),
        'blue' : (255,0,0),
    }
    if f is None:
        focus_area = [687 , 384 , 819 ,712]
    else:
        focus_area = f
    # if sig: # 10번좌석
    #     focus_area = [687 , 384 , 819 ,712]
    # else:
    #     focus_area = [154 , 216 , 225 , 391]
    # if sig: # 7번좌석
    #     focus_area = [993 , 314 , 1124 , 468]
    # else:
    #     focus_area = [316 , 183 , 378 , 260]
    
    # if sig: # 6번좌석
    #     focus_area = [1141,388,1257,530]
    # else:
    #     focus_area = [390,219,454,300]

    # if sig : # 11번좌석
    #     focus_area = [681,255,789,390]
    
    # else:
    #     focus_area = [161,138 , 222 , 222]

    # if sig : # 5번좌석
    #     focus_area = [1000,388,1154 , 674]
    
    # else:
    #     focus_area = [318 ,227 , 399 , 377]
    
    # x1 , y1 , x2 , y2 = focus_area
    
    # y_range = abs(y2 - y1)
    # y_pad = y_range // 4

    # middle_y = (y2 + y1) // 2
    
    # y1 = middle_y - y_pad
    # y2 = middle_y + y_pad

    # focus_area = x1 , y1 , x2 , y2
    
    # if sig :  # 4번좌석
    #     focus_area = [1188 , 488 , 1339 , 729]
    # else:
    #     focus_area = [420 , 271 , 515, 410]

    # if sig: # 24번좌석 
    #     foucs_area = [385 , 122 , 437 , 196]
    
    # else:
    #     foucs_area = [373 , 154 , 419 , 223]


    # if sig : # 23번좌석
    #     foucs_area = [314 , 120 , 393 , 205]
    # else:
    #     foucs_area = [312 , 156 , 381, 227]


    # PAD = 60
    # if sig: # 17번좌석
    #     foucs_area = [419 , 245 , 515, 496]
    #     x1 , y1 ,x2 , y2 = foucs_area
    #     middle_x = (x1 + x2) // 2
    #     middle_y = (y1 + y2) // 2
    #     y1 = middle_y - PAD
    #     y2 = middle_y + PAD

    #     foucs_area[1] = y1
    #     foucs_area[3] = y2

    # else:
    #     foucs_area = [402 , 255, 503, 465]

    #     x1 , y1 ,x2 ,y2 = foucs_area
    #     middle_x = (x1 + x2) // 2
    #     middle_y = (y1 + y2) // 2
    #     y1 = middle_y - PAD
    #     y2 = middle_y + PAD
    #     foucs_area[1] = y1
    #     foucs_area[3] = y2
    

    def calculate_intersection_area(box1, box2):
        # 두 박스의 교차 영역 계산c
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # 교차하지 않음

        return (x_right - x_left) * (y_bottom - y_top)

    def calculate_area(box):
        # 박스의 면적 계산
        return (box[2] - box[0]) * (box[3] - box[1])

    focus_area_area = calculate_area(focus_area)

    boxes = []
    scores = []

    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = box.conf
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                scores.append(float(score))

    nmx_boxes = apply_nms(boxes, scores)

    # 점유율 계산을 위한 변수 초기화
    highest_occupancy_rate = 0.0
    best_box = None
    cv2.rectangle(img, (focus_area[0], focus_area[1]), (focus_area[2], focus_area[3]), (255, 102, 255), 1)
    for box in nmx_boxes:
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
        box_coords = [x1, y1, x2, y2]
        center_x, center_y = x1 + w // 2, y1 + h // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
        # 교차 영역 계산
        intersection_area = calculate_intersection_area(box_coords, focus_area)
        occupancy_rate = intersection_area / focus_area_area

        # 가장 높은 점유율을 가진 박스 선택
        if occupancy_rate > highest_occupancy_rate:
            highest_occupancy_rate = occupancy_rate
            best_box = box_coords
            best_box_top_y = box_coords[1]

    if highest_occupancy_rate <= 0.5 :
        best_box = None
        is_exist = False
        box_count = 0

    

    else:

        # if abs(focus_area[1] - best_box_top_y) >= 20:
        #     best_box = None
        #     is_exist = False
        #     box_count = 0
        
        if False:
            pass
        
        else:
            x1, y1, x2, y2 = best_box
            w = x2 - x1
            h = y2 - y1
            center_x, center_y = x1 + w // 2, y1 + h // 2

            # focus_area와 선택된 박스를 이미지에 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(img, f'({center_x}, {center_y})', 
                        (center_x, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.3, 
                        (255, 255, 255), 
                        1)

            is_exist = True  # 선택된 박스가 존재하므로 True로 설정
            box_count = 1    # 선택된 박스의 개수

    if is_exist is None:
        is_exist = False
    return img, box_count, is_exist
    
def generate_box__(result , img , color='red'):
    boxes = []
    scores = []
    colors = {
        'red' : (0,0,255),
        'blue' : (255, 0 ,0),
    }

    for res in result:
        for box in res.boxes:
                x1 , y1 ,x2 , y2 = box.xyxy[0].tolist()
                score = box.conf
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                scores.append(float(score))

    nmx_boxes = apply_nms(boxes , scores)
    box_count = 0
    for box in nmx_boxes:
        x1, y1, w, h = box
        center_x, center_y = x1 + w // 2, y1 + h // 2
        
        cv2.rectangle(img , (x1 , y1) ,(x1+w , y1+h) , (0,255,0) , 1)
        cv2.circle(img, (center_x, center_y), 5, colors[color], -1)
        # 좌표를 이미지에 작게 표시
        cv2.putText(img, f'({center_x}, {center_y})', 
                    (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, 
                    (255, 255, 255), 
                    1)

        box_count += 1            
    return img , box_count 

distort_images = natsorted(glob(os.path.join('cv2_diff_test/raw_seat/8','*.jpg')))
model = YOLO('yolov8x.pt').to("cuda")


total_orgin = 0
total_resize = 0
totla_new = 0
t_count = 0
t3_count = 0
for index , i in tqdm(enumerate(distort_images) , total=len(distort_images)):
    image = cv2.imread(i)
    # image = image [: , 525:]
    resize_image = cv2.resize(image , dsize=(600,600))

    img = image
    h, w = img.shape[:2]
    scale = 600 / max(h, w)
    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
    delta_w = 600 - resized_img.shape[1]
    delta_h = 600 - resized_img.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    result_1 = model(image , verbose=False , classes= 0)
    result_2 = model(resize_image , verbose=False , classes = 0)
    result_3 = model(new_img , verbose=False , classes = 0)

    r1 , b1 , t1= generate_box(result_1 , image.copy()  , f = [1136,381,1248,534])
    r2 , b2 , t2= generate_box(result_2 , resize_image.copy()  ,f = None)
    r3 , b3 , t3= generate_box(result_3 , new_img.copy()  , f=[354,249,389,296])

    if t1:
        t_count  +=1
    if t3:
        t3_count +=1
    
    # r1, r2, r3 = map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB), [r1, r2, r3])


    print(f"orgin {b1} resize {b2} new {b3}")
    
    total_orgin += b1
    total_resize += b2
    totla_new+= b3

    # if index <=3:
    #     plt.figure(figsize=(12,12))
    #     plt.subplot(1,3,1)
    #     plt.title(b1)
    #     plt.imshow(r1)
    #     plt.subplot(1,3,2)
    #     plt.title(b2)
    #     plt.imshow(r2)
    #     plt.subplot(1,3,3)
    #     plt.title(b3)
    #     plt.imshow(r3)
    #     plt.show()
    #     plt.close()
    cv2.namedWindow("o",cv2.WINDOW_NORMAL)
    cv2.namedWindow("r",cv2.WINDOW_NORMAL)
    cv2.namedWindow("n",cv2.WINDOW_NORMAL)
    cv2.imshow('o',r1)
    cv2.imshow('r',r2)
    cv2.imshow("n",r3)
    cv2.waitKey(0)

print(total_orgin)
print(total_resize)
print(totla_new)

print(t_count)
print(t3_count)


# 819
#260
#733

#481
#435
#471

# 2량
# 6776
# 4119
# 6625  