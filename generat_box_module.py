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
from PIL import Image
import copy
import math
import box_module


false_alarm_count = 0
signal_not_same = 0

sit_count = {key : 0 for key in range(1, 14)}
sit_count_old = {key : 0 for key in range(1,14)}
f_non_sig = {
    1: [376, 405, 587, 638],
    2 : [520,411,639,639],
    3: [356, 293, 468, 519],
    4: [448, 289, 549, 437],
    5: [349, 235, 418, 331],
    6: [416, 233, 484, 320],
    7: [337, 195, 403, 277],
    8: [402, 203, 446, 267],
    9: [146, 218, 242, 429],
    10: [214, 190, 275, 321],
    11: [332, 166, 378, 203],
    12: [381, 166, 419, 235],
    13: [246, 174, 285, 219],
}

f_non_sig_rewnew = {
    1: [410, 410, 519, 639],
    2 : [520,411,639,639],
    3: [373, 297, 452, 467],
    4: [453, 297, 526, 467],
    5: [364, 237, 420, 351],
    6: [421, 237, 475, 351],
    7: [352, 205, 394, 295],
    8: [395, 205, 443, 295],
    9 : [170, 242, 232, 371],  
    10 : [220,218 ,265,310 ],
    11: [346, 180, 383 ,247],  
    12: [384, 180, 417, 247],
    13: [237, 199, 277, 257], 
}

f_non_sig_rewnew = {
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




scale_factor = 640 / 600

# 좌표를 640x640 크기에 맞게 스케일링하는 함수
def scale_coordinates(coordinates, scale):
    return [int(coord * scale) for coord in coordinates]

# 스케일링된 좌표 저장
# f_non_sig = {key : scale_coordinates(value, scale_factor) for key, value in f_non_sig.items()}
# f_non_sig_rewnew = {key : scale_coordinates(value, scale_factor) for key, value in f_non_sig_rewnew.items()}

#print(f_non_sig)


flag_call =False

def convert_2d(boxes):
    x1 , y1 , w, h = boxes
    x2 = x1 + w
    y2 = y1 + h

    return [x1 , y1 , x2 , y2]

def calc_sit_count(sit_number):
    global sit_count
    sit_number = set(sit_number)
    for i in sit_number:
        target = int(i)
        if target in sit_count:
            sit_count[target] += 1

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    iou = inter_area / union_area
    return iou


def sit_old_distinguish(boxes, img , nunber):
    global f_non_sig
    img = img.copy()  # deepcopy가 필요없으면 copy() 사용
    boxes = list(map(convert_2d, boxes))

    for i in boxes:
        x1, y1, x2, y2 = i
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    matched_boxes = []

    for target in boxes:
        occupancy_rate_max = 0
        candidate = []
        
        for sit_number, sit_cordinate in f_non_sig.items():
            focus_area_area = calculate_area(sit_cordinate)
            intersection_area = calculate_intersection_area(target, sit_cordinate)
            occupancy_rate = intersection_area / focus_area_area

            if occupancy_rate_max < occupancy_rate:
                occupancy_rate_max = occupancy_rate
                candidate = [[sit_number, target]]

        if occupancy_rate_max > 0 and candidate:
            x1, y1, x2, y2 = candidate[0][1]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw rectangle and center
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(img, f'({center_x}, {center_y})', (center_x, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            matched_boxes.append(tuple(target))

            # Increment the count for this seat number
            if candidate[0][0] in f_non_sig:
                sit_count_old[candidate[0][0]] += 1

    return img

def sit_distinguish(boxes, img , number , image_name):
    global f_non_sig
    img2 = copy.deepcopy(img)
    img = copy.deepcopy(img)
    boxes = list(map(convert_2d, boxes))
    boxes_copy = boxes.copy()

    for i in boxes:
        x1, y1, x2, y2 = i
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1) 
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  
        cv2.putText(img, f'({center_x}, {center_y})', 
                    (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, 
                    (255, 255, 255), 
                    1)

    results = []
    draw = []

    for target in boxes:
        candidates = {}
        occupancy_rate_list = []
        for sit_number, sit_cordinate in f_non_sig.items():
            focus_area_area = calculate_area(sit_cordinate)
            intersection_area = calculate_intersection_area(target, sit_cordinate)
            occupancy_rate = intersection_area / focus_area_area

            if occupancy_rate > 0.5:
                candidates[sit_number] = sit_cordinate

            print("sit_number : {} ratio : {}".format(sit_number, round(occupancy_rate, 2)))
            occupancy_rate_list.append(occupancy_rate)

            # if occupancy_rate > 0.9:
            #     x1 , y1 , x2 , y2 = target
            #     center_x = ( x1 + x2) // 2
            #     center_y = ( y1 + y2) // 2

            #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1) 
            #     cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  
            #     cv2.putText(img, f'({center_x}, {center_y})', 
            #                 (center_x, center_y), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 
            #                 0.3, 
            #                 (255, 255, 255), 
            #                 1)
            #     draw.append(tuple(target)) 


        best = None
        min_distance = math.inf
        x1, y1, x2, y2 = target
        center_x = (x1 + x2) // 2
        t_x, t_y = center_x, y1

        for sit_number, sit_cordinate in candidates.items():
            x1, y1, x2, y2 = sit_cordinate
            center_x = (x1 + x2) // 2
            x, y = center_x, y1

            distance = np.sqrt((x - t_x) ** 2 + (y - t_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                best = [sit_number, target] 

        if best:
            results.append(best)

        print()


    sit_number = []
    for i, k in results:
        if i is None:
            continue
        sit_number.append(i)
        x1, y1, x2, y2 = k  
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1) 
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  
        cv2.putText(img, f'({center_x}, {center_y})', 
                    (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, 
                    (255, 255, 255), 
                    1)

        draw.append(tuple(k)) 

    sit_number = sorted(sit_number)
    S = set(sit_number)

    for i in sit_number:
        print(f"{i} 번째 좌석 착석 ...")

    if len(sit_number) != len(S):
        sit_number = set(sit_number)

    calc_sit_count(sit_number)

    # EMPTY_BOX = []
    # for i in boxes_copy:
    #     if tuple(i) not in draw:  
    #         x1, y1, x2, y2 = i
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1) 
    #         EMPTY_BOX.append(i)

    # img , number_list , iou = remain_excute(EMPTY_BOX , img , sit_number)
    # number_list = list(zip(number_list , iou))

    global false_alarm_count
    global signal_not_same


    renew_copy = copy.deepcopy(img)


    K = zip(f_non_sig.values() , f_non_sig_rewnew.values())
    for j , q in K:
        x1 , y1 ,x2 ,y2 = q
        cv2.rectangle(renew_copy ,(x1 , y1), (x2 , y2) , (255, 0 , 0) , 1)
        x1 , y1 ,x2 , y2 = j
        cv2.rectangle(img , (x1 , y) ,(x2 , y2) , (255,0 , 0) , 1)



#   flag -> True 기존꺼 
    if boxes:
        print(image_name)
        FLAG = False
        match  , loc = box_module.second(boxes , img2 , number= 0 , FLAG=True) 
        print(match)
        print(loc)
        T = loc.keys()

        if not FLAG:
            img = renew_copy
            f_non_sig = f_non_sig_rewnew

        for i in T:
            s = f_non_sig[i]
            x1 , y1 ,x2 , y2 = s
            cv2.rectangle(img , (x1 , y1) , (x2 , y2) , (255,255 , 0) , 1)


        cv2.namedWindow("p",cv2.WINDOW_NORMAL)
        cv2.imshow("p" ,  img)
        cv2.waitKey(0)


    print("착석 사람 :",len(sit_number))    
    print("box 총 갯수 :" ,len(boxes_copy))
    print("좌석 : {}  신호 넘어옴 : {} 출처가다름 : {}".format(number , false_alarm_count , signal_not_same))

    return img

def remain_excute(boxes , img , finsied):
    global f_non_sig
    f = f_non_sig.copy()

    sit_list = []
    sit_iou = []
    for i in finsied:
        del f[i]

    for i in boxes:
        best_iou = 0 
        best_index = None
        for index , cordidate in f.items():
            focus_area_area = calculate_area(cordidate)
            area_iou = calculate_iou(i , cordidate)

            if area_iou > best_iou:
                best_iou = area_iou
                best_index = index

        
        if best_iou > 0.1 :
            x1, y1, x2, y2 = i
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2


            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1) 
            cv2.circle(img, (center_x, center_y), 5, (255, 255, 0), -1)  
            cv2.putText(img, f'({center_x}, {center_y})', 
                        (center_x, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.3, 
                        (255, 255, 255), 
                        1)
            
            X1 , Y1 , X2 , Y2 = f_non_sig[best_index]
            cv2.rectangle(img , ( X1 , Y1) , (X2 , Y2) , (255,0,255), 1)
            sit_list.append(best_index)
            sit_iou.append(best_iou)

    return img , sit_list , sit_iou

def calculate_intersection_area(box1, box2):
    # 두 박스의 교차 영역 계산
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


def apply_nms(boxes, scores, iou_threshold=0.6):
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)
    if isinstance(indices, list) and len(indices) > 0:
        return [boxes[i[0]] for i in indices]
    elif isinstance(indices, np.ndarray) and indices.size > 0:
        return [boxes[i] for i in indices.flatten()]
    else:
        return []

def compare(list1 , list2):

    if sorted(list1) == sorted(list2):
        return True
    return False

def call_generate_box_only(results , img , number , sig=True , crop=False):

    img = copy.deepcopy(img)
    number = int(number)
    boxes = []
    scores = []

    try:
         focus_area = f_non_sig[number]
    except:
        focus_area = [0,0,100,100]
        

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

    # _  = sit_distinguish(nmx_boxes , img)

    highest_occupancy_rate = 0.0
    best_box = None
    highest_occupancy_box = []
    highest_occupancy_rate = []
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

        if occupancy_rate > 0.5:
            highest_occupancy_box.append(box_coords)
            highest_occupancy_rate.append(occupancy_rate)

    if not highest_occupancy_box:
        best_box = None
        is_exist = False
        box_count = 0
        return img , box_count , is_exist , best_box
    
    f_center_y = focus_area[1]
    min_distance = float('inf')
    closer_box = []
    for index , box_candidate  in enumerate(highest_occupancy_box):
        x1 , y1 , x2 , y2 = box_candidate

        distance = abs(f_center_y - y1)
        
        if distance < min_distance:
            min_distance = distance
            closer_box = [[box_candidate , highest_occupancy_rate[index]]]

    highest_occupancy_rate = closer_box[0][1]
    best_box = closer_box[0][0]
    if highest_occupancy_rate >= 0.9:
        x1, y1, x2, y2 = best_box
        w = x2 - x1
        h = y2 - y1
        center_x, center_y = x1 + w // 2, y1 + h // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 빨간색 사각형
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(img, f'({center_x}, {center_y})',
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        is_exist = True
        box_count = 1
        return img, box_count, is_exist, best_box

    else:
        pass
    w = x2 - x1
    h = y2 - y1
    center_x, center_y = x1 + w // 2, y1 + h // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1) 
    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.putText(img, f'({center_x}, {center_y})',
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1)
    is_exist = True
    box_count = 1


    return img, box_count, is_exist, best_box


def call_generate_box_final(results , img , number  , image_name):

    global f_non_sig
    img = copy.deepcopy(img)
    number = int(number)
    boxes = []
    scores = []

    try:
         focus_area = f_non_sig[number]
    except:
        focus_area = [0,0,100,100]
        

    def calculate_intersection_area(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0  

        return (x_right - x_left) * (y_bottom - y_top)

    def calculate_area(box):
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

    _  = sit_distinguish(nmx_boxes , img , number , image_name)
    #_ = sit_old_distinguish(nmx_boxes , img , number)

    highest_occupancy_rate = 0.0
    best_box = None
    highest_occupancy_box = []
    highest_occupancy_rate = []
    cv2.rectangle(img, (focus_area[0], focus_area[1]), (focus_area[2], focus_area[3]), (255, 102, 255), 1)
    

    LIMIT_Y= (focus_area[1] + focus_area[3]) // 2
    one_step = abs(focus_area[1] - LIMIT_Y) // 2
    one_step = one_step // 4
    one_step = one_step * 3
    one_line_y_max = focus_area[1] + one_step
    one_line_y_min = focus_area[1] - one_step
    x_line = (focus_area[0] + focus_area[2]) // 2
    cv2.line(img , (x_line , one_line_y_max) , (x_line , one_line_y_min) , (255,100,100),1)

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

        if occupancy_rate > 0.5:
            highest_occupancy_box.append(box_coords)
            highest_occupancy_rate.append(occupancy_rate)

    if not highest_occupancy_box:
        best_box = None
        is_exist = False
        box_count = 0
        return img , box_count , is_exist , best_box
    
    f_center_y = focus_area[1]
    min_distance = float('inf')
    closer_box = []
    for index , box_candidate  in enumerate(highest_occupancy_box):
        x1 , y1 , x2 , y2 = box_candidate

        distance = abs(f_center_y - y1)
        
        if distance < min_distance:
            min_distance = distance
            closer_box = [[box_candidate , highest_occupancy_rate[index]]]

    highest_occupancy_rate = closer_box[0][1]
    best_box = closer_box[0][0]
    if highest_occupancy_rate >= 0.9:
        x1, y1, x2, y2 = best_box
        w = x2 - x1
        h = y2 - y1
        center_x, center_y = x1 + w // 2, y1 + h // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 빨간색 사각형
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(img, f'({center_x}, {center_y})',
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        is_exist = True
        box_count = 1
        return img, box_count, is_exist, best_box

    else:

        global flag_call
        one_step = 25

        if flag_call == False:
            print(one_step)
            flag_call =True
        if abs(y1 - focus_area[1]) >= one_step:
            best_box = None
            is_exist = False
            box_count = 0
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)  # 민트색 사각형
            print(one_step)
            return img, box_count, is_exist, best_box


    w = x2 - x1
    h = y2 - y1
    center_x, center_y = x1 + w // 2, y1 + h // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1) 
    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.putText(img, f'({center_x}, {center_y})',
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1)
    is_exist = True
    box_count = 1


    return img, box_count, is_exist, best_box