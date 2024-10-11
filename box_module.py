import os
import cv2
import numpy as np
from natsort import  natsort
import os
import collections
import collections

def scale_coordinates(coordinates, scale):
    return [int(coord * scale) for coord in coordinates]

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

for key in f_non_sig:
    # f_non_sig에서 y1 값 추출 (두 번째 요소)
    y1_value = f_non_sig[key][1]
    # f_non_sig_rewnew에서 해당 키의 y1 값을 f_non_sig의 y1 값으로 수정
    f_non_sig_rewnew[key][1] = y1_value

# 결과 출력
f_non_sig_rewnew = f_non_sig_rewnew




scale_factor = 640 / 600
f_non_sig = {key : scale_coordinates(value, scale_factor) for key, value in f_non_sig.items()}
f_non_sig_rewnew = {key : scale_coordinates(value, scale_factor) for key, value in f_non_sig_rewnew.items()}


def use_final(mapper, p_boxes): # 검증완료 일단은 ? 이거사용
    """
    박스를 사람과 매칭하는 과정 (중복 허용하지 않음, 우선순위 고려).
    사람과 박스에 대한 선호도는 mapper에 저장되어 있음.
    """
    n_boxes = len(mapper[0])  # 박스의 개수 (열의 수)
    n_people = len(mapper)  # 사람의 개수 (행의 수)
    
    matches = [-1] * n_people  # 각 인덱스마다 사람이 매칭된 박스의 idx 번호
    match_coords = []
    box_owners = [-1] * n_boxes  # 각 박스를 차지한 사람의 idx 번호
    
    for person_idx in range(n_people):  # 각 사람에 대해 매칭 시도
        best_score = -1
        best_index = None
        
        for box_idx in range(n_boxes):  # 각 박스에 대해 점수를 비교
            score = mapper[person_idx][box_idx]
            
            if score > best_score:
                best_score = score
                best_index = box_idx
        
        if best_score >= 0.3:
            current_owner = box_owners[best_index]
            
            if current_owner == -1:
                # 박스를 차지한 사람이 없다면, 현재 사람에게 할당
                matches[person_idx] = best_index
                box_owners[best_index] = person_idx
                match_coords.append([best_index, p_boxes[person_idx]])
            else:
                # 이미 박스를 차지한 사람이 있다면, 우선순위 비교
                current_owner_score = mapper[current_owner][best_index]
                
                if best_score > current_owner_score:
                    # 현재 사람이 더 높은 점수를 가지면 박스를 양보받음
                    matches[current_owner] = -1  # 이전 소유자는 매칭 해제
                    matches[person_idx] = best_index
                    box_owners[best_index] = person_idx
                    match_coords.append([best_index, p_boxes[person_idx]])
                    
        else:
            matches[person_idx] = None  # 기준 점수보다 낮으면 매칭하지 않음

    return matches, match_coords

def first(boxes, img, number = 0):
    global f_non_sig

    f_non_sig_len = len(f_non_sig)  # 박스의 수
    img = img.copy()

    mapper = [[0 for _ in range(f_non_sig_len)] for _ in range(len(boxes))]  


    for i in f_non_sig_rewnew.values():
        x1 , y1  ,x2 ,y2 = i
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    for i in boxes:
        x1, y1, x2, y2 = i
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)


    for row, cordidate in enumerate(f_non_sig.values()):  
        for col, people in enumerate(boxes): 
            focuse_area_area = calculate_area(cordidate)
            intersection_area = calculate_intersection_area(people, cordidate)
            occupancy_rate = intersection_area / focuse_area_area if focuse_area_area > 0 else 0
            if occupancy_rate < 0.3:
                occupancy_rate = 0
            mapper[col][row] = round(occupancy_rate, 2)


    matches , loc = use_final(mapper , boxes)
    
    loc = dict(loc)
    loc = {key + 1 : value for key , value in loc.items()}
    loc = dict(sorted(loc.items()))
    matches = sorted(list(map(lambda x : x+1 , filter(lambda x: x is not None, matches))))
    matches = list(filter(lambda x: x != 0, matches))
                          
    for i in matches:
        x1 ,y1 ,x2 ,y2 = f_non_sig_rewnew[i]
        cv2.rectangle(img , (x1 , y1) ,(x2 ,y2) , (255 , 255 , 0) , 1)

    return sorted(matches) , loc , img

def second(boxes , img , number = 0 , Answer = True , ALPHA = 1):
    global f_non_sig , f_non_sig_rewnew
    if Answer:
        f_non_sig = f_non_sig
    else:
        f_non_sig = f_non_sig_rewnew

    f_non_sig_len = len(f_non_sig)  # 박스의 수
    img = img.copy()

    mapper = [[0 for _ in range(f_non_sig_len)] for _ in range(len(boxes))]  


    for i in f_non_sig_rewnew.values():
        x1 , y1  ,x2 ,y2 = i
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    for i in boxes:
        x1, y1, x2, y2 = i
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    for row, cordidate in enumerate(f_non_sig.values()):  
        for col, people in enumerate(boxes): 
            focuse_area_area = calculate_area(cordidate)
            intersection_area = calculate_intersection_area(people, cordidate)
            occupancy_rate = intersection_area / focuse_area_area if focuse_area_area > 0 else 0
            if occupancy_rate < 0.3:
                occupancy_rate = 0
            mapper[col][row] = occupancy_rate

    Intersection_dis_mapper = distance_consider(mapper , boxes , Answer , ALPHA)

    matches , loc = use_final(Intersection_dis_mapper , boxes)

    loc = dict(loc)
    loc = {key + 1 : value for key , value in loc.items()}
    loc = dict(sorted(loc.items()))
    matches = sorted(list(map(lambda x : x+1 , filter(lambda x: x is not None, matches))))
    matches = list(filter(lambda x: x != 0, matches))

    for i in matches:
        x1 ,y1 ,x2 ,y2 = f_non_sig_rewnew[i]
        cv2.rectangle(img , (x1 , y1) ,(x2 ,y2) , (255 , 255 , 0) , 1)

    return sorted(matches)  , loc , img

def distance_consider(mapper, p_box , flag , ALPHA):
    # print('================= 거리 측정 ========================')
    global f_non_sig 

    if flag == False:
        f_non_sig  = f_non_sig_rewnew
    n_boxes = len(mapper[0])  # 열 방향
    n_people = len(mapper)    # 행 방향

    distance_mapper = [[0 for _ in range(n_boxes)] for _ in range(n_people)]
    A = np.array(mapper)
    B = np.array(distance_mapper)

    assert A.shape == B.shape, f"Shape Not Equal {A.shape} {B.shape}"

    for cols, box in enumerate(f_non_sig.values()):  # 열 방향
        x1, y1, x2, y2 = box
        box_cx, box_cy = (x1 + x2) // 2, y1
        for rows, people in enumerate(p_box):  # 행 방향
            px1, py1, px2, py2 = people
            p_cx, p_cy = (px1 + px2) // 2, py1
            distance = np.sqrt((box_cx - p_cx) ** 2 + (box_cy - p_cy) ** 2)
            distance_mapper[rows][cols] = distance
    for idx , i in enumerate(distance_mapper):
        max_distance = np.max(i)
        i_list = i
        i_list = list(map(lambda x : x / max_distance if x!=0 else 0 , i_list))
        distance_mapper[idx] = i_list

    # print("================= 스케일 =====================")
    # print(np.array(distance_mapper))
    # print("=============================================")
    Alpha = ALPHA  # 교집합 가중치
    Epsilon = 1e-6
    mapper = np.array(mapper)
    distance_mapper = np.array(distance_mapper)

    assert mapper.shape == distance_mapper.shape, f"{mapper.shape} {distance_mapper.shape}"

    Intersection_Distance_mapper = np.zeros_like(mapper)

    Intersection_Distance_mapper = mapper * Alpha / (distance_mapper + Epsilon)
    
    return Intersection_Distance_mapper
    # Intersection_Distance_mapper = np.where(
    #     mapper != 0,
    #     mapper * Alpha / (distance_mapper + Epsilon),
    #     0
    # )

    return Intersection_Distance_mapper

if __name__ == '__main__':
    img = np.zeros((600,600),dtype=np.uint8)
    boxes = [[159, 240, 255, 427], [341, 232, 434, 399], [222, 175, 283, 283], [332, 80, 346, 113], [339, 174, 388, 269], [339, 212, 395, 318]]
    # Q, T = first(boxes , img , 8)
    # print(Q)
    # print(T)

    
    first(boxes , img , 8 )
    second(boxes , img , 8 , FLAG=True)
    
"""
5 번째 좌석 착석 ...
7 번째 좌석 착석 ...
9 번째 좌석 착석 ...
11 번째 좌석 착석 ...
13 번째 좌석 착석 ...
"""