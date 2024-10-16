import os
import cv2
import numpy as np
from natsort import  natsort
import os
import collections
import collections


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

def calculate_intersection_area(box1 , box2 , img):

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0 , img
    
    return (x_right - x_left) * (y_bottom - y_top) , img

def calculate_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


sit_count = {key : 0 for key in range(1, 14)}
sit_count_old = {key : 0 for key in range(1,14)}

f_non_sig = { # 정답지로 사용하고있는 박스
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


# """ f_non_sig_renew를 사람 머리까지 ..."""
# for key in f_non_sig:
#     # f_non_sig에서 y1 값 추출 (두 번째 요소)
#     y1_value = f_non_sig[key][1]
#     # f_non_sig_rewnew에서 해당 키의 y1 값을 f_non_sig의 y1 값으로 수정
#     f_non_sig_rewnew[key][1] = y1_value

# # 결과 출력
# f_non_sig_rewnew = f_non_sig_rewnew


""" 
머리 + 상반
"""
head_top = {
    1: [410, 405, 519, 560],
    2: [520, 411, 639, 560],
    3: [373, 293, 452, 410],
    4: [453, 289, 526, 410],
    5: [364, 235, 420, 336],
    6: [421, 233, 475, 336],
    7: [352, 195, 394, 280],
    8: [395, 203, 443, 280],
    9: [170, 218, 232, 335],
    10: [220, 190, 265, 276],
    11: [346, 166, 383, 244],
    12: [384, 166, 417, 244],
    13: [237, 174, 277, 244],
}

# # 의자 상반신을 맞춤
# for key , index in f_non_sig_rewnew.items():
#     x1 , y1 ,x2 , y2 = index
#     if key <=8:
#         X1 , Y1 ,X2 , Y2 = head_top[key]
#         if key % 2 !=0:
#             f_non_sig_rewnew[key] =  x1 , y1 , x2 , Y2

#         else:
#             f_non_sig_rewnew[key] = x1 , y1 , x2 ,Y2

#     if key in [12,13]:
#         X1 , Y1 , X2 , Y2 =  head_top[key]
#         if key % 2 ==0:
#             f_non_sig_rewnew[key] =  x1 , y1 , x2 , Y2

#         else:
#             f_non_sig_rewnew[key] = x1 , y1 , x2 , Y2


for key , index in f_non_sig_rewnew.items():
    x1 , y1 ,x2 , y2 = index
    # gap = abs(y2 - y1)
    # gap = gap / 10
    # y1 = int(y1 - gap * 2)
    if key in [1,2]:
        f_non_sig_rewnew[key] = x1 , y1-5  , x2 , y2
    
    if key in [3,4]:
        f_non_sig_rewnew[key] = x1 , y1-10 ,x2 , y2

    if key in [5,6,9]:
        f_non_sig_rewnew[key] = x1 , y1-15  , x2 , y2
        
    if key in [7,8,10]:
        f_non_sig_rewnew[key] = x1 , y1-20, x2 , y2

    if key in [11,12,13]:
        f_non_sig_rewnew[key] = x1 , y1-25 , x2 , y2






# scale_factor = 640 / 600
# f_non_sig = {key: [int(coord * scale_factor) for coord in value] for key, value in f_non_sig.items()}
# f_non_sig_rewnew = {key : [int(coord * scale_factor)for coord in value] for key, value in f_non_sig_rewnew.items()}

def gale(mapper, p_boxes):
    """
    Matching boxes to people using the Gale-Shapley algorithm (stable matching).
    The preference scores are stored in 'mapper'.
    """
    n_people = len(mapper)  # Number of people
    n_boxes = len(mapper[0])  # Number of boxes

    # Generate preference lists for people based on scores
    person_prefs = []
    for person_idx in range(n_people):
        scores = mapper[person_idx]
        prefs = [(box_idx, scores[box_idx]) for box_idx in range(n_boxes) if scores[box_idx] >= 0.3]
        # Sort preferences by decreasing score
        prefs.sort(key=lambda x: x[1], reverse=True)
        # Keep only the box indices
        prefs = [box_idx for box_idx, score in prefs]
        person_prefs.append(prefs)

    # Generate preference lists for boxes based on scores
    box_prefs = []
    for box_idx in range(n_boxes):
        scores = [mapper[person_idx][box_idx] for person_idx in range(n_people)]
        prefs = [(person_idx, scores[person_idx]) for person_idx in range(n_people) if scores[person_idx] >= 0.3]
        # Sort preferences by decreasing score
        prefs.sort(key=lambda x: x[1], reverse=True)
        # Keep only the person indices
        prefs = [person_idx for person_idx, score in prefs]
        box_prefs.append(prefs)

    # Initialize the matching process
    next_proposal = [0] * n_people  # Next box to propose to for each person
    engaged_boxes = [None] * n_boxes  # Current person engaged to each box
    free_people = [person_idx for person_idx in range(n_people) if person_prefs[person_idx]]

    while free_people:
        person_idx = free_people.pop(0)  # Get a free person

        # Check if person has any boxes left to propose to
        if next_proposal[person_idx] >= len(person_prefs[person_idx]):
            # Person has proposed to all boxes, cannot be matched
            continue

        # Get the next box to propose to
        box_idx = person_prefs[person_idx][next_proposal[person_idx]]
        next_proposal[person_idx] += 1

        # Person proposes to box
        # Box considers the proposal
        current_engaged = engaged_boxes[box_idx]

        # Get box's preference list
        box_pref_list = box_prefs[box_idx]

        # If box is free
        if current_engaged is None:
            # Engage person and box
            engaged_boxes[box_idx] = person_idx
        else:
            # Decide whether to accept new proposer or stay with current
            if person_idx in box_pref_list:
                current_engaged_rank = box_pref_list.index(current_engaged) if current_engaged in box_pref_list else len(box_pref_list)
                new_proposer_rank = box_pref_list.index(person_idx)

                if new_proposer_rank < current_engaged_rank:
                    # Box prefers new proposer
                    # Break engagement with current person
                    free_people.append(current_engaged)
                    # Engage new person
                    engaged_boxes[box_idx] = person_idx
                else:
                    # Box rejects new proposer
                    free_people.append(person_idx)
            else:
                # Box rejects new proposer
                free_people.append(person_idx)

    # Build matches and match_coords
    matches = [None] * n_people  # matches[person_idx] = box_idx
    match_coords = []

    for box_idx, person_idx in enumerate(engaged_boxes):
        if person_idx is not None:
            matches[person_idx] = box_idx
            match_coords.append([box_idx, p_boxes[person_idx]])

    return matches, match_coords

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

def first(boxes, img, Answer = False , ALPHA = 1):
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

def second(boxes , img , number = 0 , Answer = True , ALPHA = 1 ):
    #print(" ================= 정답지 =============================")
    global f_non_sig , f_non_sig_rewnew
    if  Answer:
        BOX_CORD = f_non_sig
    else:
        BOX_CORD = f_non_sig_rewnew


    img = img.copy()

    mapper = [[0 for _ in range(len(BOX_CORD))] for _ in range(len(boxes))]  


    for index , i in enumerate(f_non_sig_rewnew.values()):
        x1 , y1  ,x2 ,y2 = i
        
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # cv2.putText(
        #     img,  # 이미지
        #     f"{index + 1}",  # 텍스트
        #     (x1 + 10, y1 + 20),  # 우측 상단에서 약간의 패딩을 준 위치
        #     cv2.FONT_HERSHEY_DUPLEX,  # 글꼴
        #     0.6,  # 글자 크기
        #     (0, 255, 0),  # 초록색 (BGR 형식으로 설정)
        #     1  # 글자 두께
        # )
    
    #TMP = [[324,162,381,244],[324,162,377,307],[236,183,285,295],[232,175,275,225]]
    #Q = [[236,183,285,295],[232,175,275,225]]

    # TMP = reversed(TMP)
    # for c , i in enumerate(TMP):
    #     colors = [(0,0,255),(0,255,255)]
    #     x1 , y1 ,x2 , y2 = i
    #     center_x , center_y = (x1 + x2) // 2 , (y1 + y2) // 2
    #     cv2.rectangle(img, (x1, y1), (x2, y2), color= colors[c % len(colors)], thickness=2)
       # cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  
        # cv2.putText(img, f'({center_x}, {center_y})', 
        #             (center_x, center_y), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.3, 
        #             (255, 255, 255), 
        #             1)


    # for idx , i in enumerate(boxes):
    #     x1, y1, x2, y2 = i
    #     center_x , center_y = (x1 + x2) // 2 , (y1 + y2) // 2
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  
        # cv2.putText(img, f'({center_x}, {center_y})', 
        #             (center_x, center_y), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.3, 
        #             (255, 255, 255), 
        #             1)

    for row, cordidate in enumerate(BOX_CORD.values()):  
        for col, people in enumerate(boxes): 
            focuse_area_area = calculate_area(cordidate)
            intersection_area , img = calculate_intersection_area(people, cordidate , img)
            occupancy_rate = intersection_area / focuse_area_area if focuse_area_area > 0 else 0
            if occupancy_rate < 0.3:
                occupancy_rate = 0
            mapper[col][row] = occupancy_rate

    #print("교집합영역 :\n",np.array(mapper , dtype=  np.float64))
    if Answer:
        Intersection_dis_mapper = distance_consider(mapper , boxes , BOX_CORD , ALPHA = 1)
    else:
        Intersection_dis_mapper , img = distance_consider_Weight(mapper , boxes , BOX_CORD , ALPHA , img)
    #print("결과 : \n " , Intersection_dis_mapper)

    matches , loc = use_final(Intersection_dis_mapper , boxes)

    loc = dict(loc)
    loc = {key + 1 : value for key , value in loc.items()}
    loc = dict(sorted(loc.items()))
    matches = sorted(list(map(lambda x : x+1 , filter(lambda x: x is not None, matches))))
    matches = list(filter(lambda x: x != 0, matches))

    for idx , i in enumerate(matches):
        x1 ,y1 ,x2 ,y2 = f_non_sig_rewnew[i]
        cv2.rectangle(img , (x1 , y1) ,(x2 ,y2) , (255 , 255 , 0) , 2)
        
        # x1 , y1 , x2 , y2 = loc[i]
        # center_x , center_y = (x1 + x2) // 2 , (y1 + y2) // 2
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  
        # cv2.putText(img, f'({center_x}, {center_y})', 
        #             (center_x, center_y), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.3, 
        #             (255, 255, 255), 
        #             1)
        #cv2.putText(img , f"{idx}" ,(x1 , y1) , cv2.FONT_HERSHEY_SCRIPT_COMPLEX , 0.3 ,(0,0,255),1)
        # cv2.putText(
        #     img,  # 이미지
        #     f"{idx+1}",  # 텍스트
        #     (x2 - 20, y1 + 20),  # 우측 상단에서 약간의 패딩을 준 위치
        #     cv2.FONT_HERSHEY_COMPLEX,  # 글꼴
        #     0.6,  # 글자 크기
        #     (0, 0 , 255),  # 초록색 (BGR 형식으로 설정)
        #     1  # 글자 두께
        # )
    

    return sorted(matches)  , loc , img

def distance_consider(mapper, p_box , BOX_CORD , ALPHA):
    # print('================= 거리 측정 ========================')


    ALPHA = 1


    n_boxes = len(mapper[0])  # 열 방향
    n_people = len(mapper)    # 행 방향

    distance_mapper = [[0 for _ in range(n_boxes)] for _ in range(n_people)]
    A = np.array(mapper)
    B = np.array(distance_mapper)

    assert A.shape == B.shape, f"Shape Not Equal {A.shape} {B.shape}"

    for cols, box in enumerate(BOX_CORD.values()):  # 열 방향
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

    #print("정답 거리 \n",np.array(distance_mapper)) 
    Alpha = ALPHA  # 교집합 가중치
    Epsilon = 1e-6
    mapper = np.array(mapper)
    distance_mapper = np.array(distance_mapper)
    # print("정답 스케일 거리 \n ", distance_mapper)
    # distance_mapper[mapper ==0 ] = 0
    assert mapper.shape == distance_mapper.shape, f"{mapper.shape} {distance_mapper.shape}"

 
    Intersection_Distance_mapper = np.zeros_like(mapper)

    Intersection_Distance_mapper = (mapper ** ALPHA) / ((distance_mapper ** ALPHA) + Epsilon)

    return Intersection_Distance_mapper

def distance_consider_Weight(mapper, p_box , BOX_CORD , ALPHA , img):
    #print('================= 거리 측정 ========================')
    #print("입력 :\n" , np.array(mapper , dtype=np.float64))


    n_boxes = len(mapper[0])  # 열 방향
    n_people = len(mapper)    # 행 방향

    distance_mapper = np.array([[0 for _ in range(n_boxes)] for _ in range(n_people)] , dtype=np.float64)
    mapper = np.array(mapper)

    assert mapper.shape == distance_mapper.shape, f"Shape Not Equal {mapper.shape} {distance_mapper.shape}"

    for cols, box in enumerate(BOX_CORD.values()):  # 열 방향
        x1, y1, x2, _ = box
        box_cx, box_cy = (x1 + x2) // 2, y1
        for rows, people in enumerate(p_box):  # 행 방향
            px1, py1, px2, _ = people
            p_cx, p_cy = (px1 + px2) // 2, py1
            distance = np.sqrt((box_cx - p_cx) ** 2 + (box_cy - p_cy) ** 2)
            distance_mapper[rows][cols] = float(distance)

            #cv2.circle(img , (box_cx , box_cy) , 1 ,( 255,255,0), -1)
            #cv2.circle(img , (p_cx , p_cy) , 1 ,(0,0,255 ) , -1)
   # print(distance_mapper , end='\n\n')
    for idx , i in enumerate(distance_mapper):
        max_distance = np.max(i)
        i_list = i
        i_list = list(map(lambda x : x / max_distance if x!=0 else 0 , i_list))
        distance_mapper[idx] = np.array(i_list , dtype=np.float64)

   # print(distance_mapper)
    #print('==================================================================')
    Alpha = ALPHA 
    Epsilon = 1e-6
    mapper = np.array(mapper)

    distance_mapper = np.where(mapper == 0 , 0 , distance_mapper)
    distance_mapper[mapper == 0 ] = 0
    distance_mapper[distance_mapper == 0] = 1e-12
    distance_mapper = np.array(distance_mapper)


    assert mapper.shape == distance_mapper.shape, f"{mapper.shape} {distance_mapper.shape}"

    Intersection_Distance_mapper = np.zeros_like(mapper)

    Intersection_Distance_mapper = np.where(
        (mapper == 0) | (distance_mapper == 0), 
        0,  
        mapper ** Alpha / ((distance_mapper ** (1 - Alpha))) 
    )

    return Intersection_Distance_mapper , img


if __name__ == '__main__':
    img = np.zeros((600,600),dtype=np.uint8)
    cord = [[159, 240, 255, 427], [341, 232, 434, 399], [222, 175, 283, 283], [332, 80, 346, 113], [339, 174, 388, 269], [339, 212, 395, 318]]
    boxes = [[0,0,0,0,0,0,0,0,16.733,4.5231,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,9.3752],
    [0,0,0,0,0,0,0,0,0,0,0,13.172,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,11.931,0,0,29.844]]
    
    matches , cords = gale(boxes ,cord )
    print(np.array(boxes , dtype=np.int64))
    print('==========================')
    print(matches)
    print(cords)

    
"""
5 번째 좌석 착석 ...
7 번째 좌석 착석 ...
9 번째 좌석 착석 ...
11 번째 좌석 착석 ...
13 번째 좌석 착석 ...
"""