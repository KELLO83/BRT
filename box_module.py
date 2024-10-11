import os
import cv2
import numpy as np
from natsort import  natsort
import os
import collections
import collections

def gale_shapley_2(matches, mapper, loc):
    """Assign unique boxes to coordinates, ensuring each box number is unique.

    For duplicate box numbers in 'loc', reassign new unique box numbers
    to duplicates while keeping the original numbers for non-duplicates.
    """
    import collections

    N = len(matches)  # Number of people
    M = len(loc)      # Number of coordinates (some may have duplicate box numbers)

    # Step 1: Identify duplicate box numbers
    box_counts = collections.Counter(box_number for box_number, _ in loc)
    duplicates = {box_number for box_number, count in box_counts.items() if count > 1}

    # Step 2: Assign new unique box numbers to duplicates (except the first occurrence)
    used_box_numbers = set(box_counts.keys())
    new_loc = []
    duplicate_counts = {}
    for idx, (box_number, coords) in enumerate(loc):
        if box_number in duplicates:
            count = duplicate_counts.get(box_number, 0)
            if count == 0:
                # First occurrence, keep original box number
                new_loc.append([box_number, coords])
            else:
                # Subsequent occurrences, assign new box numbers
                new_box_number = box_number
                while new_box_number in used_box_numbers:
                    new_box_number += 1  # Increment to find an unused box number
                used_box_numbers.add(new_box_number)
                new_loc.append([new_box_number, coords])
                # Update 'mapper' to reflect the new box number
                for i in range(N):
                    mapper[i].append(mapper[i][idx])  # Copy the scores for the new box
            duplicate_counts[box_number] = count + 1
        else:
            # Non-duplicate, keep as is
            new_loc.append([box_number, coords])

    # Update M and matches to reflect the new number of boxes
    M = len(new_loc)
    # Extend the mapper if new boxes were added
    for i in range(N):
        if len(mapper[i]) < M:
            mapper[i].extend([0] * (M - len(mapper[i])))
    # Adjust 'matches' if necessary (assuming matches correspond to indices in 'loc')
    new_matches = matches.copy()

    # Now apply Gale-Shapley algorithm to assign boxes to people based on 'mapper' scores
    # Initialize assignments: box_index -> person_index
    assignments = {}
    # Initialize people who need to be assigned
    people_to_assign = set(range(N))
    # Keep track of boxes each person has tried
    tried_boxes = [set() for _ in range(N)]
    # For each person, get their preference list (sorted boxes by score)
    preference_lists = []
    for i in range(N):
        scores = mapper[i]
        # Get list of box indices sorted by score in descending order
        prefs = sorted(range(M), key=lambda j: -scores[j])
        preference_lists.append(prefs)

    # Initial assignments with conflict detection
    counter = collections.Counter(new_matches)
    conflicts = [key for key, count in counter.items() if count > 1]

    for i in range(N):
        box = new_matches[i]
        if box is not None:
            tried_boxes[i].add(box)
            if box in assignments:
                # Conflict detected
                people_to_assign.add(i)
            else:
                assignments[box] = i
                if i in people_to_assign:
                    people_to_assign.remove(i)

    # Resolve conflicts using Gale-Shapley algorithm
    while people_to_assign:
        person = people_to_assign.pop()
        prefs = preference_lists[person]
        assigned = False
        for box in prefs:
            if box in tried_boxes[person]:
                continue  # Already tried this box
            tried_boxes[person].add(box)
            if box not in assignments:
                # Box is unassigned
                assignments[box] = person
                assigned = True
                break
            else:
                # Box is assigned to someone else
                current_person = assignments[box]
                # Compare scores
                person_score = mapper[person][box]
                current_person_score = mapper[current_person][box]
                if person_score > current_person_score:
                    # Reassign box to this person
                    assignments[box] = person
                    people_to_assign.add(current_person)
                    assigned = True
                    break
                else:
                    continue  # Current person keeps the box
        if not assigned:
            # Person could not be assigned to any box
            print(f"Person {person} could not be assigned to any box.")
            # Optionally handle unassigned person here

    # Create the final matches
    final_matches = [None] * N
    for box, person in assignments.items():
        final_matches[person] = box

    # Create match_coords
    match_coords = []
    for person in range(N):
        box_index = final_matches[person]
        if box_index is not None and box_index < len(new_loc):
            box_number, coords = new_loc[box_index]
            match_coords.append([box_number, coords])
        else:
            # Handle unassigned person
            match_coords.append([None, None])

    # 'final_loc' is the updated 'loc' with unique box numbers
    final_loc = new_loc

    return final_matches, match_coords, final_loc

def gale_shapley(matches , mapper) :
    """ step2 개선버전 """
    """Remove duplicate assignments and reassign based on scores."""
    N = len(matches)
    M = len(mapper[0])
    
    # Initialize assignments: box -> person
    assignments = {}
    # Initialize people who need to be assigned
    people_to_assign = set(range(N))
    # Keep track of boxes each person has tried
    tried_boxes = [set() for _ in range(N)]
    # For each person, get their preference list (sorted boxes by score)
    preference_lists = []
    for i in range(N):
        scores = mapper[i]
        # Get list of boxes sorted by score in descending order
        prefs = sorted(range(M), key=lambda j: -scores[j])
        preference_lists.append(prefs)
    
    # Initial assignments with conflict detection
    counter = collections.Counter(matches)
    conflicts = [key for key, count in counter.items() if count > 1]
    
    # Assign boxes, resolving conflicts
    for i in range(N):
        box = matches[i]
        tried_boxes[i].add(box)
        if box in assignments:
            # Conflict detected
            people_to_assign.add(i)
        else:
            assignments[box] = i
            if i in people_to_assign:
                people_to_assign.remove(i)
    
    # Resolve conflicts
    while people_to_assign:
        person = people_to_assign.pop()
        prefs = preference_lists[person]
        assigned = False
        for box in prefs:
            if box in tried_boxes[person]:
                continue  # Already tried this box
            tried_boxes[person].add(box)
            if box not in assignments:
                # Box is unassigned
                assignments[box] = person
                assigned = True
                break
            else:
                # Box is assigned to someone else
                current_person = assignments[box]
                # Compare scores
                person_score = mapper[person][box]
                current_person_score = mapper[current_person][box]
                if person_score > current_person_score:
                    # Reassign box to this person
                    assignments[box] = person
                    people_to_assign.add(current_person)
                    assigned = True
                    break  # Break to reprocess the current person
                else:
                    continue  # Current person keeps the box
        if not assigned:
            # Person could not be assigned to any box
            print(f"Person {person} could not be assigned to any box.")
            # Optionally handle unassigned person here
            # For example, assign to a default box or leave unassigned
    
    # Create the final matches
    final_matches = [None] * N
    for box, person in assignments.items():
        final_matches[person] = box

    return final_matches

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




# scale_factor = 640 / 600
# f_non_sig = {key : scale_coordinates(value, scale_factor) for key, value in f_non_sig.items()}
# f_non_sig_rewnew = {key : scale_coordinates(value, scale_factor) for key, value in f_non_sig_rewnew.items()}


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

def step1(mapper , p_boxes) :
    """ 박스를 사람과 매칭하는 과정 (중복허용)"""
    n_boxes = len(mapper[0])  # 박스의 개수 # 열
    n_people = len(mapper)  # 사람의 개수 # 행
    
    matches = [-1] * n_people  # 각 인덱스마다 사람이 매칭된박스 idx번호
    match_coords = []

    for rows in range(n_people): # 행방향 스캔
        best_score = -1
        best_index = None
        for cols in range(n_boxes): # 열방향스캔 박스스캔
            score = mapper[rows][cols]

            if score > best_score: # 어떤 사람을 기준으로 비교대상 박스의값이 크다면 best점수 갱신
                best_score = score
                best_index = cols
        
        # best점수인 인덱스를 matche에 매칭된 박스를 부여
        if best_score >= 0.3:
            matches[rows] = best_index
            match_coords.append([best_index , p_boxes[rows]])

        else:
            matches[rows] = None
    
    if -1 in matches:
        raise AssertionError
    
    return matches  , match_coords

def step2(matches, mapper):
    """ 중복을 제거하는 과정 """
    final_matches = [-1] * len(matches)
    assigned_boxes = set()
    
    for person, box in enumerate(matches):
        if box not in assigned_boxes:
            final_matches[person] = box
            assigned_boxes.add(box)
        else:
            for new_box in range(len(mapper[person])):
                if new_box not in assigned_boxes:
                    final_matches[person] = new_box
                    assigned_boxes.add(new_box)
                    break
                    
    return final_matches

def step2_preper(matches, mapper):

    """중복을 제거하고, 선호도를 고려하여 재배정하는 과정"""
    final_matches = [-1] * len(matches)
    assigned_boxes = set()

    for person, box in enumerate(matches):
        if box not in assigned_boxes:
            final_matches[person] = box
            assigned_boxes.add(box)
        else:
            # 중복된 경우: 현재 박스를 점유하고 있는 사람을 찾음
            current_holder = final_matches.index(box)
            if mapper[person][box] > mapper[current_holder][box]:
                # 새로운 사람이 더 선호도가 높다면, 박스를 교체
                final_matches[person] = box
                final_matches[current_holder] = -1  # 기존 사람은 재배정 필요
            else:
                # 기존 사람이 박스를 유지, 새로운 사람은 재배정 필요
                final_matches[person] = -1

            # 점수가 낮은 사람은 다른 박스를 탐색하여 배정
            for new_box in sorted(range(len(mapper[person])), key=lambda x: -mapper[person][x]):
                if new_box not in assigned_boxes:
                    final_matches[person] = new_box
                    assigned_boxes.add(new_box)
                    break

            # 기존 사람이 밀려난 경우, 그 사람도 다시 박스를 찾아야 함
            if final_matches[current_holder] == -1:
                for new_box in sorted(range(len(mapper[current_holder])), key=lambda x: -mapper[current_holder][x]):
                    if new_box not in assigned_boxes:
                        final_matches[current_holder] = new_box
                        assigned_boxes.add(new_box)
                        break

    return final_matches 

def step2_loc(matches, mapper, loc):
    """Assign unique boxes to coordinates, ensuring each box number is unique.

    For duplicate box numbers in 'loc', reassign new unique box numbers
    to duplicates while keeping the original numbers for non-duplicates.
    """
    import collections

    N = len(matches)  # Number of people
    M = len(loc)      # Number of coordinates (some may have duplicate box numbers)

    # Step 1: Identify duplicate box numbers
    box_counts = collections.Counter(box_number for box_number, _ in loc)
    duplicates = {box_number for box_number, count in box_counts.items() if count > 1}

    # Step 2: Assign new unique box numbers to duplicates (except the first occurrence)
    used_box_numbers = set(box_counts.keys())
    new_loc = []
    duplicate_counts = {}
    for idx, (box_number, coords) in enumerate(loc):
        if box_number in duplicates:
            count = duplicate_counts.get(box_number, 0)
            if count == 0:
                # First occurrence, keep original box number
                new_loc.append([box_number, coords])
            else:
                # Subsequent occurrences, assign new box numbers
                new_box_number = box_number
                while new_box_number in used_box_numbers:
                    new_box_number += 1  # Increment to find an unused box number
                used_box_numbers.add(new_box_number)
                new_loc.append([new_box_number, coords])
                # Update 'mapper' to reflect the new box number
                for i in range(N):
                    mapper[i].append(mapper[i][idx])  # Copy the scores for the new box
            duplicate_counts[box_number] = count + 1
        else:
            # Non-duplicate, keep as is
            new_loc.append([box_number, coords])

    # Update M to reflect the new number of boxes
    M = len(new_loc)
    # Extend the mapper if new boxes were added
    for i in range(N):
        if len(mapper[i]) < M:
            mapper[i].extend([0] * (M - len(mapper[i])))
    # Adjust 'matches' if necessary (assuming matches correspond to indices in 'loc')
    new_matches = matches.copy()

    # Step 3: Resolve conflicts to ensure each box is assigned uniquely
    final_matches = [-1] * N
    assigned_boxes = set()
    for person, box in enumerate(new_matches):
        if box not in assigned_boxes:
            final_matches[person] = box
            assigned_boxes.add(box)
        else:
            # Find a new box for this person that is not yet assigned
            for new_box in range(M):
                if new_box not in assigned_boxes:
                    final_matches[person] = new_box
                    assigned_boxes.add(new_box)
                    break

    # Create match_coords
    match_coords = []
    for person in range(N):
        box_index = final_matches[person]
        if box_index is not None and box_index < len(new_loc):
            box_number, coords = new_loc[box_index]
            match_coords.append([box_number, coords])
        else:
            # Handle unassigned person
            match_coords.append([None, None])

    # 'final_loc' is the updated 'loc' with unique box numbers
    final_loc = new_loc

    return final_matches, match_coords, final_loc

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