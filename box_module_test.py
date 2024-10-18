import os
import cv2
import numpy as np
from natsort import  natsort
import os
import collections
import collections

__all__ = ['Coordinate' , 'Calc_Class' , 'Mapping_Excution']
class Coordinate:
    __first_answer_location = { # 1량 정답지로 사용하는 박스
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

    __first_sit_fit_location = { # 1량 좌석에 맞추어진 박스
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

    __first_head_top = {# 1량 좌판을제외 ~ 머라
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

    __SQ = {
        1 : [33,329 , 247 , 554],
        2 : [89,268 , 301,440],
        3 : [135,218,295,386]
    }

    @classmethod
    def test(cls):
        return cls.__SQ
    
    @classmethod
    def first_answer(cls):
        """ (1량) 정답에 맞추어진 박스상자"""
        return cls.__first_answer_location
    
    @classmethod
    def first_fit_sit(cls):
        """ (1량) 좌석에 맞추어진 박스 상자 """
        return cls.__first_sit_fit_location
    
    @classmethod
    def first_fit_sit_5_10_15(cls):
        """ 좌석에 맞추어진 박스 상자 + 5_10_15_25"""
        adjusted_coords = {}
        for key, index in cls.__first_sit_fit_location.items():
            x1, y1, x2, y2 = index
            if key in [1, 2]:
                adjusted_coords[key] = (x1, y1 - 5, x2, y2)
            elif key in [3, 4]:
                adjusted_coords[key] = (x1, y1 - 10, x2, y2)
            elif key in [5, 6, 9]:
                adjusted_coords[key] = (x1, y1 - 15, x2, y2)
            elif key in [7, 8, 10]:
                adjusted_coords[key] = (x1, y1 - 20, x2, y2)
            elif key in [11, 12, 13]:
                adjusted_coords[key] = (x1, y1 - 25, x2, y2)
        return adjusted_coords

class Calc_class :
    @staticmethod
    def convert_2d(boxes):
        x1 , y1 , w, h = boxes
        x2 = x1 + w
        y2 = y1 + h

        return [x1 , y1 , x2 , y2]
    @staticmethod
    def calculate_intersection_area(box1 , box2 , img):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0 , img
        return (x_right - x_left) * (y_bottom - y_top) , img
    @staticmethod
    def calculate_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

class Mapping_Excution:

    def gale(self,mapper, p_boxes):
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
    def matched_box(self,mapper, p_boxes): 
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
    def second(self,boxes , img , Answer = True , ALPHA = 1 , Higer_5_10_15_25 = True ):
        C = Coordinate()

        STANDARD = C.first_fit_sit()
        for index , value in STANDARD.items():
            x1 , y1 , x2 , y2 = value
            cv2.rectangle(img , (x1 , y1) , (x2 , y2) , (255,0,0),1)

        if  Answer:
            BOX_CORD = C.first_answer()
        else:
            BOX_CORD = (
                        C.first_fit_sit_5_10_15() if Higer_5_10_15_25
                        else C.first_fit_sit()
                    )
            
        img = img.copy()

        mapper = [[0 for _ in range(len(BOX_CORD))] for _ in range(len(boxes))]  

        for p_idx , i in enumerate(boxes): # YOLO를 통해 탐지된 사람 박스
            x1, y1, x2, y2 = i
            center_x , center_y = (x1 + x2) // 2 , (y1 + y2) // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  
            cv2.putText(img, f'({center_x}, {center_y})', 
                        (center_x, center_y), 
                        cv2.FONT_HERSHEY_DUPLEX, 
                        0.3, 
                        (255, 255, 255), 
                        1)

        for row, cordidate in enumerate(BOX_CORD.values()):  # INTERSECTION 
            for col, people in enumerate(boxes): 
                focuse_area_area = Calc_class.calculate_area(cordidate)
                intersection_area , img = Calc_class.calculate_intersection_area(people, cordidate , img)
                occupancy_rate = intersection_area / focuse_area_area if focuse_area_area > 0 else 0
                mapper[col][row] = occupancy_rate
        
        print("교집합만\n",np.array(mapper , dtype=np.float32))
        if Answer:
            Intersection_dis_mapper = self.distance_consider(mapper , boxes , BOX_CORD , ALPHA = 1)
        else:
            Intersection_dis_mapper , img = self.distance_consider_Weight(mapper , boxes , BOX_CORD , ALPHA , img)
        print('====================================================')
        print("교집합 거리 고려\n",Intersection_dis_mapper)
        Intersection_dis_mapper = self.k_MEANS(Intersection_dis_mapper)
        matches , matches_loc = self.matched_box(Intersection_dis_mapper , boxes)
        matches_loc = dict(matches_loc)
        for index , value in matches_loc.items(): # 기준점에 해당된박스 민트색으로
            index = index + 1
            x1 , y1 ,x2 , y2 = STANDARD[index]
            cv2.rectangle(img , (x1 , y1) ,(x2 , y2) , (255,255,0),2)

        loc = matches_loc
        loc = {key + 1 : value for key , value in loc.items()}
        loc = dict(sorted(loc.items()))
        matches = sorted(list(map(lambda x : x+1 , filter(lambda x: x is not None, matches))))
        matches = list(filter(lambda x: x != 0, matches))

        return sorted(matches)  , loc , img
    def distance_consider(self,mapper, p_box , BOX_CORD , ALPHA):
  
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


        Alpha = ALPHA  # 교집합 가중치
        Epsilon = 1e-6
        mapper = np.array(mapper)
        distance_mapper = np.array(distance_mapper)

        assert mapper.shape == distance_mapper.shape, f"{mapper.shape} {distance_mapper.shape}"

    
        Intersection_Distance_mapper = np.zeros_like(mapper)

        Intersection_Distance_mapper = (mapper ** ALPHA) / ((distance_mapper ** ALPHA) + Epsilon)

        return Intersection_Distance_mapper
    def distance_consider_Weight(self,mapper, p_box , BOX_CORD , ALPHA , img):

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

        for idx , i in enumerate(distance_mapper):
            max_distance = np.max(i)
            i_list = i
            i_list = list(map(lambda x : x / max_distance if x!=0 else 0 , i_list))
            distance_mapper[idx] = np.array(i_list , dtype=np.float64)

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
    def k_MEANS(Self , mapper):
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt

        Noise_remove = np.zeros_like(mapper)
        for idx, i in enumerate(mapper):
            data = i.reshape(-1, 1)  # Reshape to a 2D array with one column
            if np.all(data == 0):
                Noise_remove[idx] = None
                continue
            K = KMeans(n_clusters=2, random_state=42).fit(data)  # Fit KMeans
            labels = K.labels_

            #print(data.flatten())  # Print original data
            #print(labels)  # Print the labels assigned by KMeans

            # Update Noise_remove based on the labels
            for j in range(len(labels)):
                if labels[j] == 1:
                    Noise_remove[idx, j] = data[j]  # Keep original data where label is 1
                else:
                    Noise_remove[idx, j] = None  # Set to 0 where label is 0

        print("노이즈제거",Noise_remove)
        return Noise_remove
if __name__ == '__main__':
    img = np.zeros((600,600),dtype=np.uint8)
    cord = [[159, 240, 255, 427], [341, 232, 434, 399], [222, 175, 283, 283], [332, 80, 346, 113], [339, 174, 388, 269], [339, 212, 395, 318]]
    boxes = [[0,0,0,0,0,0,0,0,16.733,4.5231,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,9.3752],
    [0,0,0,0,0,0,0,0,0,0,0,13.172,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,11.931,0,0,29.844]]
    

    #crd = Coordinate()
    #print(crd.test)
"""
5 번째 좌석 착석 ...
7 번째 좌석 착석 ...
9 번째 좌석 착석 ...
11 번째 좌석 착석 ...
13 번째 좌석 착석 ...
"""
