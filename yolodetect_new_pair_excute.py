import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm.auto import  tqdm
import os
import fish_map
import box_module
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

logging.info("Yolo Detect run ...")

import time
import functools
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print("ALPHA : ",args[0].alpha)
        print(f"Function '{func.__name__}' started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        print(f"Function '{func.__name__}' ended at {time.strftime('%H:%M:%S', time.localtime(end_time))}")
        print(f"Total execution time: {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper


class YOLODetector:
    def __init__(self, f1 , number , alpha ,model_path='yolo8x.pt'):
        self.number = number
        self.distortion_list = f1
        self.model = YOLO(model_path).to("cuda")
        self.empty_count = 0
        self.false_alarm = 0
        self.not_same_source = 0

        self.alpha = alpha

        self.image_error_list = [
            'cv2_diff_test/front/4.1/image_0229.jpg',
            'cv2_diff_test/front/4.1/image_0237.jpg' , 
        ]

        self.total_people = 0
        self.TP = 0 
        self.TN = 0
        self.FP = 0
        self.FN = 0
        

    def nmx_box_to_cv2_loc(self , boxes):
        x1 , y1 , w, h = boxes
        x2 = x1 + w
        y2 = y1 + h

        return [x1 , y1 , x2 , y2]

    def apply_nms(self, boxes, scores, iou_threshold=0.6):
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)
        if isinstance(indices, list) and len(indices) > 0:
            return [boxes[i[0]] for i in indices]
        elif isinstance(indices, np.ndarray) and indices.size > 0:
            return [boxes[i] for i in indices.flatten()]
        else:
            return []
        
    @timing_decorator
    def run(self):
        for index , i in tqdm(enumerate(self.distortion_list),total=len(self.distortion_list)):
            image_name = i
            distort_image   = cv2.imread(i , cv2.IMREAD_COLOR)

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
                undisort_image = cv2.resize(undisort_image, (640, 640), interpolation=cv2.INTER_LANCZOS4)



            result = self.model(undisort_image , verbose = False , classes = 0)

            boxes = []
            scores = []

            for res in result:
                for box in res.boxes:
                    if int(box.cls) == 0:
                        x1 , y1 , w , h = box.xyxy[0].tolist()
                        score = box.conf
                        boxes.append([int(x1), int(y1), int(w - x1), int(h - y1)])
                        scores.append(float(score.detach().cpu().numpy()))

            nmx_boxes = self.apply_nms(boxes , scores)
            nmx_boxes = list(map(self.nmx_box_to_cv2_loc , nmx_boxes))
            if not nmx_boxes:
                self.empty_count +=1
                continue


            answer , answer_location , ans_img =  box_module.second(nmx_boxes , undisort_image.copy() , Answer= True , ALPHA = 1 )
            compare_match , compare_location  , compare_img = box_module.second(nmx_boxes , undisort_image.copy() , Answer= False, ALPHA = self.alpha)
            self.total_people += len(compare_location.keys())     # 전체명수    


            answer_location = dict(sorted(answer_location.items()))
            compare_location = dict(sorted(compare_location.items()))
            stop_point = False

            # if image_name in self.image_error_list:
            #     if image_name == 'cv2_diff_test/front/4.1/image_0229.jpg':
            #         answer = [5 , 7 , 9 , 12 ]
            #     else:
            #         answer = [5 , 7 , 9 , 12 ,13 ]
            #     if sorted(answer) != sorted(compare_location.keys()):
            #         self.false_alarm += 1
            #         stop_point = True

            # else:
            #     if len(answer_location) == len(compare_location):  # 길이가 같은 경우
            #         for answer_key in answer_location:
            #             if answer_key not in compare_location:  # 키가 다르면 오경보 처리
            #                 self.false_alarm += 1
            #                 stop_point = True
            #                 break
            #             else:  # 키가 같다면 값 비교
            #                 if answer_location[answer_key] != compare_location[answer_key]:
            #                     self.not_same_source += 1
            #                     stop_point = True
            #                     break
            #     else:  # 길이가 다르면 오경보 처리
            #         self.false_alarm += 1
            #         stop_point = True
            

            mapper = [None for i in range(13)] 
            ans_key = answer_location.keys() # 1 ~ 13 { 9 , 10 ,13}
            compare_key = compare_location.keys()  # 1 ~ 13
            ans_key = list(map(lambda x : x- 1 , ans_key))
            compare_key = list(map(lambda x : x- 1 , compare_key))
            for i in range(13):

                if i in ans_key and i in compare_key:
                    mapper[i] = 'TP'

                elif i not in ans_key and i not in compare_key:
                    mapper[i] = 'TN'

                elif i in ans_key and i not in compare_key:
                    mapper[i] = 'FN'

                elif i not in ans_key  and i in compare_key:  #정답에 없는데 있다고 판별
                    mapper[i] = 'FP'



            for i , (compare_key , value) in enumerate(compare_location.items()):  # 9 12 13
                print(mapper)
                compare_key , value = compare_key , value
                try:
                    answer_value = answer_location[compare_key]
                except:
                    mapper[compare_key - 1] = 'FP'
                    continue

                if value != answer_value: # 정답 
                    if compare_key in ans_key:
                        mapper[compare_key - 1] = 'FP'

                    else:
                        mapper[compare_key - 1] = 'FN'


            print(mapper)
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in mapper:
                if i == 'TP':
                    self.TP += 1
                    TP += 1
                elif i == 'TN':
                    self.TN += 1
                    TN += 1
                elif i == 'FN':
                    self.FN += 1
                    FN += 1
                elif i == 'FP':
                    self.FP += 1
                    FP += 1
        
            if TP + FN + FP + TN != 13:
                cv2.namedWindow("Answer" , cv2.WINDOW_NORMAL)
                cv2.namedWindow("compare", cv2.WINDOW_NORMAL)
                cv2.imshow("Answer" , ans_img)
                cv2.imshow("compare",compare_img)
                cv2.waitKey(0)
                raise ValueError
            

            # TP = len(answer_location.keys())
            # TN = 13 - TP
            # FP = 0
            # FN = 0
            # for answer_key , answer_value in answer_location.items():
            #     if answer_key in compare_location.keys():
            #         if answer_value != compare_location[answer_key]:
            #             self.not_same_source += 1
            #             TP = TP - 2
            #             FP = FP + 2

            #     else:
            #         self.false_alarm += 1
            #         TP = TP - 1
            #         TN = TN - 1
            #         FN = FN + 1
            #         FP = FP + 1


            # print(f"TP : {TP} FN : {FN} FP : {FP} TN : {TN} total : {TP + FN + FP + TN}")
            
            # self.TP += TP
            # self.TN += TN
            # self.FP += FP
            # self.FN += FN
        
            # if TP + FN + FP + TN != 13:
            #     cv2.namedWindow("Answer" , cv2.WINDOW_NORMAL)
            #     cv2.namedWindow("compare", cv2.WINDOW_NORMAL)
            #     cv2.imshow("Answer" , ans_img)
            #     cv2.imshow("compare",compare_img)
            #     cv2.waitKey(0)
            #     raise ValueError

            if stop_point:
                print("image name : ",image_name)
                print("출처다름 : ",self.not_same_source)
                print("오경보 : ",self.false_alarm)
                print("정답 \n" , answer_location)
                print(compare_location)

            # cv2.namedWindow("Answer" , cv2.WINDOW_NORMAL)
            # cv2.namedWindow("compare", cv2.WINDOW_NORMAL)
            # cv2.imshow("Answer" , ans_img)
            # cv2.imshow("compare",compare_img)
            # cv2.waitKey(0)




        print("전체 사람 수 : ",self.total_people)
        print("TP : ", self.TP)
        print("FN : " , self.FN)
        print("FP : ",self.FP)
        print("TN : ", self.TN)

        
        
        DATA , Q = self.Performeance_Metrix()
        DATA = list(map (lambda x : round(x,4), DATA))
        #print(DATA)
        RE , PR , ACC, F1 = DATA
        print("Alpha  : ",self.alpha)
        print("Recall : {} Precision : {} Acc : {} F1 : {}".format(RE , PR , ACC , F1))
        return  DATA , Q
    
    def Performeance_Metrix(self):

        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        Recall = TP / (TP + FN)
        Precision = TP / (TP + FP)
        F1_SCORE = 2 * (Precision * Recall) / (Precision + Recall)
        Accuracy = (TP + TN) / (TP + TN + FP + FN)

        data = [Recall , Precision , Accuracy , F1_SCORE]
        Q = [TP , FP , FN , TN]
        return data  , Q 
    
if __name__ == "__main__":

    from glob import glob
    from natsort import  natsorted

    name = 'cv2_diff_test/raw_seat/11'

    number = name.split('/')[-1]
    distort_images = natsorted(glob(os.path.join(f'{name}','*.jpg')))

    distort_images = natsorted(glob(os.path.join(f"cv2_diff_test/front",'**',"*.jpg"),recursive=True))
    if not distort_images:
        raise FileExistsError
    
    #distort_images = natsorted(glob(os.path.join("cv2_diff_test/problem" , "*.jpg")))

    print(len(distort_images))

    errors = []
    import gc

    range_ = np.linspace(0, 1 ,5)
    # range_ = list(reversed(range_))
    # range_ = list(map (lambda x: round(x,2) , range_))

    #range_ = [ 0.6 , 0.8 , 1]
    #range_ = [1]
    print(range_)
    input("========= continue Press Any key ===============")
    for i in range_:
        c = YOLODetector(distort_images, number, alpha=i)
        DATA , Q = c.run()
        Recall , Precison , ACC , F1 = DATA
        TP , FP , FN ,TN = Q
        with open('Q1.txt', 'a+') as file:
            file.write(f"ALPHA\tTP\tFP\tFN\tTN\tstep\tRecall\tPrecision\tAcc\tF1\n")
            file.write(f"{i}\t{TP}\t{FP}\t{FN}\t{TN}\t{i+1}\t{Recall:.2f}\t{Precison:.2f}\t{ACC:.2f}\t{F1:.2f}\n")
            del c
            gc.collect()


    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 12))
    # plt.plot(range_ , errors, label='Error over alpha' , color = 'red')
    # plt.xlabel('Alpha')
    # plt.ylabel('Error')
    # plt.title('Error vs Alpha')
    # plt.legend()
    # plt.show()
