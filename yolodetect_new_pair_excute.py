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


            answer_location = dict(sorted(answer_location.items()))
            compare_location = dict(sorted(compare_location.items()))
            stop_point = False

            if image_name in self.image_error_list:
                if image_name != 'cv2_diff_test/front/4.1/image_0237.jpg':
                    answer = [5 , 7 , 9 , 12 ]
                else:
                    answer = [5 , 7 , 9 , 12 , 13]
                if sorted(answer) != sorted(compare_location.keys()):
                    self.false_alarm += 1
                    stop_point = True
            else:
                if len(answer) == len(compare_match): # 길이가 같은경우에 대해여
                    for (answer_key , answer_value) , (key , value) in zip(answer_location.items() , compare_location.items()):
                        if answer_key != key:
                            self.false_alarm += 1
                            stop_point = True

                        else: # 좌석 번호는 같은데 출처가 다를떄
                            if answer_value != value:
                                self.not_same_source += 1
                                stop_point = True

                else: # 길이가 다르다면 오경보
                    self.false_alarm += 1
                    stop_point = True
            
            if stop_point:
                print("image name : ",image_name)
                print("출처다름 : ",self.not_same_source)
                print("오경보 : ",self.false_alarm)
                print("정답 \n" , answer_location)
                print(compare_location)

        cv2.namedWindow("Answer" , cv2.WINDOW_NORMAL)
        cv2.namedWindow("compare", cv2.WINDOW_NORMAL)
        cv2.imshow("Answer" , ans_img)
        cv2.imshow("compare",compare_img)
        cv2.waitKey(0)

            # if len(answer_location) == len(compare_location):
            #     for (i,v) , (q,r) in zip(answer_location.items() , compare_location.items()):
            #         if i!=q or v != r:
            #             self.DEBUG +=1
            #             break
            # else:
            #     self.DEBUG += 1




        return self.false_alarm , self.not_same_source


if __name__ == "__main__":

    from glob import glob
    from natsort import  natsorted

    name = 'cv2_diff_test/raw_seat/11'

    number = name.split('/')[-1]
    distort_images = natsorted(glob(os.path.join(f'{name}','*.jpg')))

    distort_images = natsorted(glob(os.path.join(f"cv2_diff_test/front",'**',"*.jpg"),recursive=True))
    if not distort_images:
        raise FileExistsError
    
    #distort_images = natsorted(glob(os.path.join("cv2_diff_test/pro" , "*.jpg")))


    errors = []
    import gc

    range_ = np.linspace(0.1, 1 ,100).tolist()
    # range_ = list(reversed(range_))
    # range_ = list(map (lambda x: round(x,2) , range_))

    #range_ = [0.2 , 0.4 , 0.6 , 0.8 , 1]
    range_ = [1]
    print(range_)
    input("========= continue Press Any key ===============")
    for i in range_:
        c = YOLODetector(distort_images, number, alpha=i)
        f, s = c.run()

        with open('new.txt', 'a+') as file:
            file.write(f"step : {i:.1e} {f} {s} \n")

        error = f + s
        errors.append(error)
        del c
        gc.collect()


    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 12))
    plt.plot(range_ , errors, label='Error over alpha' , color = 'red')
    plt.xlabel('Alpha')
    plt.ylabel('Error')
    plt.title('Error vs Alpha')
    plt.legend()
    plt.show()
