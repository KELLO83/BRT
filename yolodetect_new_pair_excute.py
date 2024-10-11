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

class YOLODetector:
    def __init__(self, f1 , number ,model_path='yolov8x.pt'):
        self.number = number
        self.distortion_list = f1
        self.model = YOLO(model_path).to("cuda")
        self.empty_count = 0
        self.false_alarm = 0
        self.not_same_source = 0

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

    def run(self):
        for index , i in tqdm(enumerate(self.distortion_list),total=len(self.distortion_list)):
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
                #undisort_image = cv2.resize(undisort_image, (640, 640), interpolation=cv2.INTER_LANCZOS4)



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
            answer , answer_location , ans_img =  box_module.second(nmx_boxes , undisort_image.copy() , Answer= True , ALPHA = 1)


            compare_match , compare_location  , compare_img = box_module.second(nmx_boxes , undisort_image.copy() , Answer= False, ALPHA = 0.5)

            answer_location = dict(sorted(answer_location.items()))
            compare_location = dict(sorted(compare_location.items()))
            stop_point = False
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
                print("출처다름 : ",self.not_same_source)
                print("오경보 : ",self.false_alarm)
                print("정답 \n" , answer_location)
                print(compare_location)

            # cv2.namedWindow("ans" , cv2.WINDOW_NORMAL)
            # cv2.namedWindow("compare", cv2.WINDOW_NORMAL)
            # cv2.imshow("ans" , ans_img)
            # cv2.imshow("compare",compare_img)
            # cv2.waitKey(0)

        print("빈이미지" ,self.empty_count)
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

    c = YOLODetector( distort_images , number)
    c.run()