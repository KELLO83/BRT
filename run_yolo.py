import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

class YOLODetector:
    def __init__(self, dir , model_path="yolov8x.pt" , TEST_MODE = False):
        self.dir__ = dir
        self.model = YOLO(model_path)
        self.t =TEST_MODE

    def apply_nms(self, boxes, scores, iou_threshold=0.4):
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)
        if isinstance(indices, list) and len(indices) > 0:
            return [boxes[i[0]] for i in indices]
        elif isinstance(indices, np.ndarray) and indices.size > 0:
            return [boxes[i] for i in indices.flatten()]
        else:
            return []

    def process_images(self):
        if self.t == True:
            img = cv2.imread(self.dir__)
            img_copy = img.copy()
            results = self.model(img)

            boxes = []
            scores = []

            # 사람만 필터링
            for result in results:
                for box in result.boxes:
                    if (int(box.cls) == 0):  # 사람의 클래스 ID (YOLO 모델에 따라 다를 수 있음, 여기서는 0으로 가정)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        center_x, center_y = int(x1 + (x2 - x1) // 2), int(y1 + (y2 - y1) // 2)
                        score = box.conf
                        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                        scores.append(float(score))
                        
                        cv2.circle(img_copy , (center_x , center_y) , 5, (255,0,0), - 1)
                        cv2.putText(
                                img_copy,
                                f"({center_x}, {center_y})",
                                (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (255, 255, 255),
                                1,
                            )
                        

            # NMS 적용
            nms_boxes = self.apply_nms(boxes, scores)

            for box in nms_boxes:
                x1, y1, w, h = box
                center_x, center_y = x1 + w // 2, y1 + h // 2

                # 중심점을 이미지에 표시
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
                # 좌표를 이미지에 작게 표시
                cv2.putText(
                    img,
                    f"({center_x}, {center_y})",
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )

            cv2.namedWindow("t",cv2.WINDOW_NORMAL)
            cv2.imshow('t',img)
            cv2.waitKey(0)


        else:
            for root, dirs, files in os.walk(self.dir__):
                for file in tqdm(files):
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(root, file)
                        img = cv2.imread(img_path)
                        results = self.model(img)

                        boxes = []
                        scores = []

                        # 사람만 필터링
                        for result in results:
                            for box in result.boxes:
                                if (
                                    int(box.cls) == 0
                                ):  # 사람의 클래스 ID (YOLO 모델에 따라 다를 수 있음, 여기서는 0으로 가정)
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    score = box.conf
                                    boxes.append(
                                        [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                                    )
                                    scores.append(float(score))

                        # NMS 적용
                        nms_boxes = self.apply_nms(boxes, scores)

                        for box in nms_boxes:
                            x1, y1, w, h = box
                            center_x, center_y = x1 + w // 2, y1 + h // 2

                            # 중심점을 이미지에 표시
                            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
                            # 좌표를 이미지에 작게 표시
                            cv2.putText(
                                img,
                                f"({center_x}, {center_y})",
                                (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (255, 255, 255),
                                1,
                            )



path  = '8'
y = YOLODetector(path,TEST_MODE=False)
y.process_images()