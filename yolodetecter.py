import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm.auto import  tqdm
from natsort import  natsort
class YOLODetector:
    def __init__(self, input_directory, output_directory, model_path='yolov9m.pt'):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.model = YOLO(model_path)
        self.create_directories()

    def create_directories(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def save_image(self, image, file_name , camera_folder):
        filename = os.path.join(self.output_directory , camera_folder, file_name)
        os.makedirs(os.path.dirname(filename) , exist_ok=True)
        cv2.imwrite(filename, image)

    def apply_nms(self, boxes, scores, iou_threshold=0.4):
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)
        if isinstance(indices, list) and len(indices) > 0:
            return [boxes[i[0]] for i in indices]
        elif isinstance(indices, np.ndarray) and indices.size > 0:
            return [boxes[i] for i in indices.flatten()]
        else:
            return []

    def process_images(self):
        for root, dirs, files in os.walk(self.input_directory):
            files = natsort.natsorted(files)
            for file in tqdm(files):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    print(img.shape)
                    cv2.namedWindow("t",cv2.WINDOW_NORMAL)
                    cv2.imshow("t",img)
                    cv2.waitKey(0)
                    results = self.model(img , verbose=False)

                    boxes = []
                    scores = []

                    # 사람만 필터링
                    for result in results:
                        for box in result.boxes:
                            if int(box.cls) == 0:  # 사람의 클래스 ID (YOLO 모델에 따라 다를 수 있음, 여기서는 0으로 가정)
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                score = box.conf
                                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                                scores.append(float(score))

                    # NMS 적용
                    nms_boxes = self.apply_nms(boxes, scores)

                    for box in nms_boxes:
                        x1, y1, w, h = box
                        center_x, center_y = x1 + w // 2, y1 + h // 2

                        # 중심점을 이미지에 표시
                        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
                        # 좌표를 이미지에 작게 표시
                        cv2.putText(img, f'({center_x}, {center_y})', 
                                    (center_x, center_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.3, 
                                    (255, 255, 255), 
                                    1)

                    # 이미지를 저장
                    self.save_image(img, file , img_path.split('/')[-2])
                    print(f"Processed {file} in {root}")

if __name__ == "__main__":
    input_dir_8x = '8/camera6_images'
    output_dir_8x = 'Compare'

    detector_8x = YOLODetector(input_dir_8x, output_dir_8x, model_path='yolov8x.pt')
    detector_8x.process_images()

    print("Object detection completed for all images.")