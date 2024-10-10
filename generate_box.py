import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import natsort
from glob import glob
import shutil

os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.info=false"
class YOLODetector:
    def __init__(self, input_directory, output_directory, model_path, boxes):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.model = YOLO(model_path , verbose=False)  # verbose=False is not necessary here
        self.boxes = boxes
        self.create_directories()

    def create_directories(self):
        box_directories = [f'box{i + 1}' for i in range(len(self.boxes))]
        for box_dir in box_directories:
            full_path = os.path.join(self.output_directory, box_dir)
            if not os.path.exists(full_path):
                os.makedirs(full_path)

    def save_image(self, image, directory, file_name):
        full_directory = os.path.join(self.output_directory, directory)
        if not os.path.exists(full_directory):
            os.makedirs(full_directory)
        filename = os.path.join(full_directory, os.path.basename(file_name))
        cv2.imwrite(filename, image)

    def apply_nms(self, boxes, scores, iou_threshold=0.4):
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)
        if isinstance(indices, list) and len(indices) > 0:
            return [boxes[i[0]] for i in indices]
        elif isinstance(indices, np.ndarray) and indices.size > 0:
            return [boxes[i] for i in indices.flatten()]
        else:
            return []

    def is_point_in_box(self, point, box):
        x, y = point
        (x1, y1), (x2, y2) = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def process_images(self):
        for file in tqdm(natsort.natsorted(glob(os.path.join(self.input_directory, '*.jpg')))):
            img = cv2.imread(file)
            results = self.model(img , verbose=False)  # YOLOv8 expects an image in BGR format

            boxes_data = []
            scores = []

            for result in results:  # Each 'result' is a Results object
                for box in result.boxes:  # result.boxes contains the detected boxes
                    if int(box.cls) == 0:  # class 0 typically represents 'person'
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        score = box.conf
                        
                        boxes_data.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                        scores.append(float(score))

            nms_boxes = self.apply_nms(boxes_data, scores)

            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 0), (0, 128, 128)]
            for i, box_coords in enumerate(self.boxes):
                (a, b), (c, d) = box_coords
                cv2.rectangle(img, (a, b), (c, d), colors[i % len(colors)], 2)

            for box in nms_boxes:
                x1, y1, w, h = box
                center_x, center_y = x1 + w // 2, y1 + h // 2
                        
                if 0 <= center_x < img.shape[1] and 0 <= center_y < img.shape[0]:
                    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(img, f'({center_x}, {center_y})',
                                (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (255, 255, 255),
                                1)
                    frame_id = os.path.splitext(os.path.basename(file))[0]
                    for i, box_coords in enumerate(self.boxes):
                        box_dir = f"box{i + 1}"
                        if self.is_point_in_box((center_x, center_y), box_coords):
                            save_dir = os.path.join(self.output_directory, box_dir)
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"{frame_id}.jpg")
                            cv2.imwrite(save_path, img)
                            print(f"Saved image {file} to {box_dir}")
                            break
                        
def show_coordinates(event , x , y , flags , param):
    
    global image , win_name
    if event == cv2.EVENT_MOUSEMOVE:
        image_copy = image.copy()
        print(f"{x} , {y}")
        cv2.putText(image_copy ,f"{x} , {y}" , (x+10 , y+10) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 255) , 1)
        #cv2.resizeWindow(win_name , 500 , 500)
        cv2.imshow(win_name,image_copy)
    
if __name__ == "__main__":
    
    
    
    number = 33
    input_dir_8x = f'group/raw/{number}'
    output_dir_8x = f'group/box_dir/{number}'
    print("====================================================")
    print("Input Directory :",input_dir_8x)
    print("Input len :",len(input_dir_8x))
    print("Output Directory :",output_dir_8x)
    print("====================================================]")
    file_list = natsort.natsorted(glob(os.path.join(input_dir_8x,'*.jpg')))
    
    if os.path.isdir(output_dir_8x):
        shutil.rmtree(output_dir_8x)
    os.makedirs(output_dir_8x,exist_ok=True)
    image = cv2.imread(file_list[50])
    

    
    bus1_boxes = [((480 , 502) , (749, 604))]
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 0), (0, 128, 128)]
    for index , cord in enumerate(bus1_boxes):
        cv2.rectangle(image , cord[0] , cord[1],colors[index % len(colors)],2)
    win_name = 'test'
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(win_name , 1500 , 1500)
    cv2.setMouseCallback(win_name,show_coordinates)
    cv2.imshow(win_name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #exit(0)
    flag = str(input("==========Continue Press Any key ============="))
    if flag.upper == 'NO'or flag.upper == 'N':
        exit(0)
        
    detector_8x = YOLODetector(input_dir_8x, output_dir_8x, model_path='yolov8x.pt', boxes=bus1_boxes)


    detector_8x.process_images()

    print("Object detection completed for all images.")

    