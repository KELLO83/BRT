import os
import cv2
import numpy as np
from ultralytics import YOLO
from glob import glob
from tqdm.auto import tqdm

class YOLODetector:
    def __init__(self, input_directories, output_directory, model_path='yolov8x.pt'):
        self.input_directories = input_directories
        self.output_directory = output_directory
        self.model = YOLO(model_path)
        self.create_directories()

    def create_directories(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        for directory in self.input_directories:
            base_name = os.path.basename(directory)
            output_path = os.path.join(self.output_directory, base_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

    def save_image(self, image, file_name, directory):
        filename = os.path.join(self.output_directory, directory, file_name)
        cv2.imwrite(filename, image)
        #print(f"Saved: {filename}")

    def apply_nms(self, boxes, scores, iou_threshold=0.4):
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)
        if isinstance(indices, list) and len(indices) > 0:
            return [boxes[i[0]] for i in indices]
        elif isinstance(indices, np.ndarray) and indices.size > 0:
            return [boxes[i] for i in indices.flatten()]
        else:
            return []

    def test_images(self):
        for directory in self.input_directories:
            base_name = os.path.basename(directory)
            output_subdir = os.path.join(self.output_directory, base_name)
            file_list = glob(os.path.join(directory, "*.jpg"))
            
            for file_path in tqdm(file_list):
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Warning: File {file_path} does not exist or cannot be read.")
                    continue
                
                results = self.model(img, verbose=False)
                
                boxes = []
                scores = []
                
                for result in results:
                    for box in result.boxes:
                        if int(box.cls) == 0:  # Assuming class ID 0 represents a person
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            score = box.conf
                            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                            scores.append(float(score))
                
                # Apply Non-Maximum Suppression (NMS)
                nms_boxes = self.apply_nms(boxes, scores)
                
                for box in nms_boxes:
                    x1, y1, w, h = box
                    center_x, center_y = x1 + w // 2, y1 + h // 2

                    # Draw the center point on the image
                    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
                    # Draw the coordinates on the image
                    cv2.putText(img, f'({center_x}, {center_y})', 
                                (center_x, center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.3, 
                                (255, 255, 255), 
                                1)

                # Save the processed image
                output_path = os.path.join(output_subdir, os.path.basename(file_path))
                self.save_image(img, os.path.basename(file_path), base_name)

if __name__ == "__main__":
    number = str('3')
    input_dirs = [f'{number}/_camera6_image_raw',f'{number}/_camera8_image_raw']
    output_dir = f'detect_{number}'

    detector = YOLODetector(input_dirs, output_dir, model_path='yolov8x.pt')

    detector.test_images()

    print("Object detection completed for all images.")