import cv2
import glob
import os

os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.info=false;*.warning=false"
def draw_rectangle(event, x, y, flags, param):
    global x1, y1, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (x1, y1), (x, y), (0, 255, 0), 2)
            #cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (x1, y1), (x, y), (0, 255, 0), 2)
        cv2.imshow('image', img)
        print(f"{x1} {y1} {x} {y}")
        rectangles.append((x1, y1, x, y)) 
    
    
def show_image_with_rectangles(img):
    img_with_rectangles = img.copy()
    for (x1, y1, x2, y2) in rectangles:
        cv2.rectangle(img_with_rectangles, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('image', img_with_rectangles)
    

from natsort import natsorted
drawing = False
target_number = 33
file_list = natsorted(glob.glob(os.path.join(f"group/{target_number}","*.jpg")))
#file_list = natsorted(glob.glob(os.path.join('5/_camera8_image_raw',"*.jpg")))
index = 0
rectangles = [
    [802, 358, 886, 447],
    [680, 350, 784, 431],
    [746, 451, 861, 574],
    [589, 461, 717, 564],
    [1201, 445, 1313, 540],
    [480, 502, 749, 604],
    [482, 612, 749, 759],
    [472, 775, 737, 964],
    [1160, 529, 1415, 601],
    [1209, 609, 1420, 767],
    [1266, 787, 1466, 921]
]
print("=====================  DEBUG CODE =========================")

  
bus1_boxes = []

flattened_tuples = [(x1, y1, x2, y2) for ((x1, y1), (x2, y2)) in bus1_boxes]
rectangles.extend(flattened_tuples)
print(rectangles)
print("============================================================")

window_name = 'image'
cv2.namedWindow(window_name ,cv2.WINDOW_NORMAL)
img = cv2.imread(file_list[index])
show_image_with_rectangles(img)
cv2.setMouseCallback('image', draw_rectangle)

while True:
    key = cv2.waitKey(1)
    if key == 27:  
        break
    elif key == ord('d'):  
        index = (index + 1) % len(file_list)
        img = cv2.imread(file_list[index])
        #print(f"{file_list[index]}")
        show_image_with_rectangles(img)
    elif key == 255:  # Delete 키
        img = cv2.imread(file_list[index])  
        rectangles.pop()
        cv2.imshow('image', img)
    elif key == ord('a'):
        index = (index - 1) % len(file_list)
        img = cv2.imread(file_list[index])
        #print(f"{file_list[index]}")
        show_image_with_rectangles(img)
        
    elif key == 80:
        print(rectangles)

cv2.destroyAllWindows()
