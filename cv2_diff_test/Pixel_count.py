import cv2
import cv2
import numpy as np
import natsort
from glob import  glob
import os
from tqdm import tqdm


number = 22
file_list = natsort.natsorted(glob(os.path.join(f'cv2_diff_test/raw_seat/{number}',"**", '*.jpg'),recursive=True))
file_list = file_list[155:]

print("=================================== file len :  {} ===============================".format(len(file_list)))
input("Continue Press Any key !!!!1")
# test = natsort.natsorted(glob(os.path.join('cv2_diff_test/raw_seat/12',"**", '*.jpg'),recursive=True))
# file_list = test + file_list
Empty = cv2.imread('cv2_diff_test/back_empty.jpg' , cv2.IMREAD_COLOR)
if Empty is  None:
    raise FileExistsError

mask = cv2.imread(f'cv2_diff_test/IMAGE_3RGB_MASK_USE/{number}.jpg' ,cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread(f'cv2_diff_test/IMAGE_GRAY_MASK/{number}.jpg', cv2.IMREAD_GRAYSCALE)

count = np.sum(mask>200)
b ,g ,r = Empty.shape
threshold = 15
out_image_count = 0
print("전체 픽셀수 :" ,count)
for i in file_list:
    image = cv2.imread(i , cv2.IMREAD_COLOR)
    image_copy = image.copy()   

    abs_diff_image = cv2.absdiff(image , Empty)        
    red_pixels = np.where(mask > 200)
    red_pixel_coords = list(zip(red_pixels[0], red_pixels[1]))

    pixel_count = 0  # 변수명 수정

    for coord in red_pixel_coords:
        y, x = coord  
        cv2.circle(image_copy , (x,y) , radius=1 , color=(0,0,255) , thickness=-1)
        b_e, g_e, r_e = Empty[y][x] 
        b, g, r = image[y][x]     
        assert Empty.shape == image.shape, 'Image shapes are not the same'

        b_diff = abs(b_e - b)
        g_diff = abs(g_e - g)
        r_diff = abs(r_e - r)

        if max(b_diff, g_diff, r_diff) <= threshold:
            pixel_count += 1   # 변화가없는 픽셀의 갯수 변화가 많을수록 해당변수의 값은 작아진다

    if pixel_count > 50 :
        out_image_count += 1
    print(f"Threshold 이하인 픽셀 수: {pixel_count} / {count}")
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("circle" , cv2.WINDOW_NORMAL)
    # cv2.namedWindow("diff",cv2.WINDOW_NORMAL)
    # cv2.imshow("image",image)
    # cv2.imshow("circle",image_copy)
    # cv2.imshow("diff",abs_diff_image)
    # cv2.waitKey(0)

print("out image count  : ",out_image_count)
