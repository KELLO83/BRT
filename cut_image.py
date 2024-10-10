import cv2
import numpy as np
import fish_map

disort_image = cv2.imread('8/camera6_images/frame_1712302092_000004.jpg' , cv2.IMREAD_COLOR)
undisort_image = cv2.imread("Undisotrted_image_raw_set/8/camera6_images/frame_1712302092_000004.jpg" , cv2.IMREAD_COLOR)

print(disort_image.shape)
h ,w , _ = disort_image.shape

center_x = w // 2
center_y = h // 2

pad_x = 500
x1 , x2 = center_x - pad_x , center_x + pad_x


pad_y = 450
y1 , y2 = center_y - pad_y , center_y + pad_y

disort_image = disort_image[y1: y2 , x1 : x2  , : ]
#disort_image = disort_image[y1 : y2, x1: x2, :]
print(disort_image.shape)
cv2.namedWindow("d",cv2.WINDOW_NORMAL)
cv2.namedWindow("und",cv2.WINDOW_NORMAL)
cv2.imshow("und" , undisort_image)
cv2.imshow("d",disort_image)
cv2.waitKey(0)