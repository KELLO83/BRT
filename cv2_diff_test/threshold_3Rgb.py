import cv2
import numpy as np
import natsort
from glob import  glob
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

def main():
    def event(pos):
        global IMAGE , Number , NAME
        image_copy = IMAGE.copy
        value = cv2.getTrackbarPos('pos','test')
        _ , mat = cv2.threshold(IMAGE , value , 255 ,cv2.THRESH_BINARY)
        cv2.imwrite('cv2_diff_test/SEAT_GRAY_MASK/NUMBER.jpg',mat)
        cv2.imshow("test" , mat)

    NAME = 'cv2_diff_test/IMAGE_3RGB_MASK/4_MERGE.jpg'
    IMAGE = cv2.imread(f'{NAME}' , cv2.IMREAD_GRAYSCALE)
    Number = ''.join(re.findall(r'\d',os.path.basename(NAME)))

    if IMAGE is None:
        FileExistsError


    cv2.namedWindow("target" , cv2.WINDOW_NORMAL)
    cv2.imshow("target" , IMAGE)
    cv2.namedWindow("test",cv2.WINDOW_NORMAL)
    cv2.createTrackbar('pos','test',0,255,event)
    cv2.imshow("test" , IMAGE)
    cv2.waitKey(0)

def test_3channel():
    number = 22
    image = cv2.imread(f'cv2_diff_test/IMAGE_3RGB_MASK_CROP/{number}.jpg')

    b ,g , r  = cv2.split(image)
    print(np.max(b) , np.min(b))
    print(b.shape)
    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    b_hist = cv2.calcHist([b],[0],None,[256],[0,256])
    print(b_hist)
    plt.plot(b_hist.flatten() , color='b')
    plt.show()
    
    # _ , b_  = cv2.threshold(b , 220 , 250 , cv2.THRESH_BINARY)
    # _ , g_ = cv2.threshold(g , 220 , 250 , cv2.THRESH_BINARY)
    # _ , r_  = cv2.threshold(r , 220 , 250 , cv2.THRESH_BINARY)

    
    # merge = cv2.bitwise_and(b_,g_)
    # merge = cv2.bitwise_and(merge , r_)
    
    # count = np.sum(merge>200)
    # print("전체 픽셀 수 :",count)
    # #cv2.namedWindow("test",  cv2.WINDOW_NORMAL)
    # #cv2.imwrite(f"cv2_diff_test/IMAGE_3RGB_MASK_USE/{number}.jpg",merge)
    # #cv2.imshow("test",merge)
    # print("전체 픽셀 수 :",count)
    # cv2.waitKey(0)



if __name__ == "__main__":
    test_3channel()