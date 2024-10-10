import cv2
import cv2
import numpy as np
import natsort
from glob import  glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["QT_DEBUG_PLUGINS"] = "0"

def test():
    empty_ = cv2.imread('krri1/_camera8_image_raw/image_0000.jpg' , cv2.IMREAD_COLOR)
    image_ = cv2.imread('krri1/_camera8_image_raw/image_0626.jpg', cv2.IMREAD_COLOR)

    diff = cv2.absdiff(empty_, image_)
    # RGB 채널 별로 차이가 5 이하인 부분을 마스크로 만듦 (각 채널별 차이가 5 이하)
    mask = np.all(diff <= 25, axis=-1).astype(np.uint8) * 255


    cv2.namedWindow('Mask',cv2.WINDOW_NORMAL)
    cv2.namedWindow("empty",cv2.WINDOW_NORMAL)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    cv2.imshow("empty",empty_)
    cv2.imshow("image",image_)
    cv2.imshow('Mask', mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run():
    NUMBER = 4
    empty_ = cv2.imread('cv2_diff_test/front_empty.jpg' , cv2.IMREAD_COLOR)
    file_list = natsort.natsorted(glob(os.path.join(f'cv2_diff_test/raw_seat/{NUMBER}','**','*.jpg'),recursive=True))
    

    # file_list = file_list[: len(file_list) // 3]

    h , w , _ = empty_.shape
    SAVE_DIR = f'cv2_diff_test/IMAGE_3RGB_MASK/{NUMBER}'
    os.makedirs(SAVE_DIR , exist_ok=True)

    #empty_ = empty_[y1 : y2 , x1 : x2]
    for index , i in tqdm(enumerate(file_list) , total=len(file_list)):
        end_name = os.path.basename(i)
        dir_name = os.path.dirname(i).split('/')[-1]
        file_name = dir_name+'_'+ end_name

        # boundary = [(757,806) , (761 , 806)]
        # inspect_cordinate = [(758,806),(759,806),(760,806)]


        image = cv2.imread(i , cv2.IMREAD_COLOR)
        
        origin_copy = image.copy()


        diff = cv2.absdiff(empty_ , image)

        mask = np.all(diff <= 20 , axis=-1).astype(np.uint8) * 255 

        # b ,g ,r = cv2.split(origin_copy)

        # print("before : " , g[758][806])
        # b = cv2.bitwise_not(b)
        # g = cv2.bitwise_not(g)
        # r = cv2.bitwise_not(r)
        # print("after : ", g[758][806])


        # image = cv2.merge([b,g,r])

        # print("merge  : ", image[758][806])

        image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)) 
        image = cv2.bitwise_not(image)
        chair = np.all(image == 255 , axis=-1)
        image[chair] = 0
        
        # b , g , r = cv2.split(image)

        # b_hist = cv2.calcHist([b],[0],None,[256],[0,256])
        # g_hist = cv2.calcHist([g],[0],None,[256],[0,256])
        # r_hist = cv2.calcHist([r],[0],None,[256],[0,256])
        
        # plt.figure(figsize=(8,8))
        # plt.subplot(2,2,1)
        # plt.ylim(0, 10000)
        # plt.plot(b_hist , color='b')
 
        # plt.subplot(2,2,2)
        # plt.ylim(0, 10000)
        # plt.plot(g_hist, color='g')
        # plt.subplot(2,2,3)
        # plt.ylim(0, 10000)
        # plt.plot(r_hist,color='r')
        # plt.show()

        #print(image.dtype)
        #image = image.astype(np.uint8)
        # for i in boundary:
        #     x , y = i
        #     diff[y][x] = (0,0,255)
        #     mask[y][x] = 100
        #     image[y][x] = (255,0,0)
        #     origin_copy[y][x] = (0,0,255)

        # for i in inspect_cordinate:
        #     x , y = i
        #     print("{},{} origin : {} diff : {} mask : {} image : {}".format(x,y,origin_copy[y][x] , diff[y][x] , mask [ y][x] , image[y][x]))

        file_name = os.path.join(SAVE_DIR , f"{file_name}")
        cv2.imwrite(file_name , image)  

        # cv2.namedWindow('Mask',cv2.WINDOW_NORMAL)
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("diff",cv2.WINDOW_NORMAL)
        # cv2.namedWindow("origin",cv2.WINDOW_NORMAL)

        # cv2.imshow('Mask', mask)
        # cv2.imshow("image",image)
        # cv2.imshow("diff",diff)
        # cv2.imshow("origin",origin_copy)
        # cv2.waitKey(0)

if __name__ == "__main__":
    run()


