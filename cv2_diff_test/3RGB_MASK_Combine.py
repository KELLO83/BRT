from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
from glob import  glob
import natsort
from tqdm import tqdm
import logging

import cv2
import numpy as np
import os
import natsort
from glob import glob

def test():
    file_list = natsort.natsorted(glob(os.path.join('cv2_diff_test/morpholgy/inverse/t35', '*.jpg')))
    sample = cv2.imread(file_list[0], cv2.IMREAD_COLOR)
    h, w, c = sample.shape
    file_len = len(file_list)

    red_score = [[0 for _ in range(w)] for _ in range(h)]
    blue_score = [[0 for _ in range(w)] for _ in range(h)]
    green_score = [[0 for _ in range(w)] for _ in range(h)]

    # 파일 개수 가져오기
    file_len = len(file_list)

    count = 0
    # 이미지마다 반복
    for file_name in tqdm(file_list):
        image = cv2.imread(file_name, cv2.IMREAD_COLOR)  # file_name으로 수정
        b, g, r = cv2.split(image)

        # 각각의 픽셀에 대해 채널별 값 더하기
        for i in range(h):
            for j in range(w):
                red_score[i][j] += r[i][j] 
                blue_score[i][j] += b[i][j] 
                green_score[i][j] += g[i][j] 

        if count %10 ==0 and count !=0 :
            logging.info(f'{count} Contiune !')
            rs =  np.array([[x // count for x in row] for row in red_score], dtype=np.uint8)
            bs =  np.array([[x // count for x in row] for row in blue_score], dtype=np.uint8)
            gs = np.array([[x // count for x in row] for row in green_score], dtype=np.uint8)
            
            merge = cv2.merge([bs , gs , rs])
            os.makedirs('cv2_diff_test/t' , exist_ok=True)
            cv2.imwrite(f'cv2_diff_test/t/merged{count}.jpg',merge)
            # cv2.namedWindow("Merged Image",cv2.WINDOW_NORMAL)
            # cv2.imshow('Merged Image',merge)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        count += 1

    # 각 채널별 평균값 계산 후 uint8로 변환
    red_score = np.array([[x // file_len for x in row] for row in red_score], dtype=np.uint8)
    blue_score = np.array([[x // file_len for x in row] for row in blue_score], dtype=np.uint8)
    green_score = np.array([[x // file_len for x in row] for row in green_score], dtype=np.uint8)

    merge = cv2.merge([blue_score, green_score, red_score])

    # 결과 이미지 확인
    cv2.imshow("Merged Image", merge)
    cv2.imwrite("merged.jpg",merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rgb():
    # 파일 목록 정렬 후 읽어오기
    NUMBER = 4
    file_list = natsort.natsorted(glob(os.path.join(f'cv2_diff_test/IMAGE_3RGB_MASK/{NUMBER}', '*.jpg')))
    #file_list = natsort.natsorted(glob(os.path.join(f'cv2_diff_test/raw_seat/{NUMBER}', '*.jpg')))

    # file_list = file_list[:len(file_list)//2]
    if file_list is None:
        raise FileNotFoundError
    sample = cv2.imread(file_list[0], cv2.IMREAD_COLOR)
    h, w, c = sample.shape


    # 스코어 numpy 배열로 초기화 (h, w, 3)의 크기로 생성, 0으로 채움
    red_score = np.zeros((h, w), dtype=np.float128)
    blue_score = np.zeros((h, w), dtype=np.float128)
    green_score = np.zeros((h, w), dtype=np.float128)

    # 이미지마다 반복
    for file_name in tqdm(file_list):
        image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        image = image.astype(np.float64)
        b, g, r = cv2.split(image)

        # 각 채널을 바로 numpy 배열에 누적
        red_score += r 
        blue_score += b 
        green_score += g 

    # 평균 계산 (각 스코어를 파일 개수로 나눔)
    red_score = (red_score / len(file_list)).astype(np.uint8)
    blue_score = (blue_score / len(file_list)).astype(np.uint8)
    green_score = (green_score / len(file_list)).astype(np.uint8)

    # 채널 합쳐서 최종 이미지 만들기
    merge = cv2.merge([blue_score, green_score, red_score])

    merge = np.clip(merge , 0 , 255 )

    b , g , r = cv2.split(merge)
    b_hist = cv2.calcHist([b] , [0] ,None , [256], [0,256])
    g_hist = cv2.calcHist([g] , [0] ,None , [256], [0,256])
    r_hist = cv2.calcHist([r] , [0] ,None , [256], [0,256])


    cv2.imwrite(f"cv2_diff_test/IMAGE_3RGB_MASK/{NUMBER}_MERGE.jpg",merge)
    # 결과 이미지 확인
    cv2.namedWindow("Merged",cv2.WINDOW_NORMAL)
    # cv2.namedWindow("red",cv2.WINDOW_NORMAL)
    # cv2.namedWindow("green" , cv2.WINDOW_NORMAL)
    # cv2.namedWindow("blue" , cv2.WINDOW_NORMAL)
    cv2.imshow("Merged", merge)
    # cv2.imshow("red",red_score)
    # cv2.imshow("blue",blue_score)
    # cv2.imshow("green",green_score)
   
    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    plt.ylim(0,10000)
    plt.plot(b_hist , color='blue')

    plt.subplot(2,2,2)
    plt.ylim(0,10000)
    plt.plot(g_hist , color='green')

    plt.subplot(2,2,3)
    plt.ylim(0,10000)
    plt.plot(r_hist , color='red')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gray():
    file_list = natsort.natsorted(glob(os.path.join('cv2_diff_test/morpholgy/mask/t35','*.jpg')))
    sampel = cv2.imread(file_list[0] , cv2.IMREAD_GRAYSCALE)
    h , w  = sampel.shape
    score_value = 255 / len(file_list)
    score_map = np.zeros((h,w) , dtype=np.float64)
    for i in tqdm(file_list):
        image = cv2.imread(i , cv2.IMREAD_GRAYSCALE)

        score_map[image == 0 ] += score_value


    score_map_uint8 = np.clip(score_map , 0 , 255).astype(np.uint8)
    cv2.namedWindow("res",cv2.WINDOW_NORMAL)
    cv2.imwrite("cv2_diff_test/t35.jpg",score_map_uint8)
    cv2.imshow("res", score_map_uint8)
    cv2.waitKey(0)

if __name__ == "__main__":
    #test()
    rgb()
    #gray()