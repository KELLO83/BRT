import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
import os
from natsort import  natsorted
import seaborn as sns
from itertools import  combinations
from math import comb
from tqdm import tqdm
os.environ["QT_DEBUG_PLUGINS"] = "0"



background = natsorted(glob(os.path.join(f'back_ground_front','*.jpg')))
sample = cv2.imread(background[0])
sample = sample[: , sample.shape[1]//2:]
h ,w , _ = sample.shape
blue_score = np.zeros((h,w), dtype=np.float128)
green_score = np.zeros((h,w), dtype=np.float128)
red_score = np.zeros((h,w), dtype=np.float128)

n = len(background)
total_combinations = comb(n, 2)  # n개 중 2개를 선택하는 조합의 경우의 수

for (i,j) in tqdm(combinations(background , 2), total=total_combinations):
    c1 = cv2.imread(i)
    c2 = cv2.imread(j)
    diff = cv2.absdiff(c1,c2)
    
    # cv2.namedWindow("d",cv2.WINDOW_NORMAL)
    # cv2.imshow("d",diff)
    # cv2.waitKey(0)
    diff = diff [ : , w : ]
    diff = diff.astype(np.float64)
    blue_score += diff[: , : ,0]
    green_score += diff[: , : ,1]
    red_score += diff[ : , : , 2]

blue_score = (blue_score / total_combinations).astype(np.uint8)
green_score = (green_score / total_combinations).astype(np.uint8)
red_score = (red_score / total_combinations).astype(np.uint8)

merge = cv2.merge([blue_score , green_score , red_score])
merge = np.clip(merge , 0 , 255)
cv2.namedWindow("m",cv2.WINDOW_NORMAL)
cv2.imshow('m',merge)
cv2.waitKey(0)
b , g , r = cv2.split(merge)
def hist__(b,g,r):
    b_hist = cv2.calcHist([b],[0],None,[256],[0,256])
    g_hist = cv2.calcHist([g],[0],None,[256],[0,256])
    r_hist = cv2.calcHist([r],[0],None,[256],[0,256])

    def calc_hist_percentiles(hist, color, subplot_position, percentiles=[0.1,0.8, 0.9, 0.99]):
        cdf = np.cumsum(hist)
        total_pixels = cdf[-1]
        p_index = []

        for p in percentiles:
            p_value = total_pixels * p
            idx = np.where(cdf >= p_value)[0][0]
            print(idx)
            p_index.append(idx)

        plt.subplot(subplot_position)
        plt.plot(hist , color = color)

        color = ['b','g','r','c','m','y']
        for i , idx in enumerate(p_index):
            plt.axvline(idx , linestyle='--', label=f'{int(percentiles[i] * 100)}% at {idx}', color= color[i % len(color)])
            plt.text(idx, max(hist) * (0.9 - i * 0.2), f'{int(percentiles[i] * 100)}%: {idx}', color=color[i % len(color)], ha='center')

        plt.legend()
    plt.figure(figsize=(12,12))
    calc_hist_percentiles(b_hist, 'b', 221)  
    calc_hist_percentiles(g_hist, 'g', 222)  
    calc_hist_percentiles(r_hist, 'r', 223) 
    plt.show()

hist__(b,g,r)


