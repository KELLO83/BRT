import cv2
import numpy as np
from glob import  glob
from natsort import natsorted
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm

os.environ["QT_DEBUG_PLUGINS"] = "0"
file_list = natsorted(glob(os.path.join('cv2_diff_test/raw_seat/24',"*.jpg"),recursive=True))
empty_image = cv2.imread('cv2_diff_test/back_empty.jpg'  , cv2.IMREAD_COLOR)
x1 ,y1 ,x2 ,y2 = [1098 , 274 ,1166, 364 ] # 24번좌석
attach_backgroud = np.zeros(empty_image.shape[:2],dtype=np.uint8)

empty_image = empty_image[y1:y2 , x1:x2]
h , w , _ = empty_image.shape

file_len = len(file_list)
red_score = np.zeros((h, w), dtype=np.float128)
blue_score = np.zeros((h, w), dtype=np.float128)
green_score = np.zeros((h, w), dtype=np.float128)

for i in tqdm(file_list):
    c = cv2.imread(i)
    c = c[y1 : y2 , x1 : x2]
    
    diff = cv2.absdiff(empty_image , c)
    mask = np.all(diff <= 10 , axis=-1)
    diff[mask] = [0,0,0]
    
    # cv2.namedWindow("c",cv2.WINDOW_NORMAL)
    # cv2.imshow("c" , diff)
    # cv2.waitKey(0)

    diff = diff.astype(np.float64)
    blue_score +=diff[ : , : ,0]
    green_score += diff[ : , : ,1]
    red_score += diff[ : , : ,2]

red_score = (red_score / len(file_list)).astype(np.uint8)
blue_score = (blue_score / len(file_list)).astype(np.uint8)
green_score = (green_score / len(file_list)).astype(np.uint8)

merge = cv2.merge([blue_score, green_score, red_score])

merge = np.clip(merge , 0 , 255 )

cv2.namedWindow("merge",cv2.WINDOW_NORMAL)
cv2.imshow('merge',merge)
cv2.waitKey(0)

b , g , r = cv2.split(merge)
def hist__(b,g,r):
    b_hist = cv2.calcHist([b],[0],None,[256],[0,256])
    g_hist = cv2.calcHist([g],[0],None,[256],[0,256])
    r_hist = cv2.calcHist([r],[0],None,[256],[0,256])

    def calc_hist_percentiles(hist, color, subplot_position, percentiles=[0.1,0.3, 0.8, 0.9, 0.95]):
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
    cv2.destroyAllWindows()
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    #plt.show()

#hist__(b,g,r)

b_mask = (b > 49)
g_mask = (g > 49)
r_mask = (r > 63)

combined_mask = b_mask & g_mask & r_mask 

combined_mask_3d = combined_mask[..., None]  # 차원을 확장하여 3채널에 적용

out_image = np.where(combined_mask_3d , cv2.bitwise_not(merge) , 0)

cv2.namedWindow("c",cv2.WINDOW_NORMAL)
cv2.imshow("c",out_image)

cv2.namedWindow("m",cv2.WINDOW_NORMAL)
combined_mask_3d = np.where(combined_mask_3d , 255 , 0).astype(np.uint8)
cv2.imshow("m" , combined_mask_3d)

attach_backgroud[y1:y2 ,x1:x2] =  combined_mask_3d.squeeze()
print(attach_backgroud.shape)
cv2.namedWindow("ab", cv2.WINDOW_NORMAL)
cv2.imwrite("cv2_diff_test/NEW_MASK_USE/24.jpg",attach_backgroud)
cv2.imshow("ab",attach_backgroud)

cv2.waitKey(0)