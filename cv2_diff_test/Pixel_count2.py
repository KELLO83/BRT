import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
import os
from natsort import  natsorted
import seaborn as sns

os.environ["QT_DEBUG_PLUGINS"] = "0"

number = 24
#empty_file_list = natsorted(glob(os.path.join('cv2_diff_test/front','**','*.jpg'),recursive=True))
empty_file_list = natsorted(glob(os.path.join('back_6',"**","*.jpg"),recursive=True))
#empty_file_list = natsorted(glob(os.path.join(f'cv2_diff_test/raw_seat/{number}',"**","*.jpg"),recursive=True))
#empty_file_list = natsorted(glob(os.path.join(f'cv2_diff_test/removed_raw_seat/{number}','**','*.jpg'),recursive=True))

mask = cv2.imread(f"cv2_diff_test/NEW_MASK_USE/{number}.jpg" , cv2.IMREAD_GRAYSCALE)
empty = cv2.imread('cv2_diff_test/back_empty.jpg')

if not empty_file_list or mask is None:
    raise Exception


MASK_PIXEL = np.sum( mask > 200)
print("Mask Pixel Count : ",MASK_PIXEL)
save_list = []

acc = []
mask_cordinate = np.where(mask > 200)
mask_cordinate = list(zip(mask_cordinate[0] , mask_cordinate[1]))
under_image = 0

stop_point = 48
is_empty = True

for image_path in empty_file_list:
    pixel_count = 0
    compare = cv2.imread(image_path , cv2.IMREAD_COLOR)
    
    compare_copy = compare.copy()

    diff = cv2.absdiff(compare , empty)
    b_d , g_d , r_d = cv2.split(diff)
    for i in mask_cordinate:
        y , x = i
        compare_copy[y][x] = (0,0,255)



        b = b_d[y][x]
        g = g_d[y][x]
        r = r_d[y][x]

        if b >= 49 and g >=49 and r>=63:
            pixel_count += 1 # b g r 이 임계점보다 작다면 좌석에 변화가 존재하지않는다 pixel_cout 변수는 좌석에 변화가많을수록 더크다


    acc.append(pixel_count)
    print(f"PIXEL :  {pixel_count} / {MASK_PIXEL}")
    
    if is_empty:
        if pixel_count >= stop_point:
            under_image +=1
            # cv2.namedWindow(f"{os.path.basename(image_path)}",cv2.WINDOW_NORMAL)
            # cv2.imshow(f"{os.path.basename(image_path)}",compare_copy)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    else:
        if pixel_count <= stop_point :
            under_image +=1


    cv2.namedWindow("t",cv2.WINDOW_NORMAL)
    cv2.imshow("t",compare_copy)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    pixel_count = 0 

# print('=====================================================================')
print("{}  total {} / {} : ".format(stop_point,under_image,len(empty_file_list)))
import pandas as pd
acc = np.array(acc).astype(np.uint32)
acc = acc[~np.isnan(acc)]  # NaN 제거
acc = acc[np.isfinite(acc)]  # Infinite 값 제거
from scipy.stats import gaussian_kde
density = gaussian_kde(acc)
xs = np.linspace(min(acc), max(acc), 1000)
density_value = density(xs)
for i in range(len(xs) - 1):
    if density_value[i] <= 0.001:  # 밀도가 0에 가까운 구간
        plt.plot(xs[i:i+2], density_value[i:i+2], color='blue')
    else:
        plt.plot(xs[i:i+2], density_value[i:i+2], color='red')

CDF = np.cumsum(density_value)
CDF = CDF / CDF[-1]

from scipy.interpolate import interp1d
INV_CDF = interp1d(CDF , xs)

p70 = INV_CDF(0.70)   # 70% 구간
p80 = INV_CDF(0.80)   # 80% 구간
p90 = INV_CDF(0.90)   # 90% 구간
p99 = INV_CDF(0.99)   # 99% 구간
p100 = INV_CDF(1.00)   # 100% 구간

plt.axvline(p100, color='b', linestyle='--', label=f'100% at x={p100:.2f}')

plt.text(p100, max(density_value), f'100%: {p100:.2f}', color='b', ha='center')

plt.axvline(p70, color='g', linestyle='--', label=f'70% at x={p70:.2f}')
plt.axvline(p80, color='y', linestyle='--', label=f'80% at x={p80:.2f}')
plt.axvline(p90, color='c', linestyle='--', label=f'90% at x={p90:.2f}')
plt.axvline(p99, color='m', linestyle='--', label=f'99% at x={p99:.2f}')

plt.text(p70, max(density_value) * 0.6, f'70%: {p70:.2f}', color='g', ha='center')
plt.text(p80, max(density_value) * 0.7, f'80%: {p80:.2f}', color='y', ha='center')
plt.text(p90, max(density_value) * 0.8, f'90%: {p90:.2f}', color='c', ha='center')
plt.text(p99, max(density_value) * 0.9, f'99%: {p99:.2f}', color='m', ha='center')


p0 = xs[0]
p1 = INV_CDF(0.01)   # 1% 구간
p3 = INV_CDF(0.03)   # 3% 구간
p5 = INV_CDF(0.05)   # 5% 구간
p10 = INV_CDF(0.10)  # 10% 구간

plt.axvline(p0, color='r', linestyle='--', label=f'0% at x={p0:.2f}')
plt.axvline(p1, color='g', linestyle='--', label=f'1% at x={p1:.2f}')
plt.axvline(p3, color='y', linestyle='--', label=f'3% at x={p3:.2f}')
plt.axvline(p5, color='c', linestyle='--', label=f'5% at x={p5:.2f}')
plt.axvline(p10, color='m', linestyle='--', label=f'10% at x={p10:.2f}')

# 수직선에 텍스트 추가
plt.text(p0, max(density_value) * 0.5, f'0%: {p0:.2f}', color='r', ha='center')
plt.text(p1, max(density_value) * 0.6, f'1%: {p1:.2f}', color='g', ha='center')
plt.text(p3, max(density_value) * 0.7, f'3%: {p3:.2f}', color='y', ha='center')
plt.text(p5, max(density_value) * 0.8, f'5%: {p5:.2f}', color='c', ha='center')
plt.text(p10, max(density_value) * 0.9, f'10%: {p10:.2f}', color='m', ha='center')

plt.xlabel('값')
plt.ylabel('빈도')
plt.show()

# 그래프 설정

# plt.plot(xs, density(xs), color='red')

# acc_hist = cv2.calcHist([acc] , [0] , None , [MASK_PIXEL] , [0 , float(MASK_PIXEL+1)])


#plt.subplot(2,2,2)


# pdf = acc_hist / acc_hist.sum()
# X = np.arange(MASK_PIXEL)
# plt.plot(X , pdf , label = 'PDF')

# cdf = np.cumsum(pdf)
# p50 = np.searchsorted(cdf, 0.50)  # 50% 구간
# p70 = np.searchsorted(cdf, 0.70)  # 70% 구간
# p80 = np.searchsorted(cdf, 0.80)  # 80% 구간
# p99 = np.searchsorted(cdf, 0.999)  # 99% 구간

# plt.axvline(p50, color='r', linestyle='--', label='50% at x={}'.format(p50))
# plt.axvline(p70, color='g', linestyle='--', label='70% at x={}'.format(p70))
# plt.axvline(p80, color='y', linestyle='--', label='80% at x={}'.format(p80))
# plt.axvline(p99, color='c', linestyle='--', label='90% at x={}'.format(p99))

# plt.text(p50, max(pdf) * 0.8, f'50%: {p50}', color='r', ha='center')
# plt.text(p70, max(pdf) * 0.7, f'70%: {p70}', color='g', ha='center')
# plt.text(p80, max(pdf) * 0.6, f'80%: {p80}', color='y', ha='center')
# plt.text(p99, max(pdf) * 0.5, f'99.9%: {p99}', color='c', ha='center')


# plt.show()
# print("=========================================")