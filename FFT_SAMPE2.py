import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 이미지를 불러와서 회색조로 변환
image = cv2.imread('Two-examples-of-noisy-images-salt-and-pepper-noise-on-the-left-and-Gaussian-noise.png', cv2.IMREAD_GRAYSCALE)

# 2. 2D FFT 수행 (주파수 도메인 변환)
fft_image = np.fft.fft2(image)

# 3. 푸리에 변환의 중심을 중앙으로 이동
fft_shift = np.fft.fftshift(fft_image)

# 4. 주파수 성분의 크기 계산 (절댓값)
magnitude_spectrum = np.abs(fft_shift)
magnitude_spectrum_log = np.log(1 + magnitude_spectrum)

# 5. 주파수 구간 필터링 (30 미만 주파수 제거)
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2  # 중앙 좌표

max_th = np.max(magnitude_spectrum_log)
print(max_th)
# 30 이하의 주파수 성분을 0으로 설정하는 마스크 생성
threshold = max_th  # 밝기 임계값 설정
mask = magnitude_spectrum_log < threshold  # 임계값 이상이면 제거

# 마스크를 적용하여 저주파 성분 제거
fft_shift_filtered = fft_shift * mask
#fft_shift_filtered = fft_shift
# 6. 역 푸리에 변환 (Inverse FFT)
fft_shift_filtered = np.fft.ifftshift(fft_shift_filtered)  # 중심을 원래대로 이동
image_back = np.fft.ifft2(fft_shift_filtered)  # 역변환 수행
image_back = np.abs(image_back)  # 복소수를 실수로 변환

# 7. 결과 시각화
plt.figure(figsize=(10, 5))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 필터 적용 후 주파수 스펙트럼
plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + np.abs(fft_shift_filtered)), cmap='gray')
plt.title('Filtered Spectrum')
plt.axis('off')

# 역변환 후 이미지
plt.subplot(1, 3, 3)
plt.imshow(image_back, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

plt.show()
