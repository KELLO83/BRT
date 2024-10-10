import cv2
import numpy as np

A = cv2.imread('Two-examples-of-noisy-images-salt-and-pepper-noise-on-the-left-and-Gaussian-noise.png', cv2.IMREAD_GRAYSCALE)


U, Sigma, VT = np.linalg.svd(A)

# Sigma 값을 대각 행렬로 변환
Sigma_matrix = np.zeros((A.shape[0], A.shape[1]))

np.fill_diagonal(Sigma_matrix, Sigma)
print("Sigma_matirx shape", Sigma_matrix.shape)
# Sigma_matrix = Sigma_matrix[:2 , :]

Sigma_matrix_reduced = np.zeros_like(Sigma_matrix)
Sigma_matrix_reduced[:10, :] = Sigma_matrix[ :10 , :]  # 첫 번째, 두 번째 특이값만 유지

print("Sigma값\n",Sigma_matrix_reduced)
# 복원된 A 행렬 계산
A_reconstructed = np.dot(U, np.dot(Sigma_matrix_reduced, VT))

print("\n원본 행렬과 복원된 행렬의 차이:\n", A - A_reconstructed)

A_reconstructed = np.clip(A_reconstructed , 0 , 255).astype(np.uint8)

cv2.namedWindow("t",cv2.WINDOW_NORMAL)
cv2.imshow("t",A_reconstructed)
cv2.waitKey(0)