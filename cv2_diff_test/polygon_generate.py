import cv2
import numpy as np

# 좌표를 저장할 리스트
polygon_points = []

# 현재 좌표를 저장할 변수
current_point = None

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global polygon_points, current_point
    
    # 마우스 왼쪽 버튼 클릭 시 좌표를 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        current_point = (x, y)  # 현재 클릭한 좌표를 저장
        polygon_points.append(current_point)  # 클릭한 좌표를 리스트에 저장
        print(f"Point added: ({x}, {y})")
        
        # 클릭한 위치에 점을 그려서 확인
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('image', img)

# 빈 이미지 생성
img = cv2.imread("cv2_diff_test/IMAGE_3RGB_MASK/22_MERGE.jpg")

# 창 생성 및 마우스 콜백 함수 설정
cv2.namedWindow('image' , cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', mouse_callback)

# d키로 저장된 좌표
stored_points = []

while True:
    cv2.imshow('image', img)
    
    # 키보드 입력 처리
    key = cv2.waitKey(1) & 0xFF  # 키보드 입력 감지

    # 'd' 키를 누르면 현재 포인트 저장
    if key == ord('d'):
        if current_point is not None:  # 좌표가 선택된 경우에만 저장
            stored_points.append(current_point)
            print(f"Stored point: {current_point}")
        else:
            print("No point selected yet.")

    # Esc 키를 누르면 종료
    if key == 27:
        break

# 저장된 좌표 출력
print("All Stored Points (from 'd' key presses):", stored_points)

# 창 닫기
cv2.destroyAllWindows()
