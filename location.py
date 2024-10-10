import cv2
import numpy as np

# Mouse event 확인하기
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

click = False  # Mouse 클릭된 상태 (false = 클릭 x / true = 클릭 o) : 마우스 눌렀을때 true로, 뗐을때 false로
x1, y1 = -1, -1

# Mouse Callback 함수 : 파라미터는 고정됨
def draw_rectangle(event, x, y, flags, param):
    global x1, y1, click  # 전역변수 사용

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        x1, y1 = x, y
        print("사각형의 왼쪽 위 설정 : (" + str(x1) + ", " + str(y1) + ")")
		
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
        if click:  # 마우스를 누른 상태 일 경우
            result[:] = img.copy()  # 이전 사각형 지우기 위해 원본 이미지 복사
            cv2.rectangle(result, (x1, y1), (x, y), (255, 0, 0), -1)
            print("(" + str(x1) + ", " + str(y1) + "), (" + str(x) + ", " + str(y) + ")")

    elif event == cv2.EVENT_LBUTTONUP:
        click = False  # 마우스를 떼면 상태 변경
        cv2.rectangle(result, (x1, y1), (x, y), (255, 0, 0), -1)


# 카메라로 촬영한 이미지 하나를 가져오기
img = cv2.imread('detect_scen1-1/_camera6_image_raw/image_0183.jpg')

# 이미지를 180도 회전
#img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

result = img.copy()  # 이미지를 복사하여 result 변수에 저장

cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('cam', draw_rectangle)  # 마우스 이벤트 후 callback 수행하는 함수 지정


while True:
    cv2.imshow('cam', result)  # 화면을 보여준다.

    k = cv2.waitKey(1) & 0xFF  # 키보드 입력값을 받고

    if k == 27:  # esc를 누르면 종료
        break

cv2.destroyAllWindows()