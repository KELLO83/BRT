import cv2


def mouse_callback(event, x, y, flags, param):
    global image
    image_ = image.copy()
    if event == cv2.EVENT_MOUSEMOVE:
        b, g, r = image[y, x]  # BGR 값 추출

        # BGR 값을 텍스트로 출력
        text = f"{b}, {g}, {r}"
        cv2.putText(image_, f"{text}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('image', image_)

image = cv2.imread('2번.png',cv2.IMREAD_COLOR)
#image__ = cv2.imread("cv2_diff_test/24/detect_scen1-1_image_0000.jpg" , cv2.IMREAD_COLOR)
cv2.namedWindow("image" , cv2.WINDOW_NORMAL) 
cv2.setMouseCallback('image',mouse_callback)
#cv2.namedWindow("image__", cv2.WINDOW_NORMAL)
cv2.imshow('image',image)

while True:
    if cv2.waitKey(1) & 0XFF == 27:
        break

cv2.destroyAllWindows()