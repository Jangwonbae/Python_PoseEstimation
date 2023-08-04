#
#   공대선배
#   opencv python 코딩
#   영상 조정 코드
#

import cv2
import numpy as np
from opencv_functions import *

# 카메라 영상을 받아올 객체 선언 및 설정(영상 소스, 해상도 설정)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()     # 카메라로부터 영상을 받아 frame에 저장
    cv2.imshow("original", frame)   # 원본 영상 출력
    # filered = color_filter(frame, 'red', 1.2)   # 원본영상에서 빨간색을 강조
    # cv2.imshow("red", filered)      # 색감을 바꾼 영상 출력
    # brightness = set_brightness(frame, 20)  # 밝기를 전체적으로 20픽셀 밝게 해줌
    # cv2.imshow("brightness", brightness)    # 밝기를 바꾼 영상 출력
    # constrast = set_contrast(frame, 0.9)    # 대비를 0.9만큼 변경
    # cv2.imshow("constrast", constrast)      # 대비를 바꾼 영상 출력
    # big_size = set_size(frame, 2)    # 대비를 0.9만큼 변경
    # cv2.imshow("big_size", big_size)      # 대비를 바꾼 영상 출력
    if cv2.waitKey(1) == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()