
import cv2
import numpy as np

# 색감 변경 함수
def color_filter(img, color, scale):    # 입력변수는 영상, 변경할 색상, 변경할 비율
    dst = np.array(img, np.uint8)       # 입력 영상과 같은 영상을 복사
    if color == 'blue' or color == 0:   # 색상이 파란색이라면
        dst[:, :, 0] = cv2.multiply(dst[:, :, 0], scale)    # 영상중 파란색의 비율을 바꿔줌
    elif color == 'green' or color == 1:    # 색상이 초록색이라면
        dst[:, :, 1] = cv2.multiply(dst[:, :, 1], scale)    # 영상중 초록색의 비율을 바꿔줌
    if color == 'red' or color == 2:    # 색상이 빨간색이라면
        dst[:, :, 2] = cv2.multiply(dst[:, :, 2], scale)    # 영상중 빨간색의 비율을 바꿔줌
    return dst      # 처리된 영상을 반환

# 밝기 변경 함수
def set_brightness(img, scale):         # 입력변수는 영상, 밝기를 변경할 값
    return cv2.add(img, scale)          # 전체 픽셀값에 scale을 더하여 밝기를 변경

# 대비 변경 함수
def set_contrast(img, scale):           # 입력변수는 영상, 대비를 변경할 값
    return np.uint8(np.clip((1 + scale) * img - 128 * scale, 0, 255))   # 대비를 scale 비율로 변환

def set_size(img, scale):
    return cv2.resize(img, dsize=(int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)