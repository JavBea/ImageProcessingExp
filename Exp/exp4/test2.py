import cv2
import numpy as np


def custom_face_detection(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用阈值进行二值化处理
    _, thresholded = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # 进行形态学操作，填充小孔和平滑边缘
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始图像上标记检测到的人脸
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 显示结果图像
    cv2.imshow('Custom Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 调用自定义人脸检测方法
custom_face_detection('../../resource/exp4/Orical1.jpg')
