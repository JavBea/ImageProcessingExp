import cv2
import numpy as np


# 加载人脸检测器
def face_detect(input, output, scale_factor_x=1.0, scale_factor_y=1.0):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 加载图像
    img = cv2.imread(input)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 按指定的缩放系数修改识别框的大小
        # 新的宽度和高度
        new_w = int(w * scale_factor_x)
        new_h = int(h * scale_factor_y)

        # 计算新的右下角坐标
        x_new = x - int((new_w - w) / 2)
        y_new = y - int((new_h - h) / 2)

        # 图像上标记检测到的人脸
        cv2.rectangle(img, (x_new, y_new), (x_new + new_w, y_new + new_h), (255, 0, 0), 2)

    # 保存结果到指定路径
    cv2.imwrite(output, img)

    # 显示结果图像
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input1 = r"../../resource/exp4/Orical1.jpg"
    input2 = r"../../resource/exp4/Orical2.jpg"
    output1 = r"../../resource/exp4/Orical1_detected.jpg"
    output2 = r"../../resource/exp4/Orical2_detected.jpg"

    face_detect(input1, output1, 1.0, 1.2)
    face_detect(input2, output2, 1.0, 1.2)
