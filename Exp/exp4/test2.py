# import cv2
# from scipy.ndimage import binary_erosion, binary_dilation
#
# from skin import skin
# import numpy as np
#
#
# def facedetect(input, output):
#     origin = cv2.imread(input)
#
#     # 转换为灰度图像
#     gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
#     # 转换为YCbCr颜色空间
#     ycbcr = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
#
#     height = origin.shape[0]
#     width = origin.shape[1]
#
#     for i in range(height):
#         for j in range(width):
#
#             y = ycbcr[i, j, 0]
#             cb = ycbcr[i, j, 1]
#             cr = ycbcr[i, j, 2]
#
#             if y < 80:
#                 gray[i, j] = 0
#             else:
#                 if skin(y, cb, cr) == 1:
#                     gray[i, j] = 255
#                 else:
#                     gray[i, j] = 0
#
#     # 创建一个结构元素，这里使用一个5x5的矩形
#     se = np.ones((5, 5), dtype=np.uint8)
#
#     # 进行腐蚀操作
#     gray = binary_erosion(gray, structure=se)
#
#     # 进行膨胀操作
#     gray = binary_dilation(gray, structure=se)
#
#     # 显示图像（这里使用matplotlib库，确保安装）
#     import matplotlib.pyplot as plt
#
#     plt.imshow(gray, cmap='gray')
#     plt.title('Processed Image')
#     plt.show()
#
#
# if __name__ == "__main__":
#     origin = r"../../resource/exp4/Orical1.jpg"
#     out=r"../../resource/exp4/Orical1_grayed.jpg"
#
#     facedetect(origin,out)


import numpy as np
import cv2


def skin(Y, Cb, Cr):
    # 参数定义
    a = 25.39
    b = 14.03
    ecx = 1.60
    ecy = 2.41
    sita = 2.53
    cx = 109.38
    cy = 152.02

    # 旋转矩阵
    xishu = np.array([[np.cos(sita), np.sin(sita)], [-np.sin(sita), np.cos(sita)]])

    # 如果亮度大于 230，则将长短轴同时扩大为原来的 1.1 倍
    if Y > 230:
        a *= 1.1
        b *= 1.1

    # 根据公式进行计算
    Cb = np.double(Cb)
    Cr = np.double(Cr)
    t = np.array([(Cb - cx), (Cr - cy)])
    temp = np.dot(xishu, t)
    value = (temp[0] - ecx) ** 2 / a ** 2 + (temp[1] - ecy) ** 2 / b ** 2

    # 大于 1 则不是肤色，返回 0；否则为肤色，返回 1
    if value > 1:
        result = 0
    else:
        result = 1

    return result


def findeye(bImage, x, y, w, h):
    part = np.zeros((h, w), dtype=np.uint8)

    for i in range(y, y + h):
        for j in range(x, x + w):
            if bImage[i, j] == 0:
                part[i - y, j - x] = 255
            else:
                part[i - y, j - x] = 0

    labeled_part, num = cv2.connectedComponents(part)

    if num < 2:
        eye = 0
    else:
        eye = 1

    return eye


# 读入原始图像
I = cv2.imread(r'../../resource/exp4/Orical1.jpg')

# 获取灰度图
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# 将图像转化为 YCbCr 空间
ycbcr = cv2.cvtColor(I, cv2.COLOR_BGR2YCrCb)

# 读取图像尺寸
height, width = gray.shape

# 利用肤色模型二值化图像
for i in range(height):
    for j in range(width):
        # y 代表亮度
        Y = ycbcr[i, j, 0]
        # cb 代表蓝色浓度的偏移性
        Cb = ycbcr[i, j, 1]
        # cr 代表红色浓度的偏移性
        Cr = ycbcr[i, j, 2]

        # 当亮度小于80时，将对应坐标的灰度图的值赋为0
        if Y < 80:
            gray[i, j] = 0
        else:
            # 根据色彩模型进行图像二值化
            if skin(Y, Cb, Cr) == 1:
                # 如果判断为肤色，灰度值赋为255；否则赋值为0
                gray[i, j] = 255
            else:
                gray[i, j] = 0

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)

cv2.imshow('Result', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 采用标记方法选出图中的白色区域
# _, labeled = cv2.connectedComponents(gray)
#
# # 度量区域属性
# stats = cv2.connectedComponentsWithStats(labeled, connectivity=8)
# 二值化图像
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

# 使用 connectedComponentsWithStats
stats = cv2.connectedComponentsWithStats(binary, connectivity=8)

n = 1
result = np.zeros((n, 4), dtype=np.int32)

for i in range(1, stats[0]):
    box = stats[3][i]
    x, y, w, h = box[0], box[1], box[2], box[3]

    ratio = h / w

    ux = int(x)
    uy = int(y)

    if ux > 1:
        ux = ux - 1

    if uy > 1:
        uy = uy - 1

    if w < 20 or h < 20 or w * h < 400:
        continue
    elif 0.6 < ratio < 2 and findeye(gray, ux, uy, w, h) == 1:
        result[n - 1, :] = [ux, uy, w, h]
        n += 1

if result.shape[0] == 1 and result[0, 0] > 0:
    cv2.rectangle(I, (result[0, 0], result[0, 1]), (result[0, 0] + result[0, 2], result[0, 1] + result[0, 3]),
                  (0, 0, 255), 2)
else:
    a = 0
    arr1 = []
    arr2 = []

    for m in range(result.shape[0]):
        m1, m2, m3, m4 = result[m, 0], result[m, 1], result[m, 2], result[m, 3]

        if m1 + m3 < width and m2 + m4 < height and m3 < 0.2 * width:
            a += 1
            arr1.append(m3)
            arr2.append(m4)

    arr3 = sorted(arr1)[0]
    arr4 = sorted(arr2)[0]

    for m in range(result.shape[0]):
        m1, m2, m3, m4 = result[m, 0], result[m, 1], result[m, 2], result[m, 3]

        if m1 + m3 < width and m2 + m4 < height and m3 < 0.2 * width:
            m3 = arr3
            m4 = arr4
            cv2.rectangle(I, (m1, m2), (m1 + m3, m2 + m4), (0, 0, 255), 2)

cv2.imshow('Final Result', I)
cv2.waitKey(0)
cv2.destroyAllWindows()
