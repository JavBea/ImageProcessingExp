# import numpy as np
# from scipy.ndimage import label
#
#
# def findeye(b_image, x, y, w, h):
#     part = np.zeros(w, h)
#
#     # 二值化
#     for i in range(y, y + h):
#         for j in range(x, x + w):
#             if b_image[i, j] == 0:
#                 b_image[i - y + 1, j - x + 1] = 255
#             else:
#                 part[i - y + 1, j - x + 1] = 0
#
#     L, num = label(part, structure=np.ones((3, 3)))
#
#     if num < 2:
#         return 0
#     else:
#         return 1


import numpy as np
from scipy.ndimage import label

def findeye(bImage, x, y, w, h):
    # 创建一个与矩形相同大小的零矩阵
    part = np.zeros((h, w), dtype=np.uint8)

    # 二值化
    for i in range(y, y + h):
        for j in range(x, x + w):
            if bImage[i, j] == 0:
                part[i - y, j - x] = 255
            else:
                part[i - y, j - x] = 0

    # 使用label函数进行连通区域标记
    labeled_part, num = label(part, structure=np.ones((3, 3), dtype=np.int))

    # 如果区域中有两个以上的矩形则认为有眼睛
    if num < 2:
        eye = 0
    else:
        eye = 1

    return eye
