# import math
# import numpy as np
#
# from numpy import double
#
#
# def skin(y, cb, cr):
#     a = 25.39
#     b = 14.03
#     ecx = 1.60
#     ecy = 2.41
#     sita = 2.53
#     cx = 109.38
#     cy = 152.02
#
#     coe = [[math.cos(sita), math.sin(sita)], [-math.sin(sita), math.cos(sita)]]
#
#     if y > 230:
#         a = 1.1 * a
#         b = 1.1 * b
#     cb = double(cb)
#     cr = double(cr)
#     t = [[cb - cx], [cr - cy]]
#     temp = np.dot(coe, t)
#     # temp = [coe * double(t[0]), coe * double(t[1])]
#     value = (temp[0][0] - ecx) ** 2 / (a ** 2) + (temp[1][0] - ecy) ** 2 / (b ** 2)
#
#     if value > 1:
#         return 0
#     else:
#         return 1
#     pass
#
#
# if __name__ == "__main__":
#     print(skin(80, 30, 90))
#     pass


import numpy as np

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
    value = (temp[0] - ecx)**2 / a**2 + (temp[1] - ecy)**2 / b**2

    # 大于 1 则不是肤色，返回 0；否则为肤色，返回 1
    if value > 1:
        result = 0
    else:
        result = 1

    return result
