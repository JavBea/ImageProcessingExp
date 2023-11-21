import numpy as np
import cv2
import math


def filter_generator(size, sigma, dimension):
    """
    生成滤波器内核
    :param size:滤波器窗口大小
    :param sigma:标准差，自由指定
    :param dimension:维度数，指定生成一维或二维
    :return: 一维返回两个内核，二维返回一个
    """
    if dimension == 1:
        """
        生成一维高斯核
        """
        # 水平方向的一维高斯内核
        filter_1d_horizontal = np.zeros(size)
        for i in range(size):
            filter_1d_horizontal[i] = gauss_1d(i - int(size / 2), 0, sigma)
        # 转置，得到垂直方向的一维高斯内核
        filter_1d_vertical = np.transpose(filter_1d_horizontal)
        # 返回两个高斯内核
        return filter_1d_horizontal, filter_1d_vertical
    elif dimension == 2:
        """
        生成二维高斯核
        """
        filter_2d = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_2d[i, j] = gauss_2d(i - int(size / 2), j - int(size / 2), sigma)
        return filter_2d


def gauss_1d(x, x_mean, sigma):
    """
    生成一维滤波器的方法
    :param x: 输入值
    :param x_mean: x平均值，因为中一化，故调用时取零
    :param sigma: x标准差，自由指定
    :return: 返回一个f(x)值，即一维高斯函数下的值
    """
    return 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp((-(x - x_mean) ** 2) / (2 * (sigma ** 2)))
    # return int(1 / (sigma * math.sqrt(2 * math.pi)) * math.exp((-(x - x_mean) ** 2) / (2 * (sigma ** 2))) * 273)


def gauss_2d(x, y, sigma):
    """
    生成二维滤波器的方法（未用到）
    :param x: x坐标
    :param y: y坐标
    :param sigma: 标准差，自由指定
    :return: 返回一个f(x,y)值，即二维高斯函数下的值
    """
    return 1 / (2 * math.pi * (sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    # return int(1 / (2 * math.pi * (sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2))) * 273)


def gaussian(input, output, sigma):
    """
    高斯滤波主方法，主要采用水平与垂直分离的方法
    :param input: 输入图像路径
    :param output: 输出图像路径
    :param sigma: 指定的标准差
    :return: NONE
    """
    # 读图片
    origin = cv2.imread(input)

    # 计算大小
    size = math.floor((6 * sigma - 1) / 2) * 2 + 1

    # 声明一个存储输出图像的三维数组
    shape = origin.shape
    result = np.zeros(shape)

    # 得到一维高斯内核
    filter_horizon, filter_vertical = filter_generator(size, sigma, 1)

    # 由于是RGB图像，所以需要遍历三个颜色通道
    for color in range(3):
        for i in range(shape[0]):
            # ”：“代表切片操作，此处即代表取每一行的color通道
            result[i, :, color] = np.convolve(origin[i, :, color], filter_horizon, mode='same')
    for color in range(3):
        for j in range(shape[1]):
            # ”：“代表切片操作，此处即代表取每一列的color通道
            result[:, j, color] = np.convolve(origin[:, j, color], filter_vertical, mode='same')

    # 输出保存
    cv2.imwrite(output, result)


if __name__ == "__main__":
    input_path = r"../../resource/exp3/exp3_1/a.jpg"
    output_path = r"../../resource/exp3/exp3_1/a_filtered.jpg"
    gaussian(input_path, output_path, 1.3)
    pass
