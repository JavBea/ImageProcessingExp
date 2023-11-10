import cv2
import numpy as np
from PIL import Image


# 放缩方法，参数分别为图片路径、输出路径、横向放缩参数、纵向放缩参数
def zoom(in_path, out_path, x_scale_factor, y_scale_factor):
    try:
        # 尝试读取图像文件
        origin = cv2.imread(in_path)

        # 如果图像读取成功，进行相应的操作
        if origin is not None:
            # 进行图像处理等操作
            pass
        else:
            print("Failed to read the image. Image is None.")
            return

    except cv2.error as e:
        # 捕获cv2.error异常（文件未找到等错误）
        print("Error: 未找到图片")
        return

    # 取得图片大小
    shape = origin.shape

    # 新建三维数组
    y_len = shape[0] * y_scale_factor
    x_len = shape[1] * x_scale_factor
    out = np.zeros((int(y_len), int(x_len), 3))

    for i in range(int(x_len - x_scale_factor)):
        for j in range(int(y_len - y_scale_factor)):
            # 理论上对应的像素坐标
            real_i = i / x_scale_factor
            real_j = j / y_scale_factor

            # 取小数，代表在两个像素点之间的偏向，用于加权
            real_i_fraction = real_i % 1
            real_j_fraction = real_j % 1

            # 取整数，即最靠近的像素点，在此像素点旁另取三个像素点，组成四个参考点
            real_i_integer = int(real_i - real_i_fraction)
            real_j_integer = int(real_j - real_j_fraction)

            # 四个参考点：
            # （x0,y0）   （x0+1,y0）
            # （x0,y0+1） （x0+1,y0+1）
            # 对四个参考点的BGR进行加权并计算
            out[j, i] = (1 - real_i_fraction) * (1 - real_j_fraction) * origin[real_j_integer, real_i_integer] \
                        + (1 - real_i_fraction) * real_j_fraction * origin[real_j_integer + 1, real_i_integer] \
                        + real_i_fraction * (1 - real_j_fraction) * origin[real_j_integer, real_i_integer + 1] \
                        + real_i_fraction * real_j_fraction * origin[real_j_integer + 1, real_i_integer + 1]

            # 将BGR进行切片反转得到RGB
            out[j, i] = out[j, i][::-1]

    # 转为8进制串的格式
    image = Image.fromarray(out.astype('uint8'))

    # 保存

    image.save(out_path)

    image.show()


if __name__ == "__main__":
    input = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp2\lab2.png'
    output = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp2\double.png'
    zoom(input, output, 0.5, 2)
