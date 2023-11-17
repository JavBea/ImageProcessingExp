
from PIL import Image
import numpy as np
import cv2
import exp2_2_demapping
from ImageRotator import ImageRotator
from exp2_1 import zoom
import math


# 中心归一化坐标
def centralize(x, y, w, h):
    result = np.zeros(2)
    result[0] = (x - 0.5 * w) / (0.5 * w)
    result[1] = (y - 0.5 * h) / (0.5 * h)
    return result


# 反中心归一化坐标
def decentralize(x, y, w, h):
    result = np.zeros(2)
    result[0] = (x + 1) * 0.5 * w
    result[1] = (y + 1) * 0.5 * h
    return result


def crop_2d_list(input_list):
    # 获取最短的一维列表长度
    min_length = min(len(sublist) for sublist in input_list)

    # 裁剪每个一维列表为相同的长度
    output_list = [sublist[:min_length] for sublist in input_list]

    return output_list


origin_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp2\lab2.png'
origin = cv2.imread(origin_path)

shape = origin.shape

# 输出数组
out = []
for i in range(shape[1]):
    out_row = []
    for j in range(shape[0]):
        x, y = centralize(i, j, shape[1], shape[0])
        real_x, real_y = exp2_2_demapping.demapping(x, y)

        decentralized = decentralize(real_x, real_y, shape[0], shape[1])

        if 0 <= decentralized[1] < shape[0] and 0 <= decentralized[0] < shape[1]:
            out_row.append(origin[int(decentralized[1]), int(decentralized[0])][::-1])
    out.append(out_row)

out = crop_2d_list(out)

# for row in out:
#     print(len(row))
# 创建NumPy数组并转换数据类型
out_array = np.array(out, dtype=np.uint8)

# 创建Image对象
image = Image.fromarray(out_array)

# 保存图像
save_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp2\transformed.png'
image.save(save_path)
rotator = ImageRotator()
rotator.rotate_image(save_path, save_path, 90)

newShape = cv2.imread(save_path).shape
zoom(save_path, save_path, shape[1] / newShape[1], shape[0] / newShape[0])
