import cv2
import math
import numpy as np
from PIL import Image


def jbf(in_path, out_path, w, sigma_f, sigma_g):
    origin = cv2.imread(in_path)

    # 由原图得到下采样图像
    sampled = cv2.resize(origin, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # 在刚才的基础上得到上采样图像
    sampled = cv2.resize(sampled, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    size = origin.shape

    result = np.zeros(size)

    # w值一半的向下取整
    hw_floor = math.floor(w / 2)
    # w值一半的向上取整
    hw_ceil = math.ceil(w / 2)

    for i in range(hw_ceil, size[1] - hw_ceil):
        for j in range(hw_ceil, size[0] - hw_ceil):
            # BGR分子
            numerator = [0, 0, 0]
            # BGR分母
            denominator = [0, 0, 0]

            for m in range(i - hw_floor, i+hw_floor):
                for n in range(j - hw_floor, j+hw_floor):
                    for r in range(3):
                        temp = f(math.sqrt((i - m) ** 2 + (j - n) ** 2), sigma_f) * g(
                            abs(sampled[j, i, r] - sampled[n, m, r]), sigma_g)
                        numerator[r] += temp * sampled[n, m, r]
                        denominator[r] += temp

            result[j, i] = [numerator[2] / denominator[2], numerator[1] / denominator[1], numerator[0] / denominator[0]]

    image = Image.fromarray(result.astype("uint8"))
    image.save(out_path)
    image.show()


def f(df, sigma_f):
    return math.exp((-df ** 2) / (2 * sigma_f ** 2))


def g(dg, sigma_g):
    return math.exp((-dg ** 2) / (2 * sigma_g ** 2))


if __name__ == "__main__":
    ori = r"../../resource/exp3/exp3_2/2_2.png"
    out = r"../../resource/exp3/exp3_2/result.png"
    jbf(ori, out, 3, 1, 1)
    # image = cv2.imread(ori)
