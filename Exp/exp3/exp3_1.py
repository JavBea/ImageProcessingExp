# import math
# from scipy.integrate import quad
# import numpy as np
# from PIL import Image
# import cv2
#
#
# # sigma = 2
# # c = 12
#
#
# def gaussian_1d_function(t,sigma):
#     result = math.exp(0 - (t ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * math.pi * (sigma ** 2)))
#     return result
#
#
# def gaussian_1d_filter(c, sigma):
#     result, error = quad(
#         lambda t: c * math.exp(0 - (t ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * math.pi * (sigma ** 2))), -np.inf,
#         np.inf)
#     return result
#
#
# origin = cv2.imread(r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp3\exp3_1\a.jpg')
# shape = origin.shape
#
# out = np.zeros(shape)
#
# for i in range(shape[1]):
#     for j in range(shape[0]):
#         out[j, i] = [gaussian_1d_filter(origin[j, i , 0], 2),gaussian_1d_filter(origin[j, i , 1], 2),gaussian_1d_filter(origin[j, i , 2], 2)]
#         out[j, i] = out[j,i][::-1]
#         print(i,j)
#
# save_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp3\exp3_1\a_filtered.jpg'
# image = Image.fromarray(out.astype('uint8'))
# image.save(save_path)
# image.show()


# from PIL import Image, ImageFilter
#
# # 打开图像
# input_image_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp3\exp3_1\a.jpg'
# output_image_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp3\exp3_1\a_filtered.jpg'
# image = Image.open(input_image_path)
#
# # 在水平方向上进行一维高斯滤波
# horizontal_filtered_image = image.filter(ImageFilter.GaussianBlur(radius=0.75))
#
# # 在垂直方向上进行一维高斯滤波
# vertical_filtered_image = horizontal_filtered_image.transpose(Image.Transpose.ROTATE_90)
#
# image=vertical_filtered_image.transpose(4)
# # 保存处理后的图像
# image.save(output_image_path)
#
# print("图像处理完成并保存为a_filtered.png")


# from scipy import ndimage
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
#
# # 读取原始图像
# input_image_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp3\exp3_1\a.jpg'
# output_image_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp3\exp3_1\a_filtered.jpg'
# original_image = mpimg.imread(input_image_path)
#
# # 定义水平方向和垂直方向的标准差
# sigma_horizontal = 1.0
# sigma_vertical = 1.0
#
# # 对水平方向进行一维高斯滤波,axis=0表示对行进行处理
# filtered_horizontal = ndimage.gaussian_filter1d(original_image, sigma=sigma_horizontal, axis=1, mode='reflect')
#
# # 对垂直方向进行一维高斯滤波,axis=0表示对列进行处理
# filtered_image = ndimage.gaussian_filter1d(filtered_horizontal, sigma=sigma_vertical, axis=0, mode='reflect')
#
# # 保存滤波后的图像
# mpimg.imsave(output_image_path, filtered_image)
#
# # 显示原始图像和滤波后的图像（可选）
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(original_image)
# plt.title('Original Image')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(filtered_image)
# plt.title('Filtered Image')
# plt.axis('off')
#
# plt.show()
import numpy as np
from PIL import Image

# 加载图像
original_image = Image.open(r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp3\exp3_1\a.jpg')
original_array = np.array(original_image)

# # 定义高斯核函数
# def gaussian_kernel(sigma, size=3):
#     kernel = np.fromfunction(
#         lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
#                      np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
#         (size, size)
#     )
#     return kernel / np.sum(kernel)
#
# # 水平方向滤波
# horizontal_kernel = gaussian_kernel(sigma=1)  # 根据需要调整sigma值
# horizontal_filtered_image = np.apply_along_axis(lambda row: np.convolve(row, horizontal_kernel[1, :], mode='same'), axis=1, arr=original_array)
#
# # 垂直方向滤波
# vertical_kernel = gaussian_kernel(sigma=1)  # 根据需要调整sigma值
# vertical_filtered_image = np.apply_along_axis(lambda col: np.convolve(col, vertical_kernel[:, 1], mode='same'), axis=0, arr=horizontal_filtered_image)
#
# # 将滤波后的图像保存为a_filtered.png
# filtered_image = Image.fromarray(np.uint8(vertical_filtered_image))
# filtered_image.save(r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp3\exp3_1\a_filtered.jpg')
#
#
# 定义高斯核函数（包括归一化处理）
def gaussian_kernel(sigma, size=3):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

# 水平方向滤波
horizontal_kernel = gaussian_kernel(sigma=0.5)
horizontal_filtered_image = np.apply_along_axis(lambda row: np.convolve(row, horizontal_kernel[1, :], mode='same'), axis=1, arr=original_array)

# 垂直方向滤波
vertical_kernel = gaussian_kernel(sigma=0.5)
vertical_filtered_image = np.apply_along_axis(lambda col: np.convolve(col, vertical_kernel[:, 1], mode='same'), axis=0, arr=horizontal_filtered_image)

# 将滤波后的图像保存为a_filtered.png
filtered_image = Image.fromarray(np.uint8(vertical_filtered_image))
filtered_image.save(r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp3\exp3_1\a_filtered.jpg')

