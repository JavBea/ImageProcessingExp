import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D


# 这是基于论文实验的复刻，论文名为：Obtaining Depth Maps From Color Images By Region Based Stereo Matching Algorithms

def disparity(imgLeft, imgRight, windowSize=(3, 3), dMax=40, alpha=1):
    """
    论文中的
    "a) Global Error Energy Minimization by Smoothing Functions"

    :param imgLeft: 左图
    :param imgRight: 右图
    :param windowSize: (n, m)，窗口的尺寸为 n x m
    :param dMax: 中值滤波迭代次数，默认为30
    :param alpha: 阈值系数，默认为1
    :return: 视差图，可靠差异的视差图，平均误差能量矩阵，函数运行时间
    """
    timeBegin = cv.getTickCount()  # 记录开始时间

    n, m = windowSize  # 窗口大小
    rows, cols, channels = imgLeft.shape  # 该实验中 imgLeft 和 imgRight 的 shape 是一样的
    cols = 190
    errorEnergyMatrixD = np.zeros((rows, cols, dMax), dtype=float)  # 误差能量矩阵（共dMax层），方便后续计算
    errorEnergyMatrixAvgD = np.zeros((rows, cols, dMax), dtype=float)  # 平均误差能量矩阵（共dMax层），方便后续计算
    imgDisparity = np.zeros((rows, cols), dtype=np.uint8)  # 具有可靠差异的视差图，将作为结果返回

    # 计算误差能量矩阵 errorEnergyMatrix，平均误差能量矩阵 errorEnergyMatrixAvg
    # 我在处理的时候，计算的是x从i到i+n-1，y从j到j+m-1，保持水平为x，竖直为y，这样方便后续计算

    # 先padding，这样方便使用numpy加速计算
    # 0 , 0: 在图像的上部和下部分别添加0行
    # n-1, m-1+dMax: 在左侧和右侧分别添加 n-1 列和 m-1+dMax 列
    # borderType=cv.BORDER_REPLICATE: 使用边界复制方式进行填充，即将边缘像素进行复制。
    # 这样一来所有的像素点包括边界像素点就可以完全被包裹在一个n*m的窗口中了，避免了可能的越界问题
    imgLeftPlus = cv.copyMakeBorder(imgLeft, 0, 0, n - 1, m - 1 + dMax, borderType=cv.BORDER_REPLICATE)
    imgRightPlus = cv.copyMakeBorder(imgRight, 0, 0, n - 1, m - 1, borderType=cv.BORDER_REPLICATE)

    # 迭代 dMax 次，dMax是最大迭代次数
    # d代表视差，这个位移是左右两个图像之间相同物体的位移，也就是说，左图像中的某个物体，右图像中的对应物体的位移
    for d in range(dMax):
        # 对整个图像进行遍历
        for i in range(rows):
            for j in range(cols):
                # 对于每个 (i, j, d) 根据公式(1)计算误差能量矩阵
                errorEnergy = (imgLeftPlus[i:i + n, j + d:j + m + d, ...] - imgRightPlus[i:i + n, j:j + m, ...]) ** 2
                # 将所有的元素相加，除以3*n*m，因为有3个通道，且是 n x m 的窗口
                errorEnergyMatrixD[i, j, d] = np.sum(errorEnergy) / (3 * n * m)
        # 对 errorEnergyMatrix 进行遍历
        for i in range(rows):
            for j in range(cols):
                # 对于每个 (i, j, d) 根据公式(2)计算平均误差能量矩阵
                errorEnergyMatrixAvgD[i, j, d] = np.sum(errorEnergyMatrixD[i:i + n, j:j + m, d]) / (n * m)
        # 论文中说到了（公式(1)下方）
        # For a predetermined disparity
        # search range (w), every e(i, j, d) matrix respect to disparity is smoothed by applying
        # averaging filter many times. (See Figure 1.b)
        # 也就是对于每个e(i, j, d)，进行多次平均滤波。在这里我选择执行3次。
        for k in range(3):
            for i in range(rows):
                for j in range(cols):
                    # 对于每个 (i, j, d) 根据公式(2)计算平均误差能量矩阵
                    # 下面i+n和j+m越了界也是没有问题，因为已经提前填充了，切片会正常计算
                    errorEnergyMatrixAvgD[i, j, d] = np.sum(errorEnergyMatrixAvgD[i:i + n, j:j + m, d]) / (n * m)

    # 按照论文中的算法，需要找到所有d层中最小的那个,axis=2代表第三个维度，也就是d这个维度
    errorEnergyMatrixAvg = np.min(errorEnergyMatrixAvgD, axis=2)  # 平均误差能量矩阵（最终的，只有一层）
    # argmin返回的是最小值的索引，因此imgDisparity中的每个元素都是0~dMax-1之间的整数
    imgDisparity[:, :] = np.argmin(errorEnergyMatrixAvgD, axis=2)  # 视差图
    imgOrignal = imgDisparity.copy()  # 保留一份，并作为结果返回

    # 下面的部分我们实现论文中的：（包含公式 5、6、7、8、9）
    # 可靠差异的视差图
    # "Filtering Unreliable Disparity Estimation By Average Error Thresholding Mechanism"

    # 阈值ve，取平均值乘以alpha
    Ve = alpha * np.mean(imgDisparity)  # 计算Ve
    # 创建了一个副本
    temp = errorEnergyMatrixAvg.copy()
    # 将temp中小于Ve的元素设置为0，大于等于Ve的元素设置为1
    temp[temp > Ve] = 0
    temp[temp != 0] = 1

    # 将备份中的数据类型转换为int，然后与imgDisparity相乘，得到可靠差异的视差图
    temp = temp.astype(int)
    imgDisparity = np.multiply(imgDisparity, temp).astype(np.uint8)

    timeEnd = cv.getTickCount()  # 记录结束时间
    time = (timeEnd - timeBegin) / cv.getTickFrequency()  # 计算总时间

    return imgOrignal, imgDisparity, errorEnergyMatrixAvg, time


def depthGeneration(imgDisparity, f=30, T=20, coff=0.0):
    """
    实现论文中的"Depth Map Generation From Disparity Map"
    根据视差图，实现深度图

    :param imgDisparity: 具有可靠差异的视差图
    :param f: 焦距
    :param T: 间距
    :return: 深度图
    """
    # 实现公式(4)

    # 获取一个和imgDisparity一样大小的全0矩阵
    rows, cols = imgDisparity.shape
    imgDepth = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            # 避免除0错误，如果imgDisparity[i, j]为0，那么imgDepth[i, j]也为0
            if imgDisparity[i, j] == 0:
                imgDepth[i, j] = 0
            else:
                # f是焦距，T是间距，这里的公式实现了公式4，f和T的值是论文中的推荐值
                imgDepth[i, j] = f * T // imgDisparity[i, j]

    # 如果系数为0，不进行抹平操作
    if coff == 0:
        return imgDepth

    # 进行z系数抹平操作
    std = np.std(imgDepth)
    mean = np.mean(imgDepth)

    for i in range(rows):
        for j in range(cols):
            z = (imgDepth[i, j] - mean) / std
            # 设置阈值，例如 Z 分数超过 3 的值被认为是异常值
            if np.abs(z) > coff:
                imgDepth[i, j] = 0

    return imgDepth


def show3d(imgDepth):
    # Display 3D view of the depth map
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rows, cols = imgDepth.shape
    x = np.arange(0, cols, 1)
    y = np.arange(0, rows, 1)
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, y, imgDepth, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')

    plt.show()


def show2d(target, title=""):
    plt.figure()
    plt.imshow(target, cmap='gray', vmin=0, vmax=40)
    plt.title(title)  # 设置标题
    ax = plt.gca()  # 返回坐标轴实例
    x_major_locator = MultipleLocator(20)  # 坐标刻度间隔为20
    y_major_locator = MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)  # 设置坐标间隔
    ax.yaxis.set_major_locator(y_major_locator)
    plt.colorbar()  # 设置灰度颜色条，用于标明不同灰度值对应的颜色
    plt.show()


# 读取图片
imgLeft = cv.imread('../../resource/exp5/view1m.png')
imgRight = cv.imread('../../resource/exp5/view5m.png')
# 得到视差图imgOriginal，可靠差异的视差图imgDisparity, 平均误差矩阵errorEnergyMatrixAvg
imgOrignal, imgDisparity, errorEnergyMatrixAvg, time = disparity(imgLeft, imgRight)
# print(time)  # 打印函数运行时间
# 由视差图得到深度图
imgDepth = depthGeneration(imgDisparity, 30, 20)
imgDepth1 = depthGeneration(imgDisparity, 30, 20, 2)
imgDepth2 = depthGeneration(imgDisparity, 30, 20, 1.5)

# 下面绘制三幅图像
# 视差图图像
show2d(imgOrignal,'disparity figure')
# 平均误差能量矩阵图像
show2d(errorEnergyMatrixAvg,'errorEnergyMatrixAvg')

# 绘制3D图像
show3d(imgDepth)
show3d(imgDepth1)
show3d(imgDepth2)

# plt.savefig('../../resource/exp5/3D figure.png', bbox_inches='tight', pad_inches=0.1)
cv.waitKey(0)
cv.destroyAllWindows()
