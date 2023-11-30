import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read left and right stereo images
left_image = cv2.imread('../../resource/exp5/view1m.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('../../resource/exp5/view5m.png', cv2.IMREAD_GRAYSCALE)

# Create StereoSGBM object
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,
    blockSize=5,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2
)

# Compute disparity map
disparity = stereo.compute(left_image, right_image)

# Convert disparity map to depth map
depth_map = 1.0 / disparity

# Display disparity map and depth map
plt.subplot(1, 2, 1)
plt.imshow(disparity, cmap='jet')
plt.title('Disparity Map')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(depth_map, cmap='jet')
plt.title('Depth Map')
plt.colorbar()

plt.show()

# Display 3D view of the depth map
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

rows, cols = depth_map.shape
x = np.arange(0, cols, 1)
y = np.arange(0, rows, 1)
x, y = np.meshgrid(x, y)

ax.plot_surface(x, y, depth_map, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth')

plt.show()
