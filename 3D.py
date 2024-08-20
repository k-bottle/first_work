import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
left_image = cv2.imread('left/1.bmp', 0)  # 灰度图像
right_image = cv2.imread('right/1.bmp', 0)  # 灰度图像

# 相机内参和畸变系数
K1 = np.array([[2612.454502, 0, 978.9],
               [0, 2546.7, 546.2],
               [0, 0, 1]])

K2 = np.array([[3244.242486, 0, 1461.4],
               [0, 3156.2, 553.0],
               [0, 0, 1]])

D1 = np.array([-0.075342, 0.247985, 0.000617, 0.000148, -2.277605])
D2 = np.array([-0.097084, -0.142248, -0.000019, -0.018069, 0.835860])

# 畸变校正
h, w = left_image.shape[:2]
new_camera_matrix1, roi1 = cv2.getOptimalNewCameraMatrix(K1, D1, (w, h), 1, (w, h))
left_image_undistorted = cv2.undistort(left_image, K1, D1, None, new_camera_matrix1)

new_camera_matrix2, roi2 = cv2.getOptimalNewCameraMatrix(K2, D2, (w, h), 1, (w, h))
right_image_undistorted = cv2.undistort(right_image, K2, D2, None, new_camera_matrix2)

# 创建立体匹配对象
stereo = cv2.StereoBM_create(numDisparities=16*8, blockSize=15)

# 计算视差图
disparity = stereo.compute(left_image_undistorted, right_image_undistorted).astype(np.float32) / 16.0

# 视差图的后处理
disparity = cv2.normalize(disparity, disparity, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# 可视化视差图
plt.imshow(disparity, 'gray')
plt.title('Disparity Map')
plt.show()

# 焦距 (fx) 和基线长度 (Tx)
fx = 2928.348494
baseline = 3715.881505  # 基线长度

# 计算深度图
depth_map = np.zeros_like(disparity, dtype=np.float32)
valid_mask = disparity > 0  # 仅对有效视差值进行计算
depth_map[valid_mask] = fx * baseline / disparity[valid_mask]

# 假设我们有一个乒乓球商标的图像坐标 (u_L, v_L) 和 (u_R, v_R)
u_L = 100  # 左相机 x 坐标
v_L = 200  # 左相机 y 坐标
u_R = 90   # 右相机 x 坐标
v_R = 200  # 右相机 y 坐标 (理论上应与左相机的 y 坐标相同，但实际可能会有偏差)

# 获取该像素点的视差值
disparity_value = u_L - u_R

# 获取该像素点的深度值
depth = fx * baseline / (disparity_value + 1e-6)

# 左相机的光学中心 (cx, cy)
cx = 978.9
cy = 546.2

# 计算三维坐标 (X, Y, Z)
X = (u_L - cx) * depth / fx
Y = (v_L - cy) * depth / fx
Z = depth

# 输出左相机坐标系中的三维坐标
print("3D coordinates in left camera coordinate system (X, Y, Z):", X, Y, Z)

# 左相机坐标系中的三维点 P_L
P_L = np.array([X, Y, Z])

# 旋转矩阵 R 和平移向量 T
R = np.array([[ 0.0077, -0.0354, -0.9993],
              [ 0.0324,  0.9989, -0.0352],
              [ 0.9994, -0.0321,  0.0088]])

T = np.array([-3046.3, -2.0, 2127.8])

# 计算右相机坐标系中的三维点 P_R
P_R = R.dot(P_L) + T

# 输出右相机坐标系中的三维坐标
print("3D coordinates in right camera coordinate system (X, Y, Z):", P_R)
