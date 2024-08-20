import os
from initialization_data import *
import numpy as np

from detect import detect, set_logging, select_device, attempt_load, check_img_size, LoadImages
import torch
from pathlib import Path
import torch.backends.cudnn as cudnn
import cv2
import warnings
import csv

class Opt:
    pass

# 定义两个三维坐标的容器来存储坐标
ball_container = []
contour_container=[]

# 相机内参和畸变系数
K1 = np.array([[2612.454502, 0, 978.9],
               [0, 2546.7, 546.2],
               [0, 0, 1]])

K2 = np.array([[3244.242486, 0, 1461.4],
               [0, 3156.2, 553.0],
               [0, 0, 1]])

D1 = np.array([-0.075342, 0.247985, 0.000617, 0.000148, -2.277605])
D2 = np.array([-0.097084, -0.142248, -0.000019, -0.018069, 0.835860])
baseline = 3715.881505  # 基线长度，单位毫米

fx = 2928.348494  # 焦距，单位像素

def calculate_3D_coordinates(ball_L, ball_R):
    """
    将左右相机图像坐标转换为三维空间坐标
    :param ball_L: 左相机图像中的点坐标 (u_L, v_L)
    :param ball_R: 右相机图像中的点坐标 (u_R, v_R)
    :return: 三维空间坐标 (X, Y, Z)
    """
    # 计算视差
    disparity = ball_L[0] - ball_R[0]

    # 计算深度 Z
    if disparity != 0:
        Z = fx * baseline / disparity
    else:
        Z = np.inf  # 避免除零错误

    # 左相机的光学中心 (cx, cy)
    cx_L = K1[0, 2]
    cy_L = K1[1, 2]

    # 计算 X 和 Y 坐标
    X = (ball_L[0] - cx_L) * Z / fx
    Y = (ball_L[1] - cy_L) * Z / fx

    return np.array([X, Y, Z])

def process_real_number(data):
    # 检查数据列表中是否有连续三个数值的序列
    if len(data) >= 3:
        for i in range(len(data) - 2):
            if data[i] + 1 == data[i + 1] and data[i + 1] + 1 == data[i + 2]:
                middle_number = data[i + 1]  # 提取连续三个数值的中间值
                real_number.append(middle_number)
                print(f"最准确的数值为：{middle_number}")
                return middle_number

# 卡尔曼滤波进行预测
import numpy as np

class KalmanFilter:
    def __init__(self, dt=1, process_noise=1e-5, measurement_noise=1e-2):
        # 时间间隔
        self.dt = dt

        # 初始状态 (x, y, dx, dy)
        self.x = np.zeros(4)

        # 状态转移矩阵 A
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # 观测矩阵 H
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # 过程噪声协方差矩阵 Q
        self.Q = process_noise * np.eye(4)

        # 观测噪声协方差矩阵 R
        self.R = measurement_noise * np.eye(2)

        # 初始估计误差协方差矩阵 P
        self.P = np.eye(4)

    def predict(self):
        # 预测状态
        self.x = np.dot(self.A, self.x)

        # 预测估计误差协方差
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        # 返回预测的二维坐标
        return self.x[:2]

    def update(self, z):
        # 计算卡尔曼增益 K
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新状态向量 x
        y = z - np.dot(self.H, self.x)  # 计算残差
        self.x += np.dot(K, y)

        # 更新估计误差协方差 P
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        # 返回更新后的二维坐标
        return self.x[:2]



# 二维旋转计算  frame_rate为相机帧率 total_frames为总图片数  rotation_frames为转一圈所花费的图片数
def calculate_rotation_rate(rotation_frames):
    image_time = 1 / frame_rate  # 图片代表的时间，单位为秒
    rotation_time = (rotation_frames / total_frames) * image_time  # 乒乓球旋转一周的时间，单位为秒
    if rotation_time == 0:  # 处理除零情况，避免计算错误
        return 0
    rotation_rate = 1 / rotation_time  # 乒乓球每秒转动的圈数
    return rotation_rate

# 三维旋转计算  可以求某段轨迹的平均角速度，也可以求相邻的角速度
def calculate_rotation_speed(ball_centers, contour_centers, frame_rate):

    angular_velocities = []

    for i in range(1, len(ball_centers)):
        # 计算两个连续帧之间的球心位置差和商标位置差
        delta_ball_center = np.array(ball_centers[i]) - np.array(ball_centers[i-1])
        delta_contour_center = np.array(contour_centers[i]) - np.array(contour_centers[i-1])

        # 计算商标相对球心的位移
        relative_motion = delta_contour_center - delta_ball_center

        # 计算角速度（假设角速度是围绕Z轴旋转）
        angular_velocity = np.linalg.norm(relative_motion) * frame_rate
        angular_velocities.append(angular_velocity)

    # 返回平均角速度
    average_angular_velocity = np.mean(angular_velocities)
    return average_angular_velocity

if __name__ == '__main__':
    opt = Opt()
    opt.weights = "runs/train/exp3/weights/best.pt"
    opt.img_size = 640  # 根据需要设置此值
    opt.conf_thres = 0.25
    opt.iou_thres = 0.3
    opt.device = ""  # 根据需要设置此值
    opt.view_img = False  # 根据需要设置此值
    opt.save_txt = True  # 根据需要设置此值
    opt.nosave = False  # 根据需要设置此值
    opt.classes = None
    opt.agnostic_nms = True
    opt.augment = True
    opt.project = "runs/detect/exp2/"
    opt.name = "exp"
    opt.exist_ok = False  # 根据需要设置此值
    opt.save_conf = True


    # 循环处理文件中的每张图片
    left_image_dir = 'C:/Users/13620/Desktop/l'
    right_image_dir = 'C:/Users/13620/Desktop/r'

    left_images = [os.path.join(left_image_dir, f) for f in os.listdir(left_image_dir)]
    right_images = [os.path.join(right_image_dir, f) for f in os.listdir(right_image_dir)]

    # 确保左右相机图片数量匹配
    num_images = min(len(left_images), len(right_images))
    Number = []  # 计数工具
    real_number = []  # 满足条件的张数差筛选
  # 导出数据进行仿真
with open('coordinates.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['X', 'Y'])  # 写入表头

    for i in range(num_images):
        # 左相机
        opt.source = left_images[i]
        circleL = detect(opt)
        ball_L = circleL[0]  # 球中心点坐标
        print(ball_L)
        writer.writerow([float(ball_L[0]), float(ball_L[1])])
        if len(circleL) > 1:
            contour_L = circleL[1]  # 商标中心点坐标
            Number.append(i)
            if len(Number) == 1:
                relative_coords = [a - b for a, b in zip(contour_L, ball_L)]  # 求出商标与球的相对坐标
            else:  # 还需要设置一个帧率保护，就是在某一范围里出现的数据将其废弃。这个范围根据相机帧率的变化而变化。
                relative_coord_ = [a - b for a, b in zip(contour_L, ball_L)]  # 更新后的相对坐标要与之前的相对坐标做对比
                if relative_coord_[0] - relative_coords[0] < threshold and relative_coord_[1] - relative_coords[
                    1] < threshold:
                    number = Number[-1] - Number[0]  # 第一次得到的“时间”与此时得到的“时间"做差 为了准确的得到数值，可以在多次连续出现时取中间值作为真正的值
                    if number > 6:
                        real_number.append(number)
                        diff = process_real_number(real_number)
                        #print(f"被用于计算的是{diff}")
                        if diff is not None:
                            v = calculate_rotation_rate(diff)
                            print(f"该球的旋转速度为：{v}")
            opt.source = None
            opt.source = right_images[i]
            circleR = detect(opt)
            # 计算三维空间中的坐标
            if len(circleR) > 1: #判断商标是否被识别到
                ball_R = circleR[0]
                contour_R = circleR[1]
                ball_L = cv2.undistortPoints(circleL, K1, D1, P=K1)  # 畸变矫正左相机
                ball_R = cv2.undistortPoints(circleR, K2, D2, P=K2) #畸变矫正右相机
                # 左相机视角下的三维空间坐标 球坐标和商标坐标
                ball_3D= calculate_3D_coordinates(ball_L, ball_R)
                contour_3D = calculate_3D_coordinates(contour_L, contour_R)
                # 存储数据到容器中
                ball_container = ball_container.append(ball_3D)
                contour_container = contour_container.append(contour_3D)
                print(f"左相机坐标系下的3D坐标: {contour_3D}")
                # 三维空间下的旋转速度计算
                final_v = calculate_rotation_speed(contour_3D,ball_3D,500)
                print("final_v：", final_v)
            # 如果没有左相机识别到商标，右相机没有识别到商标，需要通过卡尔曼滤波来得到右相机的商标坐标
            else:
                kf = KalmanFilter()
                # 容器中的二维坐标列表
                # 预测新的坐标
                for z in contour_container:
                    pred = kf.predict()
                    print(f"预测坐标: {pred}")
                    contour_container=contour_container.append(pred)
                    # 更新滤波器
                    # updated = kf.update(np.array(z))
                    # print(f"更新后的坐标: {u#dated}")
                    ##### 这一部分可以封装成函数 ####
                    ball_R = circleR[0]
                    contour_R = pred
                    ball_L = cv2.undistortPoints(circleL, K1, D1, P=K1)  # 畸变矫正左相机
                    ball_R = cv2.undistortPoints(circleR, K2, D2, P=K2)  # 畸变矫正右相机
                    # 左相机视角下的三维空间坐标 球坐标和商标坐标
                    ball_3D = calculate_3D_coordinates(ball_L, ball_R)
                    contour_3D = calculate_3D_coordinates(contour_L, contour_R)
                    # 存储数据到容器中
                    ball_container = ball_container.append(ball_3D)
                    contour_container = contour_container.append(contour_3D)
                    print(f"左相机坐标系下的3D坐标: {contour_3D}")
                    # 三维空间下的旋转速度计算
                    final_v = calculate_rotation_speed(contour_3D, ball_3D, 500)
                    print("final_v：", final_v)


    warnings.filterwarnings("ignore")
    cv2.waitKey(0)
    # 你可以在这里进行进一步的处理，如进行图像标注等
