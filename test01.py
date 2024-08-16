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


def D3_turn_camera_L(x_l, y_l):
    P_c_l = np.array([d3_x, d3_y, d3_z])
    Picture_L = np.array([[0], [0], [0]], dtype=np.float64)
    Picture_L = np.dot(LRinv, (Picture_L - LT))
    Picture_L = tune_x * Picture_L
    Picture_L = tune_zx * Picture_L
    Picture_L = tune_zy * Picture_L - Translation_L
    Picture_L = tune_zy_ * tune_zx_ * Picture_L

    P_g_l = np.array([Picture_L[0, 0], Picture_L[1, 1], Picture_L[2, 2]])

    D_l = np.sqrt((P_c_l[0] - P_g_l[0]) ** 2 + (P_c_l[1] - P_g_l[1]) ** 2 + (P_c_l[2] - P_g_l[2]) ** 2)

    U_b_l, V_b_l = x_l, y_l

    X_bc_l = -D_l * fy_l * np.sqrt(1 / (cx_l ** 2 * fy_l ** 2 - 2 * cx_l * fy_l ** 2 * U_b_l
                                        + cy_l ** 2 * fx_l ** 2 - 2 * cy_l * fx_l ** 2 * V_b_l
                                        + fx_l ** 2 * fy_l ** 2 + fx_l ** 2 * V_b_l ** 2
                                        + fy_l ** 2 * U_b_l ** 2)) * (cx_l - U_b_l)

    Y_bc_l = -D_l * fx_l * np.sqrt(1 / (cx_l ** 2 * fy_l ** 2 - 2 * cx_l * fy_l ** 2 * U_b_l
                                        + cy_l ** 2 * fx_l ** 2 - 2 * cy_l * fx_l ** 2 * V_b_l
                                        + fx_l ** 2 * fy_l ** 2 + fx_l ** 2 * V_b_l ** 2
                                        + fy_l ** 2 * U_b_l ** 2)) * (cy_l - V_b_l)

    Z_bc_l = D_l * fx_l * fy_l * np.sqrt(1 / (cx_l ** 2 * fy_l ** 2 - 2 * cx_l * fy_l ** 2 * U_b_l
                                              + cy_l ** 2 * fx_l ** 2 - 2 * cy_l * fx_l ** 2 * V_b_l
                                              + fx_l ** 2 * fy_l ** 2 + fx_l ** 2 * V_b_l ** 2
                                              + fy_l ** 2 * U_b_l ** 2))

    P_bc_l = np.array([X_bc_l, Y_bc_l, Z_bc_l])
    return P_bc_l


def process_real_number(data):
    # 检查数据列表中是否有连续三个数值的序列
    if len(data) >= 3:
        for i in range(len(data) - 2):
            if data[i] + 1 == data[i + 1] and data[i + 1] + 1 == data[i + 2]:
                middle_number = data[i + 1]  # 提取连续三个数值的中间值
                real_number.append(middle_number)
                print(f"最准确的数值为：{middle_number}")
                return middle_number


# 旋转计算  frame_rate为相机帧率 total_frames为总图片数  rotation_frames为转一圈所花费的图片数
def calculate_rotation_rate(rotation_frames):
    image_time = 1 / frame_rate  # 图片代表的时间，单位为秒
    rotation_time = (rotation_frames / total_frames) * image_time  # 乒乓球旋转一周的时间，单位为秒
    if rotation_time == 0:  # 处理除零情况，避免计算错误
        return 0
    rotation_rate = 1 / rotation_time  # 乒乓球每秒转动的圈数
    return rotation_rate


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
        ball_l = cv2.undistortPoints(np.array([ball_L], dtype=np.float32), M1, D1, None, R1, P1)  # 左相机下的左视角畸变矫正
        ball_l_ = cv2.undistortPoints(np.array([ball_L], dtype=np.float32), M2, D2, None, R2, P2)  # 左相机下的右视角畸变矫正
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
            # 右相机  得到右相机图像的参数来矫正左相机的世界坐标
            opt.source = right_images[i]
            circleR = detect(opt)
            ball_R = circleR[0]
            ball_r = cv2.undistortPoints(np.array([ball_R], dtype=np.float32), M1_, D1_, None, R1_, P1)  # 右相机下的左视角畸变矫正
            ball_r_ = cv2.undistortPoints(np.array([ball_R], dtype=np.float32), M2, D2, None, R2_, P2_)  # 右相机下的右视角畸变矫正
            r_x = ball_r[0, 0, 0]
            r_y = ball_r[0, 0, 1]

            # 矫正后开始转相机坐标
            l_x = ball_l[0, 0, 0]
            l_y = ball_l[0, 0, 1]
            pixel_location = np.zeros((4, 1), dtype=np.float64)
            pixel_location[0, 0] = float(l_x)
            pixel_location[1, 0] = float(l_y)
            pixel_location[2, 0] = float((l_x - r_x))  # 新相机视差为X-X
            pixel_location[3, 0] = 1.0
            Lcam_xyz = Q.dot(pixel_location)
            # 赋值(X,Y,Z)
            table_xyz = np.zeros((3, 1))
            table_xyz[0, 0] = Lcam_xyz[0, 0] / Lcam_xyz[3, 0]
            table_xyz[1, 0] = Lcam_xyz[1, 0] / Lcam_xyz[3, 0]
            table_xyz[2, 0] = Lcam_xyz[2, 0] / Lcam_xyz[3, 0]
            #  print("左相机坐标", table_xyz)
            # # 相机转世界坐标
            xyz = np.dot(LRinv, table_xyz - LT)
            xyz = tune_x * xyz
            xyz = tune_zx * xyz
            xyz = tune_zy * xyz - Translation_L
            xyz = tune_zy_ * tune_zx_ * xyz
            #   print('左相机相对右相机矫正的世界XYZ', xyz)
            # 世界坐标的赋值
            d3_x = xyz[0, 0]
            d3_y = xyz[1, 1]
            d3_z = xyz[2, 2]
            point_l = (d3_x, d3_y, d3_z)
            # l_x 和 l_y是像素坐标
            # 上述的内容是为了验证左右相机的矫正是否成功  是为了验证
            # 返回左相机中心再相机坐标系下的坐标 下面是三维空间坐标的转换
            LL = D3_turn_camera_L(l_x, l_y)
           # print(LL)

            X_bc_L = LL[0]
            Y_bc_L = LL[1]
            Z_bc_L = LL[2]
            # 球标的三维坐标的转换  下面分别是球标的 像素坐标
            if len(circleR) > 1:
                contour_R = circleR[1]
            else:
                contour_R = []
                # print(f'右相机商标的中心坐标：{circleR}')
                U_m_L = circleL[1][0]
                V_m_L = circleL[1][1]
                # 使用球的参数，根据k_l>0来判断商标点是否在球面上
                a_l = ((U_m_L - cx_l) / fx_l) ** 2 + ((V_m_L - cy_l) / fy_l) ** 2 + 1
                b_l = -2 * (((U_m_L - cx_l) / fx_l) * X_bc_L + ((V_m_L - cy_l) / fx_l) * Y_bc_L + Z_bc_L)
                c_l = X_bc_L ** 2 + Y_bc_L ** 2 + Z_bc_L ** 2 - r ** 2
                k_l = b_l ** 2 - 4 * a_l * c_l
                Z_mc_1_l = -(b_l + np.sqrt(b_l ** 2 - 4 * a_l * c_l)) / (2 * a_l)
                Z_mc_2_l = -(b_l - np.sqrt(b_l ** 2 - 4 * a_l * c_l)) / (2 * a_l)
                if Z_mc_1_l > Z_mc_2_l:
                    if k_l > 0:
                        X_mc_L = ((cx_l - U_m_L) * (b_l - np.sqrt(np.square(b_l) - 4 * a_l * c_l))) / (2 * a_l * fx_l)
                        Y_mc_L = ((cy_l - V_m_L) * (b_l - np.sqrt(np.square(b_l) - 4 * a_l * c_l))) / (2 * a_l * fy_l)
                        Z_mc_L = Z_mc_2_l
                        # 相机坐标系下的轮廓点
                        point = [X_mc_L, Y_mc_L, Z_mc_L]
                        point_ = point - LL
                        # 下方代码为相机坐标转换为世界坐标
                        Pl = [[point_[0]], [point_[1]], [point_[2]]]
                        Pll = tune_zy_ * tune_zx_ * tune_zy * tune_zx * tune_x * np.linalg.inv(LR) @ Pl
                        # 世界坐标系下的商标轮廓坐标
                        point_l = (Pll[0], Pll[1], Pll[2])
                        print(f'球标相对于球的世界坐标为{point_l}')
                else:
                    if k_l > 0:
                        X_mc_L = ((cx_l - U_m_L) * (b_l - np.sqrt(np.square(b_l) - 4 * a_l * c_l))) / (2 * a_l * fx_l)
                        Y_mc_L = ((cy_l - V_m_L) * (b_l - np.sqrt(np.square(b_l) - 4 * a_l * c_l))) / (2 * a_l * fy_l)
                        Z_mc_L = Z_mc_1_l
                        # 相机坐标系下的轮廓点
                        point = [X_mc_L, Y_mc_L, Z_mc_L]

                        # 将相对于相机坐标系下的球心坐标的所有三维轮廓点存入Con_3D中
                        point_ = point - LL

                        # 下方代码为相机坐标转换为世界坐标
                        Pl = [[point_[0]], [point_[1]], [point_[2]]]
                        Pll = tune_zy_ * tune_zx_ * tune_zy * tune_zx * tune_x * np.linalg.inv(LR) @ Pl
                        # 世界坐标系下的商标轮廓坐标
                        point_l = (Pll[0], Pll[1], Pll[2])
                       # print(f'球标相对于球的世界坐标为{point_l}')

    warnings.filterwarnings("ignore")
    cv2.waitKey(0)
    # 你可以在这里进行进一步的处理，如进行图像标注等
    cv2.destroyAllWindows()
