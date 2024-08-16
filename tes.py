import os
import cv2
import numpy as np
from point import BallPoint
from picture import BallPicture
import csv
from detect import detect, set_logging, select_device, attempt_load, check_img_size, LoadImages, LoadStreams
import torch
from pathlib import Path
import torch.backends.cudnn as cudnn

# 初始化
P_mc_L = np.array([0.0, 0.0, 0.0])
P_mc_R = np.array([0.0, 0.0, 0.0])
D3_x, D3_y, D3_z, D2_x_L, D2_y_L, D2_x_R, D2_y_R, D3_x_R, D3_y_R, D3_z_R = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0
# 获取左右轮廓
A_see_L, A_see_R = 0.0, 0.0
# 创建球图片对象
# ball_L_ = (280, 5, 1370, 1060)
# ball_R_ = (320, 5, 1300, 1060)

# 创建图像对象
return_image_L = np.zeros((1080, 1920), dtype=np.uint8)
return_image_R = np.zeros((1080, 1920), dtype=np.uint8)
Picture_L = np.zeros((1080, 1920), dtype=np.uint8)
Picture_R = np.zeros((1080, 1920), dtype=np.uint8)
# b = np.zeros((1080, 1920), dtype=np.uint8)

# 外部变量
T_ball_R = (0, 0, 0)
T_ball_L = (0, 0, 0)
r_l, r_r = 0.0, 0.0
PSphere_center = (0.0, 0.0)
Trademark_center_L = (0.0, 0.0)
Trademark_center_R = (0.0, 0.0)
PS_radius, T_radius_L, T_radius_R = 0.0, 0.0, 0.0

# 导入从matlab得到的数据
cx_r, cy_r, fx_r, fy_r, r = 944.9859916621085, 543.9878913566486, 2491.184439535553, 2493.248053343991, 20
cx_l, cy_l, fx_l, fy_l = 973.3283645499203, 550.5654026556251, 2495.473316478546, 2499.086225433991

#  第二部分

# 初始化输入左图和右图的大小
image_L = np.zeros((1080, 1920), np.uint8)
image_R = np.zeros((1080, 1920), np.uint8)

# 初始化左右相机返回球的大小
# ball_L = (320, 5, 1370, 1060)
# ball_R = (320, 5, 1300, 1060)

# 超参数的导入 --左相机
intrinsic_filename = "matlabintrinsics.yml"
extrinsic_filename = "matlabextrinsics.yml"
img_size = image_L.shape[1], image_L.shape[0]
# img_size = image_L.shape[::-1]
# 超参数的导入 --右相机
intrinsic_filename1 = "matlabintrinsics1.yml"
extrinsic_filename1 = "matlabextrinsics1.yml"

# 初始化 识别到球的坐标
circleL = (9999, 0)
circleR = (9999, 0)
circleL1 = (9999, 0)
circleR1 = (9999, 0)

# 左相机转到世界坐标系的r、t矩阵
leftTranslation = np.array([[118.29247101334],
                            [12.44640292592256],
                            [2346.630892105753]])

leftRotation = np.array([[0.009636122400589198, 0.8019351280937385, 0.5973334039498888],
                         [0.9999523507525808, -0.008661307497514703, -0.004503107462838274],
                         [0.001562488230116232, 0.5973483339574268, -0.8019803779076002]])

xRotation = np.array([[0.9999641823, -0.008463696839, 0],
                      [0.008463696839, 0.9999641823, 0],
                      [0, 0, 1]])

zyRotation = np.array([[1, 0, 0],
                       [0, 0.815267516, 0.5790845166],
                       [0, -0.5790845166, 0.815267516]])

zxRotation = np.array([[0.9999919881, 0, 0.004002967928],
                       [0, 1, 0],
                       [-0.004002967928, 0, 0.9999919881]])

zxRotation_ = np.array([[0.9999411439, 0, -0.01084936141],
                        [0, 1, 0],
                        [0.01084936141, 0, 0.9999411439]])

zyRotation_ = np.array([[1, 0, 0],
                        [0, 0.9998893112, 0.01487835295],
                        [0, -0.01487835295, 0.9998893112]])

Translation_L = np.array([[7.72161],
                          [1364.53],
                          [583.002]])

Translation_R = np.array([[1.47643],
                          [-1547.94],
                          [645.78]])

tune_x = np.array(xRotation)
tune_zx = np.array(zxRotation)
tune_zy = np.array(zyRotation)
tune_zx_ = np.array(zxRotation_)
tune_zy_ = np.array(zyRotation_)

pixel_location = np.zeros((4, 1))  # (x, y, d, 1)
table_xyz = np.zeros((3, 1))  # Transformed table coordinates (Xw/W, Xy/W, Xz/W)
LR = np.array(leftRotation)
LRinv = np.linalg.inv(LR)  # Inverse of the leftRotation matrix
LT = np.array(leftTranslation)  # Define a 3x1 matrix
Lcam_xyz = np.zeros((4, 1))  # (Xw, Yw, Zw, W)
xyz = np.array([0, 0, 0])

# 右相机转到世界坐标系的r、t矩阵
rightTranslation = np.array([[-104.1452490445515], [-21.29896767856934], [2449.087466755583]])
rightRotation = np.array([[-0.01265922632537408, 0.7683937356356526, -0.6398521790419485],
                          [0.9996824705314936, 0.02366885591304047, 0.0086454249006669641],
                          [0.02178765937158239, -0.6395395627290421, -0.7684493773850386]])
xRotation_r = np.array([[0.9999132868, 0.01316885799, 0],
                        [-0.01316885799, 0.9999132868, 0],
                        [0, 0, 1]])
zyRotation_r = np.array([[1, 0, 0],
                         [0, 0.7646493944, -0.644465095],
                         [0, 0.644465095, 0.7646493944]])
zxRotation_r = np.array([[0.9998688058, 0, -0.01619787465],
                         [0, 1, 0],
                         [0.01619787465, 0, 0.9998688058]])
zxRotation_r_ = np.array([[0.9997652882, 0, 0.0216649138],
                          [0, 1, 0],
                          [-0.0216649138, 0, 0.9997652882]])
zyRotation_r_ = np.array([[1, 0, 0],
                          [0, 0.9999984477, -0.001761997264],
                          [0, 0.001761997264, 0.9999984477]])

tune_x_r = np.array(xRotation_r)
tune_zx_r = np.array(zxRotation_r)
tune_zy_r = np.array(zyRotation_r)
tune_zx_r_ = np.array(zxRotation_r_)
tune_zy_r_ = np.array(zyRotation_r_)

pixel_location_R = np.zeros((4, 1))
table_xyz_R = np.array((3, 1))
RR = np.array(rightRotation)
RRinv = np.linalg.inv(RR)
RT = np.array(leftRotation)
Lcam_xyz_R = np.zeros((4, 1))
xyz_R_ = np.array([0, 0, 0])

# 数据导出的代码
# save_path_end = "C:/Users/CoCoo/Desktop/2023-6-30.csv"  # 导出球心的数据
#
# # 打开CSV文件以写入数据
# with open(save_path_end, mode='w', newline='') as f_data_end:
#     csv_writer = csv.writer(f_data_end)


# 从超参数文件中获取坐标矫正数据
# 左相机
fs = cv2.FileStorage(intrinsic_filename, cv2.FILE_STORAGE_READ)
M1 = fs.getNode("M1").mat()
D1 = fs.getNode("D1").mat()
M2 = fs.getNode("M2").mat()
D2 = fs.getNode("D2").mat()
fs.release()

fs.open(extrinsic_filename, cv2.FileStorage_READ)
R = fs.getNode("R").mat()
T = fs.getNode("T").mat()
fs.release()

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    M1, D1, M2, D2, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=img_size
)

ball = BallPoint(roi1, "L")
ball.set_mats(M1, D1, R1, P1)
ball.set_mats2(M2, D2, R2, P2)

# 右相机
fs_ = cv2.FileStorage(intrinsic_filename1, cv2.FileStorage_READ)
M1_ = fs_.getNode("M1").mat()
D1_ = fs_.getNode("D1").mat()
M2_ = fs_.getNode("M2").mat()
D2_ = fs_.getNode("D2").mat()
fs_.release()

fs_.open(extrinsic_filename1, cv2.FileStorage_READ)
R_ = fs_.getNode("R").mat()
T_ = fs_.getNode("T").mat()
fs_.release()

R1_, R2_, P1_, P2_, Q_, roi1_, roi2_ = cv2.stereoRectify(
    M1_, D1_, M2_, D2_, img_size, R_, T_, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=img_size
)

ball = BallPoint(roi1, "R")
ball.set_mats(M1_, D1_, R1_, P1_)
ball.set_mats2(M2_, D2_, R2_, P2_)


# 左相机获取图像
def cenera_l(mat_image_l):
    image_L = mat_image_l
    image_l = image_L.copy()
    # result = BallPoint(roi1, "l")
    result = ball
    result = result.circl(image_l, True)  # circl 是对的球预处理 在其他文件中 等下需要对ball进行初始化  返回的是坐标
    circleL = result[0]  # 返回的是左相机视角下球的二维坐标
    circleL1 = result[1]  # 返回的是右相机视角下球的二维坐标
    con_2d_l = []  # 矫正后的二维坐标存储在这里面
    con_3d_l = []  # 先给个空值，后续二维转三维
    inputpoint_l = []
    # 调用YOLO，输出商标的二维中心坐标
    Trademark_l = detect(opt)
    #  # YOLO 切入点 数据给到Trademark_l   这边只有数据，具体的图像展示在YOLO代码中
    # 矫正坐标
    inputpoint_l = Trademark_l
    outputpoint_l = cv2.undistortPoints(np.array([inputpoint_l]), M1, D1, R=R1, P=P1)
    # outputpoint_l = cv2.undistortPoints(np.append([inputpoint_l], dtype=np.float32), M1, D1, R=R1, P=P1)
    con_2d_l = outputpoint_l
    #  print("con_2d_l:", con_2d_l)
    # print("circleL[0]:", circleL[0])
    # 赋值（x,y,d,1）,t  左相机对右相机对世界坐标进行矫正  球心
    D2_X = circleL[0]
    D2_x_L = D2_X[0]
    D2_y_L = D2_X[1]
    print('左相机球的二维坐标：', D2_x_L)
    pixel_location = np.zeros((4, 1), dtype=np.float64)
    pixel_location[0, 0] = float(D2_X[0])
    pixel_location[1, 0] = float(D2_X[1])
    pixel_location[2, 0] = float(D2_X[0] - circleR[0])  # 新相机视差为X-X
    pixel_location[3, 0] = 1.0
    Lcam_xyz = Q.dot(pixel_location)
    # 赋值(X,Y,Z)
    table_xyz = np.zeros((3, 1))
    table_xyz[0, 0] = Lcam_xyz[0, 0] / Lcam_xyz[3, 0]
    table_xyz[1, 0] = Lcam_xyz[1, 0] / Lcam_xyz[3, 0]
    table_xyz[2, 0] = Lcam_xyz[2, 0] / Lcam_xyz[3, 0]
    print("L相机坐标", table_xyz[0, 0])

    xyz = np.dot(LRinv, table_xyz - LT)
    xyz = tune_x * xyz
    xyz = tune_zx * xyz
    xyz = tune_zy * xyz - Translation_L
    xyz = tune_zy_ * tune_zx_ * xyz
    print('左相机矫正的世界XYZ', xyz[0, 0])

    d3_x = xyz[0, 0]
    d3_y = xyz[1, 1]
    d3_z = xyz[2, 2]
    point_l = (d3_x, d3_y, d3_z)
    # 上述的内容是为了验证左右相机的矫正是否成功
    # 返回左相机中心再相机坐标系下的坐标 下面是三维空间坐标的转换
    LL = D3_turn_camera_L(D2_x_L, D2_y_L)
    X_bc_L = LL[0]
    Y_bc_L = LL[1]
    Z_bc_L = LL[2]
    # 商标中心二维转三维坐标的过程
    U_m_L, V_m_L = _m_l = con_2d_l[0, 0]

    a_l = ((U_m_L - cx_l) / fx_l) ** 2 + ((V_m_L - cy_l) / fy_l) ** 2 + 1
    b_l = -2 * (((U_m_L - cx_l) / fx_l) * X_bc_L + ((V_m_L - cy_l) / fx_l) * Y_bc_L + Z_bc_L)
    c_l = X_bc_L ** 2 + Y_bc_L ** 2 + Z_bc_L ** 2 - r ** 2
    k_l = b_l ** 2 - 4 * a_l * c_l
    Z_mc_1_l = -(b_l + np.sqrt(b_l ** 2 - 4 * a_l * c_l)) / (2 * a_l)
    Z_mc_2_l = -(b_l - np.sqrt(b_l ** 2 - 4 * a_l * c_l)) / (2 * a_l)

    if Z_mc_1_l > Z_mc_2_l:
        if k_l > 0:
            X_mc_L = ((cx_l - U_m_L) * (b_l - np.sqrt(b_l ** 2 - 4 * a_l * c_l))) / (2 * a_l * fx_l)
            Y_mc_L = ((cy_l - V_m_L) * (b_l - np.sqrt(b_l ** 2 - 4 * a_l * c_l))) / (2 * a_l * fy_l)
            Z_mc_L = Z_mc_2_l

            # Camera coordinates of the contour point
            point = np.array([X_mc_L, Y_mc_L, Z_mc_L])
            # Transform to world coordinates
            point_ = np.dot(np.dot(np.dot(np.dot(np.dot(tune_zy_, tune_zx_), tune_zy), tune_zx), tune_x),
                            np.dot(LRinv, point))

            # 世界坐标系下的轮廓坐标
            point_l = (point_[0], point_[1], point_[2])

        else:
            if k_l > 0:
                X_mc_L = b_l + (b_l ** 2 - 4 * a_l * c_l) ** 0.5 * (cx_l - U_m_L) / 2 * a_l * fx_l
                Y_mc_L = b_l + (b_l ** 2 - 4 * a_l * c_l) ** 0.5 * (cy_l - V_m_L) / 2 * a_l * fy_l
                Z_mc_L = Z_mc_1_l

                point = (X_mc_L, Y_mc_L, Z_mc_L)
                point = point - LL
                Pl = np.array([point[0], point[1], point[2]]).reshape(3, 1)
                Pll = tune_zy_ * tune_zx_ * tune_zy * tune_zx * tune_x * RRinv * Pl
                # 世界坐标系下的商标轮廓坐标
                point_l = (Pll[0], Pll[1], Pll[2])

                con_3d_l = point_l
    return image_l


def cenera_r(mat_image_r):
    image_R = mat_image_r
    image_r = image_R.copy()
    result = ball
    result = result.circl(image_r, True)  # circl 是对的球预处理 在其他文件中 等下需要对ball进行初始化  返回的是坐标
    circleR = result[0]  # 返回的是左相机视角下球的二维坐标
    circleR1 = result[1]  # 返回的是右相机视角下球的二维坐标
    con_2d_r = []  # 矫正后的二维坐标存储在这里面
    con_3d_r = []  # 先给个空值，后续二维转三维
    inputpoint_r = []
    # 调用YOLO，输出商标的二维中心坐标
    Trademark_r = detect(opt)
    # 矫正坐标
    inputpoint_r = Trademark_r
    outputpoint_r = cv2.undistortPoints(np.array([inputpoint_r]), M1, D1, R=R1, P=P1)
    con_2d_r = outputpoint_r

    # *****赋值(x,y,d,1).t
    D2_X = circleR[0]
    D2_x_R = D2_X[0]
    D2_y_R = D2_X[1]
    print('右相机的球二维坐标：', D2_x_R)
    pixel_location = np.zeros((4, 1), dtype=np.float64)
    pixel_location[0, 0] = D2_x_R
    pixel_location[1, 0] = D2_y_R
    pixel_location[2, 0] = D2_x_R - circleL[0]
    pixel_location[3, 0] = 1
    Lcam_xyz = np.dot(Q, pixel_location)
    # 赋值(X,Y,Z)
    table_xyz = np.zeros((3, 1), dtype=np.float64)
    table_xyz[0, 0] = Lcam_xyz[0, 0] / Lcam_xyz[3, 0]
    table_xyz[1, 0] = Lcam_xyz[1, 0] / Lcam_xyz[3, 0]
    table_xyz[2, 0] = Lcam_xyz[2, 0] / Lcam_xyz[3, 0]
    # print("L相机坐标",table_xyz[0,0])
    xyz = np.dot(RRinv, table_xyz - RT)
    xyz = tune_x * xyz
    xyz = tune_zx * xyz
    xyz = tune_zy * xyz - Translation_R
    xyz = tune_zy_ * tune_zx_ * xyz
    print('右相机矫正的世界XYZ', xyz[0, 0])

    d3_x = xyz[0, 0]
    d3_y = xyz[1, 0]
    d3_z = xyz[2, 0]
    point_l = (d3_x, d3_y, d3_z)

    # 返回左相机中心再相机坐标系下的坐标 下面是三维空间坐标的转换
    D2_x_R = circleR1[0]
    D2_y_R = circleR1[1]
    RR = D3_turn_cenera_R(D2_x_R, D2_y_R)  # 二维球心转三维
    x_bc_R = RR[0]
    y_bc_R = RR[1]
    z_bc_R = RR[2]
    # 商标轮廓的转换  2D 转 3D
    for i in range(len(con_2d_r)):
        u_m_r = con_2d_r[i].x
        v_m_r = con_2d_r[i].y
    a_r = ((u_m_r - cx_r) / fx_r) ** 2 + ((v_m_r - cy_r) / fy_r) ** 2 + 1
    b_r = -2 * (((u_m_r - cx_r) / fx_r) * x_bc_R + ((v_m_r - cy_r) / fx_r) * y_bc_R + z_bc_R)
    c_r = x_bc_R ** 2 + y_bc_R ** 2 + z_bc_R ** 2 - r ** 2
    k_r = b_r ** 2 - 4 * a_r * c_r

    Z_mc_1_r = (-b_r + np.sqrt(b_r ** 2 - 4 * a_r * c_r)) / (2 * a_r)
    Z_mc_2_r = (-b_r - np.sqrt(b_r ** 2 - 4 * a_r * c_r)) / (2 * a_r)

    if Z_mc_1_r > Z_mc_2_r:
        if k_r > 0:
            X_mc_R = ((cx_r - u_m_r) * (np.sqrt(b_r ** 2 - 4 * a_r * c_r) - b_r)) / (2 * a_r * fx_r)
            Y_mc_R = ((cy_r - v_m_r) * (np.sqrt(b_r ** 2 - 4 * a_r * c_r) - b_r)) / (2 * a_r * fy_r)
            Z_mc_R = Z_mc_2_r

            # 相机坐标系下的轮廓点
            point = np.array([X_mc_R, Y_mc_R, Z_mc_R])
            # 将所有相对于相机坐标系下的球心坐标的所有三维轮廓点存入到Con_3D中
            point_ = point - RR  # 球标中心相对于球心的位置，得到结果后在转为世界坐标

            # 下方代码为相机坐标转换为世界坐标
            PR = np.array([point_.x, point_.y, point_.z]).reshape(3, 1)
            PRR = tune_zy_ * tune_zx_ * tune_zy * tune_zx * tune_x * RRinv * P1

            # 世界坐标系下的商标轮廓坐标   相机坐标转为世界坐标
            point_R = (PRR[0, 0], PRR[1, 0], PRR[2, 0])

            con_3d_r.append(point_R)
    else:
        if k_r > 0:
            X_mc_R = b_r + (b_r ** 2 - 4 * a_r * c_r) ** 0.5 * (cx_r - u_m_r) / 2 * a_r * fx_r
            Y_mc_R = b_r + (b_r ** 2 - 4 * a_r * c_r) ** 0.5 * (cy_r - v_m_r) / 2 * a_r * fy_r
            Z_mc_R = Z_mc_1_r
            point = (X_mc_R, Y_mc_R, Z_mc_R)
            point = point - RR
            Pr = np.array([point.x, point.y, point.z]).reshape(3, 1)
            Prr = tune_zy_ * tune_zx_ * tune_zy * tune_zx * tune_x * RRinv * P1
            # 世界坐标系下的商标轮廓坐标
            point_R = (Prr[0, 0], Prr[1, 0], Prr[2, 0])

            con_3d_r.append(point_R)

    return image_r


# 下面的两个函数的作用是  将得到的二维圆的坐标转换为三维世界坐标然后返回出去
#  不明白为什么要写分开写两边，我等它跑通了再做优化，写出一个函数两次调用就行
def D3_turn_camera_L(x_l, y_l):
    P_c_l = np.array([D3_x, D3_y, D3_z])

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


def D3_turn_cenera_R(x_r, y_r):  # 右相机
    P_c_r = []
    P_c_r[0] = D3_x_R
    P_c_r[1] = D3_y_R
    P_c_r[2] = D3_z_R

    # Perform the calculations
    Picture_R = np.array([0, 0, 0])
    Picture_R = np.dot(RRinv, Picture_R - RT)
    Picture_R = tune_x_r * Picture_R
    Picture_R = tune_zx_r * Picture_R
    Picture_R = tune_zy_r * Picture_R - Translation_R
    Picture_R = tune_zy_r_ * tune_zx_r_ * Picture_R

    P_g_r = np.array([Picture_R[0, 0], Picture_R[1, 1], Picture_R[2, 2]])

    D_r = np.sqrt((np.power(P_c_r[0] - P_g_r[0], 2) +
                   np.power(P_c_r[1] - P_g_r[1], 2) +
                   np.power(P_c_r[2] - P_g_r[2], 2)))
    u_b_r = x_r
    v_b_r = y_r

    X_bc_r = -D_r * fy_r * np.sqrt(1 / (np.power(cx_r, 2) * np.power(fy_r, 2) - 2 * cx_r * np.power(fy_r, 2) * u_b_r +
                                        np.power(cy_r, 2) * np.power(fx_r, 2) - 2 * cy_r * np.power(fx_r, 2) * v_b_r +
                                        np.power(fx_r, 2) * np.power(fy_r, 2) + np.power(fx_r, 2) * np.power(v_b_r, 2) +
                                        np.power(fy_r, 2) * np.power(u_b_r, 2))) * (cx_r - u_b_r)

    Y_bc_r = -D_r * fx_r * np.sqrt(1 / (np.power(cx_r, 2) * np.power(fy_r, 2) - 2 * cx_r * np.power(fy_r, 2) * u_b_r +
                                        np.power(cy_r, 2) * np.power(fx_r, 2) - 2 * cy_r * np.power(fx_r, 2) * v_b_r +
                                        np.power(fx_r, 2) * np.power(fy_r, 2) + np.power(fx_r, 2) * np.power(v_b_r, 2) +
                                        np.power(fy_r, 2) * np.power(u_b_r, 2))) * (cy_r - v_b_r)

    Z_bc_r = D_r * fx_r * fy_r * np.sqrt(
        1 / (np.power(cx_r, 2) * np.power(fy_r, 2) - 2 * cx_r * np.power(fy_r, 2) * u_b_r +
             np.power(cy_r, 2) * np.power(fx_r, 2) - 2 * cy_r * np.power(fx_r, 2) * v_b_r +
             np.power(fx_r, 2) * np.power(fy_r, 2) + np.power(fx_r, 2) * np.power(v_b_r,
                                                                                  2) +
             np.power(fy_r, 2) * np.power(u_b_r, 2)))

    P_bc_r = np.array([X_bc_r, Y_bc_r, Z_bc_r])
    return P_bc_r


class Opt:
    pass


if __name__ == '__main__':
    opt = Opt()
    opt.weights = "runs/train/exp4/weights/best.pt"
    opt.img_size = 640  # 根据需要设置此值
    opt.conf_thres = 0.25
    opt.iou_thres = 0.45
    opt.device = ""  # 根据需要设置此值
    opt.view_img = True  # 根据需要设置此值
    opt.save_txt = True  # 根据需要设置此值
    opt.nosave = True  # 根据需要设置此值
    opt.classes = None
    opt.agnostic_nms = True
    opt.augment = True
    opt.project = "runs/detect"
    opt.name = "exp"
    opt.exist_ok = False  # 根据需要设置此值
    opt.save_conf = True
    # 获取图片，代码的入口 - ------------------------------------
    opt.source = "C:/Users/13620/Desktop/The final the code/l68.jpg"

    image = cv2.imread("C:/Users/13620/Desktop/The final the code/l68.jpg")  # 调用图片处理函数
    cenera_l(image)
    # 显示图片
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    # 你可以在这里进行进一步的处理，如进行图像标注等
    cv2.destroyAllWindows()
