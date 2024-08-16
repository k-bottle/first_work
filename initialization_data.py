import cv2
import numpy
import numpy as np

# 坐标初始化
ball_L_ = [9999, 0]
ball_R_ = [9999, 0]
# 初始化输入左图和右图的大小
image_L = np.zeros((1080, 1920), np.uint8)
image_R = np.zeros((1080, 1920), np.uint8)

threshold = 1  # 二维图片筛选相同图片的阈值
frame_rate = 500  # 相机帧率
total_frames = 70  # 拍摄总图片数

# 导入从matlab得到的数据
cx_r, cy_r, fx_r, fy_r, r = 944.9859916621085, 543.9878913566486, 2491.184439535553, 2493.248053343991, 20
cx_l, cy_l, fx_l, fy_l = 973.3283645499203, 550.5654026556251, 2495.473316478546, 2499.086225433991

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

Translation_L = np.array([[7.72161], [1364.53],
                          [583.002]])

Translation_R = np.array([[1.47643],
                          [-1547.94],
                          [645.78]])
tune_x = np.array(xRotation)
tune_zx = np.array(zxRotation)
tune_zy = np.array(zyRotation)
tune_zx_ = np.array(zxRotation_)
tune_zy_ = np.array(zyRotation_)

LR = np.array(leftRotation)
LRinv = np.linalg.inv(LR)  # Inverse of the leftRotation matrix
LT = np.array(leftTranslation)  # Define a 3x1 matrix

# 超参数的导入 --左相机
intrinsic_filename = "matlabintrinsics.yml"
extrinsic_filename = "matlabextrinsics.yml"
img_size = image_L.shape[1], image_L.shape[0]
# 超参数的导入 --右相机
intrinsic_filename1 = "matlabintrinsics1.yml"
extrinsic_filename1 = "matlabextrinsics1.yml"

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

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    M1, D1, M2, D2, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=img_size
)
