import cv2
import numpy as np
import math


class BallPoint:

    def __init__(self, ROI, _s):
        self.Image = None
        self.background = None
        self.Back = None
        self.M = None
        self.D = None
        self.R = None
        self.P = None
        self.M2 = None
        self.D2 = None
        self.R2 = None
        self.P2 = None
        self.Roi = ROI
        self.roi = self.Roi
        self.es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.ROI_W = 200
        self.ROI_H = 150
        self._str = _s
        self.circleF = None
        self.circleI = None
        self.Binary_thr = 110
        self.setback = 1

    def set_back(Binary_thr, diff_thr):
        BallPoint.setback = 0
        BallPoint.Binary_thr = Binary_thr
        BallPoint.diff_thr = diff_thr

    def set_mats(self, m, d, r, p):
        self.M = m
        self.D = d
        self.R = r
        self.P = p
        # print(self.M)

    def set_mats2(self, m, d, r, p):
        self.M2 = m
        self.D2 = d
        self.R2 = r
        self.P2 = p

    def cnt(self, img, df, show):
        # diff = df
        diff = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        image = img
        area_max = 100  # Replace this value with the appropriate maximum area
        at_time = 0
        xx, yy = 0, 0
        cnts, hierarcchy = cv2.findContours(diff.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if len(cnts) == 0:
            roi = self.Roi
            if show:
                # cv2.imshow(Image, img)
                cv2.waitKey(1)
            outputpoint = [np.array([9999, 0]), np.array([9999, 0])]
            return outputpoint
        else:
            Area = 0
            for i in range(len(cnts)):
                area = cv2.contourArea(cnts[i])
                # print("ccccccccccccc", area)
                if 150 < area < 1700:
                    Area += 1

            if Area == 0:
                roi = self.Roi
                if show:
                    # cv2.imshow(Image, img)
                    cv2.waitKey(1)
                outputpoint = [np.array([9999, 0]), np.array([9999, 0])]
                return outputpoint

        for i in range(len(cnts)):
            if 150 < cv2.contourArea(cnts[i]) < 1700:
                (x, y), radius = cv2.minEnclosingCircle(cnts[i])
                circleF = np.array([x, y])

                if self._str == "L":
                    r_l = radius
                else:
                    r_r = radius

                circleF += self.roi[:2]
                circleI = (int(circleF[0]), int(circleF[1]))
                print("未校正的圆心坐标：", circleI)

                if circleF[0] > 1919 or circleF[1] > 1079:
                    if i == len(cnts) - 1:
                        roi = self.Roi
                        if show:
                            # cv2.imshow(Image, img)
                            cv2.waitKey(1)
                        outputpoint = [np.array([9999, 0]), np.array([9999, 0])]
                        return outputpoint
                    else:
                        if show:
                            cv2.waitKey(1)
                        continue

                if cv2.contourArea(cnts[i]) / (radius * radius * np.pi) < 0.5:
                    if i == len(cnts):
                        if show:
                            # cv2.imshow(Image, img)
                            cv2.waitKey(1)
                        outputpoint = [np.array([9999, 0]), np.array([9999, 0])]
                        return outputpoint
                    else:
                        if show:
                            cv2.waitKey(1)
                        continue

                xx = circleI[0] - (self.ROI_W / 2)
                yy = circleI[1] - (self.ROI_H / 2)

                if (circleI[0] - (self.ROI_W / 2)) < 0:
                    xx = 0
                if (circleI[1] - (self.ROI_H / 2)) < 0:
                    yy = 0
                if (circleI[0] + (self.ROI_W / 2)) > 1919:
                    xx = 1918 - self.ROI_W
                if (circleI[1] + (self.ROI_H / 2)) > 1079:
                    yy = 1078 - self.ROI_H

                temproi = (xx, yy, self.ROI_W, self.ROI_H)
                self.roi = temproi

                inputpoint = [circleF]
                outputpoint_L = cv2.undistortPoints(np.array([inputpoint], dtype=np.float32), self.M, self.D,None,self.R,self.P)
                outputpoint_R = cv2.undistortPoints(np.array([inputpoint], dtype=np.float32), self.M2, self.D2, None,self.R2,
                                                    self.P2)
                outputpoint_L = outputpoint_L.tolist()
                outputpoint_L.append(outputpoint_R.tolist())

                if show:
                    # cv2.putText(img, "(" + str(xx) + "," + str(yy) + ")", (100, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    #             (255, 255, 255), 5, 8, 0)
                    # cv2.imshow(image, img)
                    cv2.waitKey(1)

                return outputpoint_L

        outputpoint = [np.array([9999, 0]), np.array([9999, 0])]
        return outputpoint

    def circl(self, IMG, show):
        image = IMG
        # diff = None
        binary = None

        if self.setback == 1:
            self.Back = IMG.copy()
            _, self.Back = cv2.threshold(self.Back, self.Binary_thr, 255, cv2.THRESH_BINARY)  # 去噪
            # print(type(self.Back))
            self.Back = cv2.morphologyEx(self.Back, cv2.MORPH_OPEN, self.es)
            self.Back = cv2.morphologyEx(self.Back, cv2.MORPH_ERODE, self.es)
            self.setback = 0

            image_dif = image - self.Back
            ROI = image_dif[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]]
            _, binary = cv2.threshold(ROI, self.Binary_thr, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, self.es)  # 形态学变换 膨胀腐蚀去除小黑点
            binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, self.es)
            binary = cv2.dilate(binary, self.es)  # 膨胀
            binary = cv2.dilate(binary, self.es)
            binary = cv2.dilate(binary, self.es)
            cv2.waitKey(1)
        return self.cnt(image, binary, show)
