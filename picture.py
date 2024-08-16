import cv2
import numpy as np


class BallPicture:
    def __init__(self, ROI, s):
        self.Back_ = None
        self.Roi_ = ROI
        self.roi_ = self.Roi_
        self._str = s
        self.es_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.ROI_W_ = 100
        self.ROI_H_ = 150
        self.binary_ = None
        self.setback_ = 1
        self.Binary_thr_ = 98

    def process_image(self, img_, dif_, show_):
        diff_ = dif_
        image_ = img_
        xx_, yy_ = 0, 0
        cnts_, hierarchy_ = cv2.findContours(diff_.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(cnts_)):
            if 100 < cv2.contourArea(cnts_[i]) < 1800:
                circleF_, r_ = cv2.minEnclosingCircle(cnts_[i])
                circleF_ = (circleF_[0] + self.roi_.x, circleF_[1] + self.roi_.y)
                circleI_ = (int(circleF_[0]), int(circleF_[1]))

                if circleF_[0] > 1919 or circleF_[1] > 1079:
                    if i == len(cnts_) - 1:
                        self.roi_ = self.Roi_
                        return np.zeros((272, 512), np.uint8)
                    else:
                        continue

                if cv2.contourArea(cnts_[i]) / (r_ * r_ * np.pi) < 0.5:
                    if i == len(cnts_) - 1:
                        self.roi_ = self.Roi_
                        return np.zeros((272, 512), np.uint8)
                    else:
                        continue
                xx_ = circleI_[0] - (self.ROI_W_ / 2)
                yy_ = circleI_[1] - (self.ROI_H_ / 2)

                if xx_ < 0:
                    xx_ = 0
                if yy_ < 0:
                    yy_ = 0
                if circleI_[0] + (self.ROI_W_) / 2 > 1919:
                    xx_ = 1918 - self.ROI_W_
                if circleI_[1] + (self.ROI_H_) / 2 > 1079:
                    yy_ = 1078 - self.ROI_H_
                temproi_ = (xx_, yy_, self.ROI_W_, self.ROI_H_)
                self.roi_ = temproi_

                if show_:
                    if self._str == "L":
                        T_ball_L = (
                            int(circleI_[0] - 1.09 * r_), int(circleI_[1] - 0.98 * r_), int(2.12 * r_), int(2.13 * r_))
                        Part_image_ = image_[T_ball_L[1]:T_ball_L[1] + T_ball_L[3],
                                      T_ball_L[0]:T_ball_L[0] + T_ball_L[2]]
                    else:
                        T_ball_R = (
                            int(circleI_[0] - 0.9 * r_), int(circleI_[1] - 1.0 * r_), int(2.2 * r_), int(2.15 * r_))
                        Part_image_ = image_[T_ball_R[1]:T_ball_R[1] + T_ball_R[3],
                                      T_ball_R[0]:T_ball_R[0] + T_ball_R[2]]
                return Part_image_

        return np.zeros((272, 512), np.uint8)

    def circl_(self, IMG_, show_):
        image_ = IMG_
        diff_ = None

        if self.setback_ == 1:
            self.Back_ = IMG_.copy()
            cv2.threshold(self.Back_, self.Back_, self.Binary_thr_, 255, cv2.THRESH_BINARY)
            cv2.morphologyEx(self.Back_, self.Back_, cv2.MORPH_ERODE, self.es_)
            cv2.morphologyEx(self.Back_, self.Back_, cv2.MORPH_ERODE, self.es_)
            self.setback_ = 0

        image_dif_ = image_ - self.Back_
        self.Roi_ = image_dif_(self.roi_)
        cv2.threshold(self.Roi_, self.binary_, self.Binary_thr_, 255, cv2.THRESH_BINARY)
        cv2.morphologyEx(self.binary_, self.binary_, cv2.MORPH_ERODE, self.es_)
        cv2.morphologyEx(self.binary_, self.binary_, cv2.MORPH_ERODE, self.es_)
        cv2.dilate(self.binary_, self.binary_, self.es_)
        cv2.dilate(self.binary_, self.binary_, self.es_)
        cv2.dilate(self.binary_, self.binary_, self.es_)

        kal_ = self.process_image(image_, self.binary_, show_)

        return kal_
