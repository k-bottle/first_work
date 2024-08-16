import os
from detect import detect, set_logging, select_device, attempt_load, check_img_size, LoadImages, LoadStreams
import torch
from pathlib import Path
import torch.backends.cudnn as cudnn


class Opt:
    pass


# 需要根据需求设置其他 opt 属性。
if __name__ == "__main__":
    opt = Opt()
    opt.weights = "runs/train/exp4/weights/best.pt"
    opt.img_size = 640  # 根据需要设置此值
    opt.conf_thres = 0.7
    opt.iou_thres = 0.4
    opt.device = ""  # 根据需要设置此值
    opt.view_img = True  # 根据需要设置此值
    opt.save_txt = False  # 根据需要设置此值
    opt.nosave = False  # 根据需要设置此值
    opt.classes = None
    opt.agnostic_nms = True
    opt.augment = True
    opt.save_conf = True
    opt.project = "runs/detect/exp14"
    opt.name = "exp"
    opt.exist_ok = False  # 根据需要设置此值
    opt.source = "C:/Users/13620/Desktop/r/"

    dataset = LoadImages(opt.source, img_size=opt.img_size)

    # 循环处理每张图片
    for path, img, im0s, vid_cap in dataset:
        # 在这里调用 detect 函数，确保每次处理不同的图片
        detect(opt)
