from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--image", dest='image', help="Image",
                        default="dog-cycle-car.png", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--confidence", dest="confidence", type=float,
                        help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", type=float,
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. "
                                                    "Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--device", default="cpu")

    return parser.parse_args()


def draw_func(x, results, color):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


args = arg_parse()
image = args.image
confidence = args.confidence
nms_thesh = args.nms_thresh
start = 0
CUDA = torch.cuda.is_available() and args.device != "cpu"

num_classes = 80  # For COCO
classes = load_classes("coco.names")

# 初始化网络和加载权重
# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

# 读取图片，尝试读取文件夹下的图片，不行直接单个图片，保存在imlist数组里
read_dir = time.time()
# Detection phase

# 确认输出目录是否存在
if not os.path.exists(args.det):
    os.makedirs(args.det)

# 一次性读取图片
load_batch = time.time()
loaded_ims = cv2.imread(image)
# resize, bgr->rgb, div 255, add dim ...
# PyTorch Variables for images
im_batches = letterbox_image(loaded_ims, (inp_dim, inp_dim))
im_batches = prep_image(im_batches, inp_dim)


# List containing dimensions of original images
im_dim_list = [(loaded_ims.shape[1], loaded_ims.shape[0])]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

if CUDA:
    im_dim_list = im_dim_list.cuda()

if CUDA:
    im_batches = im_batches.cuda()
# !!! 此时输入模型的图片是经过resize的
prediction = model(im_batches)
# write_results输出的bbox坐标是针对resize过后的图片
prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)
output = prediction
# 在绘制边界框之前，输出张量中包含的预测与网络的输入大小一致，而不与图像的原始大小一致。
# 因此，在绘制边界框之前，让我们将每个边界框的角属性转换为图像的原始尺寸。

# 在绘制边界框之前，输出张量中包含的预测是对填充图像的预测，而不是原始图像的预测。
# 仅仅将它们重新缩放到输入图像的尺寸在这里是行不通的。
# 我们首先需要相对于包含原始图像的填充图像上区域的边界，转换要测量的框的坐标。
im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

# 现在，我们的坐标符合填充区域上图像的尺寸。
# 但是，在letterbox_image函数中，我们已经通过缩放因子调整了图像的两个尺寸（请记住，两个尺寸都用一个公共因子进行划分以保持纵横比）。
# 现在，我们撤消此重新缩放，以获取原始图像上边界框的坐标。
output[:, 1:5] /= scaling_factor

class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

list(map(lambda x: draw_func(x, [loaded_ims], random.choice(colors)), output))
# draw_func(output, loaded_ims, random.choice(colors))
det_name = "{}/det_{}".format(args.det, image)
cv2.imwrite(det_name, loaded_ims)
end = time.time()
