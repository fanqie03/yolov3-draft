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

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="dog-cycle-car.png", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", type=int, default=1)
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
images = args.images
batch_size = args.bs
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
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

# 确认输出目录是否存在
if not os.path.exists(args.det):
    os.makedirs(args.det)

# 一次性读取图片
load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

# resize, bgr->rgb, div 255, add dim ...
# PyTorch Variables for images
im_batches = list(map(letterbox_image, loaded_ims, [(inp_dim, inp_dim) for x in range(len(imlist))]))
im_batches = list(map(prep_image, im_batches, [inp_dim for x in range(len(imlist))]))

# List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

if CUDA:
    im_dim_list = im_dim_list.cuda()

#### Create the Batches
leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                            len(im_batches))])) for i in range(num_batches)]
#### The Detection Loop
write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    # load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    # !!! 此时输入模型的图片是经过resize的
    prediction = model(batch)
    # write_results输出的bbox坐标是针对resize过后的图片
    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in imlist

    if not write:  # If we have't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
        im_id = i * batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()

output_recast = time.time()

# 检查output是否有结果，没有直接退出程序
try:
    output
except NameError:
    print("No detections were made")
    exit()

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

draw = time.time()

list(map(lambda x: draw_func(x, loaded_ims, random.choice(colors)), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))
end = time.time()

print(
    f"""SUMMARY
----------------------------------------------------------
{"Task":25s}: {"Time Taken (in seconds)"}
{"Reading addresses":25s}: {load_batch - read_dir:2.3f}
{"Detection (" + str(len(imlist)) + " images)":25s}: {output_recast - start_det_loop:2.3f}
{"Output Processing":25s}: {class_load - output_recast:2.3f}
{"Drawing Boxes":25s}: {end - draw:2.3f}
{"Average time_per_img":25s}: {(end - load_batch) / len(imlist):2.3f}
----------------------------------------------------------
""")

torch.cuda.empty_cache()
