from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction: torch.Tensor,
                      inp_dim: int, anchors: list, num_classes: int):
    CUDA = prediction.is_cuda
    # eg. inp_dim=416, prediction.size(2) = 13, stride=32, grid_size=13
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5+num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size**2)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size**2*num_anchors, bbox_attrs)

    # 按照公式转换特征图的输出
    # 默认聚类算出来的anchor是以416x416的图像为基础的
    # 此时anchors适应feature map的大小
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid the center_x, center_y, object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    # 将网格偏移量添加到中心坐标预测中。
    # #####
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # 将锚应用与边界框的尺寸
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]*anchors)

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5+num_classes]))

    # 最后一件事是将detection map还原为图像大小
    prediction[:, :, :4] *= stride

    return prediction