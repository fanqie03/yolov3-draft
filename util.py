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
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size ** 2)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size ** 2 * num_anchors, bbox_attrs)

    # 按照公式转换特征图的输出
    # 默认聚类算出来的anchor是以416x416的图像为基础的
    # 此时anchors适应feature map的大小
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

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

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # 最后一件事是将detection map还原为图像大小
    prediction[:, :, :4] *= stride

    return prediction


def unique(tensor):
    # tensor_np = tensor.detach().cpu().numpy()
    # unique_np = np.unique(tensor_np)
    # unique_tensor = torch.from_numpy(unique_np)
    #
    # tensor_res = tensor.new(unique_tensor.shape)
    # tensor_res.copy_(unique_tensor)
    # return tensor_res
    return torch.unique(tensor)


def bbox_iou(box1, box2):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """
    函数的结果为dx8的张量，每个检测有8个属性，
    即检测所属批次图像的索引、四个location, object score, max class score, max class score index
    :param prediction:
    :param confidence:
    :param num_classes:
    :param nms_conf:
    :return:
    """
    # 过滤分数低的bbox，并保留他们，方便后续向量化操作（每张图过滤后的数目不同）
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    # nms: 对每个类别相似的边界框做过滤
    # 转成对角线的坐标的形式，使用两个对角线的坐标的形式更好计算IOU
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    # 每张图片经过nms出来的结果数目不一致，
    # 不能通过向量操作
    batch_size = prediction.size(0)
    write = False  # 是否初始化output的标志

    for ind in range(batch_size):
        image_pred = prediction[ind]

        # 每个边界框有85个属性，其中80个是类别score。
        # 只关心最高分的class score，
        # 每行删除80个类别分数，添加具有最大值的class score的索引和class score
        max_conf, max_conf_index = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_index = max_conf_index.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_index)
        image_pred = torch.cat(seq, 1)

        # 过滤分数低的bbox，可能存在没有obj score大于阈值的bbox
        # debug, torch.nonzero出来的是非零元素的索引
        non_zero_ind = torch.nonzero(image_pred[:, 4])
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue

        img_classes = unique(image_pred_[:, -1])

        # 按类别执行NMS
        for cls in img_classes:
            # perform NMS
            # 1. 提取特定类的检测值
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # 将iou > threshold 的bbox置为零， 留下iou < threshold的bbox
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask
                # 消除iou > nms_conf 的bbox， 留下iou < threshold的bbox
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            # 函数的结果为dx8的张量，每个检测有8个属性，
            # 即检测所属批次图像的索引、四个location, object score, max class score, max class score index
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

        try:
            return output
        except:
            return 0