from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pprint import pprint


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.readlines()
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']


    block = {}
    blocks = []

    for line in lines:
        if line[0]=='[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)  # 最后一行会退出循环，最后一层的block没有加入

    return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_fileters = filters = 3  # 前一层的通道数，默认通道数为三层
    output_filters = []  # 每一层的输出通道数目，用于创建module

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if (x['type'] == 'convolutional'):
            activation = x['activation']
            # 有batch norm没有bias，有bias没有batch norm
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_fileters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f'conv_{index}', conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f'batch_norm_{index}', bn)

            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f'leaky_{index}', activn)

        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsampel = nn.Upsample(scale_factor=stride, mode='bilinear')
            module.add_module(f'upsampel_{index}', upsampel)

        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')

            start = int(x['layers'][0])

            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()  # 占位置,对其module_list和blocks[1:]的位置
            module.add_module(f'route_{index}', route)

            # 通道数目合并
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcur_{index}', shortcut)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f"Detection_{index}", detection)
        else:
            raise TypeError(f'unsupport type {x["type"]}')

        module_list.append(module)
        prev_fileters = filters
        output_filters.append(filters)

    return (net_info, module_list)




if __name__ == '__main__':
    blocks = parse_cfg('yolov3.cfg')
    pprint(blocks)
    print(create_modules(blocks))