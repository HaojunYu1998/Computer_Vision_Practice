import os
import time
import pickle
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score
from typing import Any, Callable, List, Optional, Tuple
from utils import conv3x3, conv1x1, transform, target_transform

class Block(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int=1,
        downsample: Optional[Callable]=None,
        dropout_rate: int=0.5
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv1 = conv3x3(self.inplanes, self.planes, stride)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.conv2 = conv3x3(self.planes, self.planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out) # https://github.com/meliketoy/wide-resnet.pytorch/ puts dropout layer before bn2
        out = self.conv2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return out

lass WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        '''
        Args:
            block (class): A subclass of nn.Module defining a block
            depth (int): The depth of the network, should be 6N+4 where N is group size
            widen_factor (int): The widen factor campared with ResNet
            dropout_rate (float): Dropout rate
            num_classes (int): Number of classes to predict
        '''
        super().__init__()
        assert (depth - 4) % 6 == 0, 'Depth of Wide ResNet should be 6n+4'
        self.depth = depth
        self.widen_factor = widen_factor
        self.widths = [int(width * widen_factor) for width in (16, 32, 64)]
        self.inplanes = 16
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.num_blocks = (self.depth - 4) // 6
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = self._make_layer(self.inplanes, self.widths[0], stride=1, padding=0)
        self.conv3 = self._make_layer(self.widths[0], self.widths[1], stride=2, padding=0)
        self.conv4 = self._make_layer(self.widths[1], self.widths[2], stride=2, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.widths[2], num_classes)

    def _make_layer(self, inplanes, planes, stride=1, padding=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride, padding=padding),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(Block(inplanes, planes, stride, downsample, self.dropout_rate))
        self.inplanes = planes
        for _ in range(1, self.num_blocks):
            layers.append(Block(self.inplanes, planes, dropout_rate=self.dropout_rate))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
