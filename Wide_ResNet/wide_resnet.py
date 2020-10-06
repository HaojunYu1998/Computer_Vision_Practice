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
from utils import conv3x3, conv1x1, conv_init, transform, target_transform

class Block(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int=1,
        downsample: Optional[Callable]=None,
        dropout_rate: int=0.3,
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv1 = conv3x3(self.inplanes, self.planes)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.conv2 = conv3x3(self.planes, self.planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        return out

class WideResNet(nn.Module):
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
        self.stages = [16, 32, 64]
        self.widths = [int(width * widen_factor) for width in self.stages]
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.num_blocks = (self.depth - 4) // 6
        self.conv1 = conv3x3(3, self.stages[0])
        self.conv2 = self._make_layer(self.stages[0], self.widths[0], stride=1, padding=0)
        self.conv3 = self._make_layer(self.widths[0], self.widths[1], stride=2, padding=0)
        self.conv4 = self._make_layer(self.widths[1], self.widths[2], stride=2, padding=0)
        self.bn = nn.BatchNorm2d(self.widths[2], momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.widths[2], num_classes)

    def _make_layer(self, inplanes, planes, stride=1, padding=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride, padding=padding),
            )
        layers = []
        layers.append(Block(inplanes, planes, stride, downsample, self.dropout_rate))
        self.inplanes = planes
        for _ in range(1, self.num_blocks):
            layers.append(Block(self.inplanes, planes, dropout_rate=self.dropout_rate))
        return nn.Sequential(*layers)      

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bn(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
