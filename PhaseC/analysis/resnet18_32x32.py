"""ResNet-18 for 32x32 inputs, as OpenOOD v1.5 defines it (OpenOOD/openood/networks/resnet18_32x32.py).

Why vendored: the CIFAR-10 checkpoint the audit freezes (OpenOOD `cifar10_res18_v1.5`)
is a state dict for this exact architecture, not for torchvision's ImageNet
ResNet-18: a 3x3 stride-1 stem with no max-pool, four stages of two BasicBlocks
(64, 128, 256, 512 channels; strides 1, 2, 2, 2), global average pool, a linear
head. Parameter names (conv1, bn1, layerN.M.{conv1,bn1,conv2,bn2,shortcut.*}, fc)
match the OpenOOD checkpoint so it loads with strict=True. Only the pieces the
embedding extractor needs are kept: `forward(x, return_feature=...)` and
`forward_features(x)`, which returns the 512-d post-avgpool vector, the same
vector OpenOOD feeds to its feature-space detectors (Mahalanobis, kNN, ...).

No command line.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BasicBlock", "ResNet18_32x32"]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18_32x32(nn.Module):
    def __init__(self, num_classes=10, num_blocks=(2, 2, 2, 2)):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.feature_size = 512 * BasicBlock.expansion

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return torch.flatten(self.avgpool(out), 1)

    def forward(self, x, return_feature=False):
        feat = self.forward_features(x)
        logits = self.fc(feat)
        return (logits, feat) if return_feature else logits
