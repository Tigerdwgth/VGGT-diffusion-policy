import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import torchvision.models as models
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint





class DPEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 resnet_type='resnet18',  # 新增参数，指定 ResNet 类型 包括r3m
                 pretrained=True,        # 是否使用预训练权重
                 ):
        super().__init__()
        self.rgb_image_key = 'image'
        self.state_key = 'agent_pos'
        self.n_output_channels = out_channel

        self.state_shape = observation_space[self.state_key]
        self.image_shape = observation_space[self.rgb_image_key]

        cprint(f"[DPEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DPEncoder] image shape: {self.image_shape}", "yellow")

        # 初始化 ResNet 编码器
        if resnet_type == 'resnet18':
            self.extractor = models.resnet18(pretrained=pretrained)
        elif resnet_type == 'resnet34':
            self.extractor = models.resnet34(pretrained=pretrained)
        elif resnet_type == 'resnet50':
            self.extractor = models.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError(f"ResNet type {resnet_type} is not supported")

        # 修改 ResNet 的最后一层输出通道数
        self.extractor.fc = nn.Linear(self.extractor.fc.in_features, out_channel)

        # 初始化状态编码器
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DPEncoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        # 提取图像特征
        images = observations[self.rgb_image_key]
        assert len(images.shape) == 4, cprint(f"Image shape: {images.shape}, expected 4D tensor (B, C, H, W)", "red")
        img_feat = self.extractor(images)  # B * out_channel

        # 提取状态特征
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * state_mlp_size[-1]

        # 合并特征
        final_feat = torch.cat([img_feat, state_feat], dim=-1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels




