# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@notice  : Model for "Global Heterogeneous Graph Convolutional Network: From Coarse to Refined Land Cover and Land Use Segmentation"
"""

import copy
import torch
import torch.nn as nn
from torch.nn import init, Conv2d
from torch.nn.functional import interpolate, softmax
from collections import OrderedDict
from .BasicModule import BasicModule
from .utils import onehot, expandAndRepeat, getbackbone

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


def createEdgeCalPara(lcClassNum=5):
    """
    get the meta-path parameters a_P, b_P
    """
    # a_ij = a_ji in undirected graph
    paraNum = int(lcClassNum * (lcClassNum + 1) / 2)
    linearPara = torch.zeros((2, paraNum, 1))
    # y = ax + b
    linearPara[0] = init.xavier_uniform_(torch.randn((paraNum, 1)))  # a_P
    linearPara[1] = init.constant_(torch.randn((paraNum, 1)), 0)  # b_P
    return nn.Parameter(linearPara)


class InterGCN(nn.Module):
    def __init__(self, num_state, num_node, lcClassNum, bias=False):
        super(InterGCN, self).__init__()
        self.num_state, self.num_node, self.lcClassNum = num_state, num_node, lcClassNum
        self.edgeCalPara = createEdgeCalPara(lcClassNum)
        self.paraNum = int(lcClassNum * (lcClassNum + 1) / 2)
        for lcIdx in range(self.paraNum):
            setattr(self, f"lcconv{lcIdx+1}",
                    nn.Conv1d(num_node, num_node, kernel_size=1, padding=0, stride=1, groups=num_node, bias=True))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)
        self.bn1 = nn.BatchNorm1d(num_node)
        self.bn2 = nn.BatchNorm1d(num_state)

    def forward(self, x):
        """
        x : (n, num_state, num_node)
        lc: (n, num_node)
        """
        x, lc = x
        b = x.size(0)
        # 1. node-wise
        # (n, num_state, num_node) -> (n, num_node, num_state)
        x = x.permute(0, 2, 1).contiguous()

        x_list = []
        for lcIdx in range(self.paraNum):
            tempt = getattr(self, f"lcconv{lcIdx+1}")(x)
            x_list.append(tempt)
        # paraNum * [(n, num_node, num_state)] -> (n, num_node, paraNum, num_state)
        x_list = torch.stack(x_list, dim=2)

        h = torch.zeros_like(x)
        # (n, num_node, paraNum, num_state) -> (n, num_node * paraNum, num_state)
        x_list = x_list.view(b, -1, self.num_state)
        for batchIdx in range(b):
            tempt1 = expandAndRepeat(lc[batchIdx], 0, self.num_node)
            tempt2 = expandAndRepeat(lc[batchIdx], 1, self.num_node)
            idMat1 = torch.maximum(tempt1, tempt2)
            idMat2 = torch.minimum(tempt1, tempt2)
            # (num_node, num_node)
            kjList = (idMat1 * (idMat1 + 1) / 2 + idMat2).to(torch.int64)

            # (num_node, num_node)
            # ax + b
            a = self.edgeCalPara[0, kjList].repeat((1, 1, self.num_state))
            b = self.edgeCalPara[1, kjList].repeat((1, 1, self.num_state))
            kjList = kjList + expandAndRepeat(
                torch.arange(0, self.num_node * self.paraNum, self.paraNum, dtype=torch.int64).to(x.device), 1, self.num_node)
            h[batchIdx] = torch.sum(a * x_list[batchIdx, kjList] + b, dim=0)  # ax + b

        h = self.bn1(h)
        h = h + x
        h = self.relu(h.permute(0, 2, 1))

        # 2. channel-wise
        h = self.conv2(h)
        h = self.bn2(h)
        return (h, lc)


class GHGCN(BasicModule):
    """
    In order to utilize more spectral info(4 or more bands)
    we can't just use the vgg as encoder part for it's input bands are fixed as 3
    """

    def __init__(self, opt):
        self.n_channels = opt.indim
        self.lcClassNum = opt.lcClassNum
        self.isMultiClassifier = opt.isMultiClassifier
        self.isLCEncoder = opt.isLCEncoder
        self.n_classes = opt.outdim if self.isMultiClassifier is False else opt.outdim * self.lcClassNum
        self.backbone = opt.backbone
        self.GCNLayerNum = opt.GCNLayerNum
        self.nodeNumRate = opt.nodeNumRate
        self.isBackBoneFrozen = opt.isBackBoneFrozen
        pretrained = True if opt.train is True else False
        inChannels = {"c1": 256, "c2": 512, "c3": 1024, "c4": 2048}
        self.backboneFeatureLayer = opt.backboneFeatureLayer
        super(GHGCN, self).__init__()
        self.pretrained = getbackbone(opt.backbone, pretrained)

        # 1. CNN-baseforward
        # adjust the inputChannel from 3 to 4
        if self.n_channels != 3:
            # self.pretrained.conv1 = eval(str(self.pretrained.conv1).replace("(3", f"({n_channels}"))
            self.pretrained.conv1 = eval(str(self.pretrained.conv1).replace("Conv2d(3", f"Conv2d({self.n_channels}"))

        if self.isBackBoneFrozen:
            print("frozen!")
            for name, param in self.pretrained.named_parameters():
                if name == "conv1.weight" and self.n_channels != 3:
                    continue
                else:
                    param.requires_grad = False

        # Remove abundant module
        self.pretrained.avgpool, self.pretrained.fc = None, None
        for layerIdx in range(int(self.backboneFeatureLayer[1]), 4):
            setattr(self.pretrained, f"layer{layerIdx + 1}", None)

        # 2. Graph
        # 2.1 First time to reduce feature dim
        in_channels = inChannels[self.backboneFeatureLayer]
        in_channels = in_channels * 2 if self.isLCEncoder is True else in_channels
        inter_channels = in_channels // 4
        self.conv51 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels), nn.ReLU())

        # 2.2 Prepare the heterogeneous graph
        num_in, num_mid, kernel = inter_channels, 64, 1
        self.num_s = int(2 * num_mid)  # node feature dim
        self.num_n = int(self.nodeNumRate * num_mid)  # node num
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)

        # 2.2.1 Second time to reduce feature dim
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)

        # 2.2.2 Graph projection matrix-B
        self.conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)

        # 3. Heterogeneous GCN
        self.inter_gcn = nn.Sequential(
            OrderedDict([(f"InterGCN_{idx+1:02d}", InterGCN(self.num_s, self.num_n, self.lcClassNum))
                         for idx in range(self.GCNLayerNum)]))

        # 4. Feature fusion
        # 4.1 Increase the featrue dim
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1), groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in)

        # 4.2 Feature fusion and conv
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels), nn.ReLU())

        # 5. Classify
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, self.n_classes, 1))

        # LC-Encoder
        if self.isLCEncoder is True:
            self.lcPretrained = copy.deepcopy(self.pretrained)
            self.lcPretrained.conv1 = eval(str(self.lcPretrained.conv1).replace("Conv2d(3", f"Conv2d({self.lcClassNum}"))
            # Unfreeze
            for name, param in self.lcPretrained.conv1.named_parameters():
                param.requires_grad = True

    def pretrained_forward(self, model, x):
        """
        自定义baseforward
        """
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        for layerIdx in range(int(self.backboneFeatureLayer[1])):
            x = getattr(model, f"layer{layerIdx + 1}")(x)
        return x

    def forward(self, x, lc, mask):
        batch_size = x.size(0)
        imsize = x.size()[2:]
        rate = 256 // 32  # size of c3 is 32

        # 1.1 (n, h, w) -> (n, h, w, lcNum) -> (n, lcNum, h, w)
        lc_onehot_init = onehot(lc, self.lcClassNum).permute(0, 3, 1, 2).contiguous()

        # 0. 数据准备
        # 0.1 降采样LC
        lc = lc[:, ::rate, ::rate].contiguous()
        # (n, h, w) -> (n, h*w)
        lc_reshape = lc.view(batch_size, -1)
        # Transfer lc to one-hot
        # (n, h*w) -> (n, h*w, 5) -> (n, 5, h*w)
        lc_onehot = onehot(lc_reshape, self.lcClassNum).permute(0, 2, 1)

        # Prepare mask
        mask_reshape = mask[:, ::rate, ::rate].contiguous().view(batch_size, -1)
        mask_reshape = expandAndRepeat(mask_reshape, 1, self.num_n)

        # # 1. CNN
        feat = self.pretrained_forward(self.pretrained, x)
        featsize = feat.size()[2:]
        if self.isLCEncoder is True:
            lc_feat = self.pretrained_forward(self.lcPretrained, lc_onehot_init)
            feat = torch.cat([feat, lc_feat], dim=1)

        # # 2. Graph
        # 2.1 First time to reduce featrue dim
        # X
        # (n, 2048, h, w) --> (n, num_in, h, w)
        feat = self.conv51(feat)

        # 2.2 Prepare the heterogeneous graph
        # 2.2.1 Second time to reduce feature dim
        # $ Φ(x) $
        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(feat).view(batch_size, self.num_s, -1)

        # 2.2.2 Graph Projection matrix-B
        # $ B = θ(x) $
        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(feat).view(batch_size, self.num_n, -1)
        x_proj_reshaped[~mask_reshape] = 0

        # Get the lc type of each node
        # (n, 5, h*w) x (n, num_node, h*w)T --> (n, 5, num_node)
        x_proj_reshaped_softmax = softmax(x_proj_reshaped, dim=2)
        lc_n = torch.matmul(lc_onehot, x_proj_reshaped_softmax.permute(0, 2, 1))
        # (n, 5, num_node) -> (n, num_node)
        lc_n = torch.argmax(lc_n, dim=1)

        # Add the same-class constraint, only pixels of the same lc type is projected
        # (n, h*w) -> (n, num_node, h*w)
        lc_reshape_repeat = expandAndRepeat(lc_reshape, 1, self.num_n)
        # (n, num_node) -> (n, num_node, h*w)
        lc_n_repeat = expandAndRepeat(lc_n, 2, lc_reshape_repeat.shape[-1])
        x_proj_reshaped *= (lc_n_repeat == lc_reshape_repeat)

        # $ B^T $
        x_rproj_reshaped = x_proj_reshaped

        # # 3. Heterogeneous GCN
        # 3.1 Graph projection
        # $ V = Φ(x) * B $
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # 3.2 GCN
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        # Heterogeneous graph
        x_n_rel = self.inter_gcn((x_n_state, lc_n))
        if isinstance(x_n_rel, tuple):
            x_n_rel = x_n_rel[0]

        # 3.3 Graph reprojection
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h, w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *featsize)  # !

        # # 4. Feature fusion
        # 4.1 Fusion
        x = feat + self.blocker(self.fc_2(x_state))

        # 4.2 Conv after fusion
        x = self.conv52(x)

        # # 5. Classify
        x = self.conv6(x)

        # Class-wise classifier
        # (n, h, w) -> (n, h, w, lcNum) -> (n, lcNum, h, w)
        lc_onehot_init = lc_onehot_init[..., ::rate, ::rate]
        # (n, luNum * lcNum, h, w) -> luNum * (n, lcNum, h, w)
        x = torch.split(x, self.lcClassNum, dim=1)
        # luNum * (n, h, w) -> (n, luNum, h, w)
        x = torch.stack([torch.sum(xx * lc_onehot_init, dim=1) for xx in x], dim=1)

        # # 6. Upsample
        x = interpolate(x, imsize, **up_kwargs)

        return x