# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@notice  : Dataset
"""

import numpy as np
from os.path import join
from torch.utils import data
import torch
from config import opt
from .RasterToolbox import readTif


class datasetCNNLC(data.Dataset):

    def __init__(self, tileIds):
        self.imgdir = opt.imgdir
        self.patchsize = opt.patchsize
        self.lcdir = opt.label5dir
        self.labeldir = opt.label15dir
        self.indim = opt.indim
        self.outdim = opt.outdim

        with open(tileIds, "r") as f:
            self.patchList = f.read().split("\n")
        
        meanStdfp = r"./dataset/divide/MeanStdArr.npy"
        self.meanStd = np.load(meanStdfp)

    def __getitem__(self, index):
        '''
        一次返回一张图片所有节点的特征、邻接矩阵、节点的地表覆盖标签、节点的公园标签
        '''
        patchfn = self.patchList[index]
        patchInfo = patchfn.split("-")
        fn, xoff, yoff = f"{patchInfo[0]}-{patchInfo[1]}.tif", int(patchInfo[2]), int(patchInfo[3])

        img = readTif(join(self.imgdir, fn), xoff, yoff, self.patchsize, self.patchsize) / 255.0
        for bandIdx in range(len(img)):
            img[bandIdx] = (img[bandIdx] - self.meanStd[bandIdx, 0]) / self.meanStd[bandIdx, 1]
        lc = readTif(join(self.lcdir, fn), xoff, yoff, self.patchsize, self.patchsize)
        label = readTif(join(self.labeldir, fn), xoff, yoff, self.patchsize, self.patchsize)
        label[label > self.outdim] = 0  # snow, bareland
        mask = label > 0

        # label remap
        lc[mask] -= 1  # [0, 5] => [0, 4]
        label[mask] -= 1  # [0, 15] => [0, 14]

        if self.indim == 3:
            img = img[1:]

        return torch.from_numpy(img).to(torch.float32), \
               torch.from_numpy(label).long(), \
               torch.from_numpy(lc).long(), \
               torch.from_numpy(mask), \
               patchfn

    def __len__(self):
        return len(self.patchList)