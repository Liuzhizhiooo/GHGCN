# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@notice  : functions of model
"""

import os
import torch
import requests
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import Conv2d
from torchvision import models

def onehot(arr, classnum):
    """
    转成one-hot编码
    """
    return torch.eye(classnum)[arr].to(arr.device)


def expandAndRepeat(x, dim, num):
    if isinstance(dim, list):
        if len(dim) != len(num):
            raise ValueError(f"mismatch")
        for ddim, nnum in zip(dim, num):
            x = expandAndRepeat(x, ddim, nnum)
    else:
        repeatDim = [1 for _ in x.shape]
        repeatDim.insert(dim, num)
        x = x.unsqueeze(dim).repeat(tuple(repeatDim))
    return x



def downloadModel(url, fname):
    """
    download the pretrained model
    url: download url
    fname: save path
    """
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError("Failed downloading url %s"%url)
    total_length = r.headers.get('content-length')
    with open(fname, 'wb') as f:
        if total_length is None: # no content length header
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
        else:
            total_length = int(total_length)
            for chunk in tqdm(r.iter_content(chunk_size=1024),
                            total=int(total_length / 1024. + 0.5),
                            unit='KB', unit_scale=False, dynamic_ncols=True):
                f.write(chunk)


def getbackbone(backbone, pretrained):
    # load pretrained model
    modelUrl = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    modelDir = "./models/resnet/"
    if not os.path.exists(modelDir):
        os.makedirs(modelDir, exist_ok=True)
    modelPath = f"{modelDir}/{modelUrl.split('/')[-1]}"
    if not os.path.exists(modelPath):
        print("download pretrained model ...")
        downloadModel(modelUrl, modelPath)
    model = getattr(models, backbone)()
    
    # modify layer 3, stride: (2, 2) -> (1, 1)
    # the feature dim of layer 3 to [32, 32]
    model.layer3[0].downsample[0] = eval(str(model.layer3[0].downsample[0]).replace("stride=(2, 2)", "stride=(1, 1)"))
    model.layer3[0].conv2 = eval(str(model.layer3[0].conv2).replace("stride=(2, 2)", "stride=(1, 1)"))
    for idx in range(1, 6):
        model.layer3[idx].conv2 = eval(str(model.layer3[idx].conv2).replace("padding=(1, 1)", "padding=(2, 2), dilation=(2, 2)"))

    # load pretrained model
    if pretrained is True:
        model.load_state_dict(torch.load(modelPath))
        print("pretrained model load")
    return model