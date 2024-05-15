# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@notice  : train and test model
"""

import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path)

from os.path import join, exists
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import models
import dataset
from config import opt
from utils import setupSeed, getDevice, getLrSchedule, trainEpoch, testEpoch, drawLoss, drawLr


def train(**kwargs):
    # update settings
    opt.parse(kwargs)
    # set up random seed
    setupSeed(opt.seed)
    outputDir = join(opt.outputDir, opt.tag)

    # 1. prepare dataset
    trainDataset = getattr(dataset, opt.dataset)(opt.trainTileIds)
    valDataset = getattr(dataset, opt.dataset)(opt.valTileIds)
    trainDataloader = DataLoader(trainDataset, opt.batchSize, shuffle=True, pin_memory=True, num_workers=opt.numWorkers)
    valDataloader = DataLoader(valDataset, 1)

    # 2. define model
    device = getDevice()
    model = getattr(models, opt.model)(opt).to(device)

    # output the model structure
    modelOutputPath = join(outputDir, "model.txt")
    modelOutputMode = "a" if exists(modelOutputPath) else "w"
    with open(modelOutputPath, modelOutputMode, encoding="utf-8") as f:
        print(model, file=f)

    # 3. define the loss and optimizer
    criterion = getattr(torch.nn, opt.loss)().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad is True, model.parameters()), lr=opt.lrMax, weight_decay=opt.weightDecay)

    # 4. difine lr schedule
    lrScheduler = getLrSchedule(optimizer, opt.lrMode)

    # 5.start training
    # model save Dir
    checkpointsDir = join(outputDir, "checkpoints")
    if not exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)
    if not exists(checkpointsDir):
        os.makedirs(checkpointsDir, exist_ok=True)

    # train loss path and val loss path
    trainLossPath, valLossPath = join(outputDir, "trainLoss.txt"), join(outputDir, "valLoss.txt")
    if not exists(trainLossPath):
        with open(trainLossPath, "w", encoding="utf-8") as f:
            f.write("")
    else:
        print("there is already a trainLoss.txt!")
    if not exists(valLossPath):
        with open(valLossPath, "w", encoding="utf-8") as f:
            f.write("")
    else:
        print("there is already a valLoss.txt!")

    epochs = opt.maxEpoch
    trainLossList, valLossList, valAccList = [], [], []
    lrList = []

    with tqdm(total=epochs, unit='epoch', ncols=100, colour="green") as pbar:
        for epoch in range(epochs):
            # 5.1 train models
            trainLoss, trainAcc = trainEpoch(model, device, trainDataloader, criterion, optimizer, 1.0 * epoch / epochs)

            # 5.2 update lr
            lrScheduler.step()
            lrList.append(optimizer.param_groups[0]["lr"])

            # 5.3 save the training loss
            trainLossList.append(trainLoss)

            # 5.4 save the model
            if (epoch + 1) % opt.saveFreq == 0:  # epoch >= 69 and
                modelPath = join(checkpointsDir, f"epochs_{epoch+1}.pth")
                model.save(optimizer, modelPath)

            # 5.5 update pbar
            pbar.update(1)
            pbar.set_postfix({'lossEpoch': trainLoss, 'accEpoch': trainAcc})
            with open(join(outputDir, "trainLoss.txt"), "a", encoding="utf-8") as f:
                f.write(f"epoch{epoch+1}: lossEpoch_{trainLoss:.8} accEpoch_{trainAcc:.6}\n")

            # 5.6 validate the model
            if (epoch + 1) % opt.valStep == 0:
                valLoss, valAcc = testEpoch(model, device, valDataloader, criterion, epoch + 1, 'val')
                valLossList.append(valLoss)
                valAccList.append(valAcc)

                with open(join(outputDir, "valLoss.txt"), "a", encoding="utf-8") as f:
                    f.write(f"epoch{epoch+1}: lossEpoch_{valLoss:.8} accEpoch_{valAcc:.8}\n")

                # 5.7 draw the training loss and validation loss curve
                drawLoss(trainLossList, join(outputDir, "trainLoss.png"))
                drawLoss(valLossList, join(outputDir, "valLoss.png"))
                drawLoss(valAccList, join(outputDir, "valAcc.png"), mode="Acc")

                # 5.8 draw lr
                drawLr([lrList], ["lr"], join(outputDir, "lrScheduler.png"))


def test(**kwargs):
    # update settings
    opt.parse(kwargs)

    # 1. prepare dataset
    testDataset = getattr(dataset, opt.dataset)(opt.testTileIds)
    testDataloader = DataLoader(testDataset, 1)

    # 2. define model
    device = getDevice()
    model = getattr(models, opt.model)(opt).to(device)

    # 3. load Model
    if opt.testModel:
        testModelPath = join(opt.outputDir, opt.tag, "checkpoints", opt.testModel)
        model.load(testModelPath, None)

    # 5. loss
    criterion = getattr(torch.nn, opt.loss)()

    # 6. test
    testEpoch(model, device, testDataloader, criterion)


# import fire
if __name__ == "__main__":
    # fire.Fire()  # annotate it when debug
    # python main.py train --train=True --labelName=GID24 --model=GHGCN --tag=GID24_24-GHGCN
    # python main.py test --train=True --labelName=GID24 --model=GHGCN --tag=GID24_24-GHGCN
    
    # train(train=True,
    #       labelName="GID24",
    #       model="GHGCN",
    #       tag="GID24-GHGCN")
    
    test(
        labelName="GID15",
        model="GHGCN",
        # createTif=True,
        tag="GID15-GHGCN",
        testModel="epochs_50.pth",
    )
