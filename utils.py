import os
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from os.path import join
import sklearn.metrics as skmetrics
from dataset.RasterToolbox import createSameFormatTif, setColorTable
from config import opt

colorDict15 = {
    (0, 0, 0): 0,
    (200, 0, 0): 1,
    (250, 0, 150): 2,
    (200, 150, 150): 3,
    (250, 150, 150): 4,
    (0, 200, 0): 5,
    (150, 250, 0): 6,
    (150, 200, 150): 7,
    (200, 0, 200): 8,
    (150, 0, 250): 9,
    (150, 150, 250): 10,
    (250, 200, 0): 11,  #
    (200, 200, 0): 12,
    (0, 0, 200): 13,  #
    (0, 150, 200): 14,
    (0, 200, 250): 15
}
colorDict24 = {
    (0, 0, 0): 0,
    (200, 0, 0): 1,
    (250, 0, 150): 2,
    (200, 150, 150): 3,
    (250, 150, 150): 4,
    (0, 200, 0): 5,
    (150, 250, 0): 6,  # 
    (150, 200, 150): 7,
    (200, 0, 200): 8,
    (150, 0, 250): 9,
    (150, 150, 250): 10,
    (250, 200, 0): 11,  #
    (200, 200, 0): 12,
    (0, 0, 200): 13,  #
    (0, 150, 200): 14,
    (0, 200, 250): 15,
    (250, 200, 150): 16,
    (150, 150, 0): 17,
    (250, 150, 0): 18,
    (250, 200, 250): 19,
    (200, 150, 0): 20,
    (200, 150, 200): 21,
    (150, 200, 250): 22,
    (250, 250, 250): 23,
    (200, 200, 200): 24
}

config = {
    # "font.family": 'Times New Roman',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def setupSeed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # False
        torch.backends.cudnn.deterministic = False  # True
        print(f"set seed {seed}")
    else:
        print("do not set seed")


def getDevice():
    if opt.useGpu and torch.cuda.is_available():
        # use the last GPU by default
        deviceId = opt.deviceId if opt.deviceId != None else torch.cuda.device_count() - 1
        device = torch.device(deviceId)
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{deviceId}"
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    return device


def getLrSchedule(optimizer, mode="poly"):
    """
    get the learning rate schedule
    """
    # gamma = 0.97
    # warmUpEpochs = 50
    # TMaxEpochs = 20

    if mode == "const":
        lrLambda = lambda iter: 1
    elif mode == "poly":
        power = 0.9
        lrLambda = lambda iter: (1 - iter / opt.maxEpoch)**power
    else:
        info = f"lr mode '{mode}' illegal!"
        raise ValueError(info)

    lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrLambda)
    return lrScheduler


def drawClassificationMap(arr, patchfn):
    testTileIds = os.path.basename(opt.testTileIds)[:-4]
    dstdir = join(opt.outputDir, opt.tag, "classficationTif", f"{opt.testModel[:-4]}_{testTileIds}")
    os.makedirs(dstdir, exist_ok=True)

    dstfp = join(dstdir, f"{patchfn}.tif")
    reffn = f"{'-'.join(patchfn.split('-')[:2])}.tif"
    reffp = join(opt.dataRoot, "label-15", reffn)
    createSameFormatTif(arr, dstfp, reffp)
    colorDict = colorDict15 if opt.labelName == "GID15" else colorDict24
    setColorTable(dstfp, colorDict)


def drawLoss(loss, savePath, mode="loss"):
    fig, ax = plt.subplots()
    ax.set_ylabel(f'epoch {mode}')
    ax.set_xlabel('epoch')
    ax.plot(np.arange(len(loss)) + 1, loss, 'b-')
    if "train" in savePath:
        ax.set_title('train loss curve')
    elif "val" in savePath:
        ax.set_title(f"val {mode} curve")
        ax.set_xticks(np.arange(len(loss)) + 1)
        ax.set_xticklabels((np.arange(len(loss)) + 1) * opt.valStep)
    else:
        pass
    plt.savefig(savePath, dpi=500, bbox_inches='tight')
    plt.close("all")


def drawLr(LrList, LabelList, savePath):
    assert len(LrList) == len(LabelList), f"LrList dismatch LabelList!"
    colorList = ["orange", "g", "b", "c"]
    fig, ax = plt.subplots()
    ax.set_title('Learning rate curve')
    ax.set_ylabel('lr')
    ax.set_xlabel('epoch')
    for idx, lrCurve in enumerate(LrList):
        ax.plot(np.arange(len(lrCurve)) + 1, lrCurve, '-', color=colorList[idx], label=LabelList[idx])
    ax.legend(LabelList, loc=1, fontsize=14)
    plt.savefig(savePath, dpi=500, bbox_inches='tight')
    plt.close("all")


def maskImg(img, mask, permute=True):
    """
    [B, C, H , W] => [N, C]
    img: [B, C, H, W]
    mask: [B, H, W]
    """
    if img.ndim == 4:
        # [B, C, H, W] => [C, B, H, W]
        img = img.permute(1, 0, 2, 3)
    # mask, [C, B, H, W] => [C, N]
    img = img[..., mask]
    if permute is True and img.ndim == 2:
        # [C, N] => [N, C]
        img = img.permute(1, 0)
    return img


def metricsCal(confusionMat):
    """
    基于混淆矩阵计算精度
    """
    # prepare
    eps = 1e-8
    tp = np.diag(confusionMat)
    fp = np.sum(confusionMat, axis=0) - tp
    fn = np.sum(confusionMat, axis=1) - tp
    tn = []
    for i in range(confusionMat.shape[1]):
        tmp = np.delete(confusionMat, i, 0)  # delete ith row
        tmp = np.delete(tmp, i, 1)  # delete ith column
        tn.append(sum(sum(tmp)))
    tn = np.array(tn)
    sampleNum = sum(sum(confusionMat))

    # metrics
    # OA
    oa = (sum(tp) + eps) / (sampleNum + eps)

    # Kappa
    nClasses = confusionMat.shape[0]
    expected = np.outer(fp + tp, fn + tp) / np.sum(fp + tp)
    w_mat = np.ones([nClasses, nClasses], dtype=int)
    w_mat.flat[::nClasses + 1] = 0
    k = np.sum(w_mat * confusionMat) / np.sum(w_mat * expected)
    kappa = 1 - k

    # IoU
    IoU = (tp + eps) / (tp + fp + fn + eps)
    mIoU = np.mean(IoU)

    # fwIoU
    freq = (tp + fn) / sampleNum
    FWIoU = (freq * IoU).sum()

    precision = np.mean((tp + eps) / (tp + fp + eps))
    recall = np.mean((tp + eps) / (tp + fn + eps))
    f1 = (2 * precision * recall) / (precision + recall)
    return IoU, dict(oa=oa, kappa=kappa, mIoU=mIoU, FWIoU=FWIoU, precision=precision, recall=recall, f1=f1)


def accCal(confusionMat, epoch, name):
    """
    精度指标计算
    """
    # 1. output path
    outputDir = join(opt.outputDir, opt.tag)
    if name.startswith("test"):
        outputPath = join(outputDir, f"{name}_acc_{opt.testModel}.txt")
    else:
        outputPath = join(outputDir, f"{name}_acc.txt")

    # 2. calculate indices
    IoU, scores = metricsCal(confusionMat)

    # 3. output indices
    if opt.labelName == "GID15":
        labelList = opt.label15List
    else:
        labelList = opt.label24List
    for idx, category in enumerate(labelList):
        scores[f"IoU-{category}"] = IoU[idx]
    scoresMsg = "\n".join([f"{k}={v:.4f}" for (k, v) in scores.items()])

    with open(outputPath, 'a', encoding="utf-8") as f:
        if name.startswith("test"):
            f.write(f"test: {opt.testTileIds}\n")
        if epoch:
            f.write(f"[epoch]:{epoch}\n")
        f.write('confusion matrix \n')
        np.savetxt(f, np.array(confusionMat), fmt='%d')
        f.write("\n" + scoresMsg)
        f.write("\n---------------------\n\n")

    if name.startswith("test"):
        print(name)
        print('confusion matrix \n')
        print(confusionMat)
        print("\n")
        print(scoresMsg)
        print("\n")


def trainEpoch(model, device, dataloader, criterion, optimizer, epoch, TVLoss=None):
    """
    train one epoch
    """
    model.train()
    lossEpoch, accEpoch = 0, 0

    with tqdm(total=len(dataloader), unit='batch', leave=False, ncols=100, colour="blue") as pbar:
        for idx, batchData in enumerate(dataloader):
            optimizer.zero_grad()

            # 1. data load
            img, gt = [x.to(device) for x in batchData[:2]]
            lcgt, mask = [x.to(device) for x in batchData[2:4]]
            pred = model(img, lcgt, mask)

            # 2. loss
            # mask
            # [B, C, H, W] => [N, C]
            pred, gt = maskImg(pred, mask), maskImg(gt, mask)
            loss = criterion(pred, gt)
            lossEpoch += loss.item()
            loss.backward()
            optimizer.step()

            # 3. acc calculation
            predLabel = torch.argmax(pred, 1).detach().to(torch.uint8)
            gt = gt.detach().to(torch.uint8)
            acc = (predLabel == gt).sum().item() / torch.numel(gt)
            accEpoch += acc / len(dataloader)

            pbar.update(1)
            if (idx + 1) % 10 == 0:
                pbar.set_postfix({'loss(batch)': loss.item() / dataloader.batch_size})
    return lossEpoch, accEpoch


def testEpoch(model, device, dataloader, criterion, epoch=None, name='test'):
    """
    test one epoch
    """
    model.eval()
    lossEpoch, accEpoch = 0, 0
    confusionMatAll = np.zeros((opt.outdim, opt.outdim), dtype=np.int64)
    with torch.no_grad():
        with tqdm(total=len(dataloader), unit='batch', ncols=100, colour="yellow", leave=False) as pbar:
            for idx, batchData in enumerate(dataloader):
                # 1. data load
                img, gt = [x.to(device) for x in batchData[:2]]
                lcgt, mask = [x.to(device) for x in batchData[2:4]]
                pred = model(img, lcgt, mask)

                # 2. mask
                # [B, C, H, W] => [N, C]
                maskPred, maskGt = maskImg(pred, mask), maskImg(gt, mask)

                # 3. loss calculation (for validation)
                loss = criterion(maskPred, maskGt)
                lossEpoch += loss.item()

                # 4. acc calculation
                maskGt = maskGt.detach().cpu().to(torch.uint8)
                maskPredLabel = torch.argmax(maskPred, 1).detach().cpu().to(torch.uint8)
                acc = (maskPredLabel == maskGt).sum().item() / torch.numel(maskGt)
                accEpoch += acc / len(dataloader)
                confusionMat = skmetrics.confusion_matrix(maskGt, maskPredLabel, labels=np.arange(opt.outdim))
                confusionMatAll += confusionMat

                # visualize the results
                patchfn = batchData[-1][0]
                if opt.train is False and opt.createTif is True:
                    predLabel = torch.argmax(pred, 1).detach().cpu().to(torch.uint8)
                    predLabel += 1  # [0, outdim-1] => [1, outdim]
                    predLabel[~mask] = 0
                    drawClassificationMap(predLabel.numpy(), patchfn)

                pbar.update(1)
                if (idx + 1) % 10 == 0:
                    pbar.set_postfix({'loss(batch)': loss.item() / dataloader.batch_size})

        # calculate indices
        if opt.isAccCal:
            accCal(confusionMatAll, epoch, name)

    return lossEpoch, accEpoch


if __name__ == "__main__":
    pass