#coding:utf8
import os
from os.path import join, exists
import warnings


class DefaultConfig(object):
    # dataset
    patchsize = 256
    dataset = "datasetCNNLC"
    labelName = "GID15"  # "GID24"
    dataRoot = "./dataset/data"
    imgdir = join(dataRoot, "img")
    label5dir = join(dataRoot, "label-5")
    label15dir = join(dataRoot, "label-15")
    divideDir = f"./dataset/divide"
    trainTileIds = join(divideDir, "train.txt")
    valTileIds = join(divideDir, "val.txt")
    testTileIds = join(divideDir, "test.txt")
    lcClassNum = 5
    label5List = ["builtup", "farmland", "forest", "meadow", "water"]
    label15List = [
        "Ind. land", "Urb. resid.", "Rur. resid.", "Traff. land", "P. field", "Irr. land", "Dry cropl.", "Garden",
        "Arb. forest", "Shr. land", "Nat. mead.", "Art. mead.", "River", "Lake", "Pond"
    ]
    label24List = [
        "Indu", "Urba", "Rura", "Road", "Padd", "Irri", "Dryc", "Gard", "Arbo", "Shru", "Natu", "Arti", "Rive",
        "Lake", "Pond", "Stad", "Squa", "Over", "Rail", "Airp", "Park", "Fish"
    ]
    outputDir = './outputs'
    
    # model
    labelNameDict = {"GID15": 15, "GID24": 22}
    indim = 3
    outdim = labelNameDict[labelName]
    model = 'FCN'  # model Name
    backbone = "resnet50"
    isBackBoneFrozen = True
    isLCEncoder = True
    backboneFeatureLayer = "c3"
    isMultiClassifier = True
    GCNLayerNum = 1
    nodeNumRate = 1
    tag = model  # output tag

    # train
    seed = 2023
    train = False
    createTif = False
    loss = "CrossEntropyLoss"
    isAccCal = True
    testModel = None
    batchSize = 8
    useGpu = True
    deviceId = None  # None: use the last one by default
    numWorkers = 2
    saveFreq = 5
    valStep = 5
    maxEpoch = 50
    lrMax = 0.001
    lrMode = "poly"  # const
    weightDecay = 1e-4


def parse(self, kwargs):
    '''
    update parameters
    '''
    # Update parameters
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

        if k == "labelName":
            self.outdim = self.labelNameDict[self.labelName]
            self.label5dir = join(self.dataRoot, "label-5")
            if self.labelName == "GID15":
                self.label15dir = join(self.dataRoot, "label-15")
            elif self.labelName == "GID24":
                self.label15dir = join(self.dataRoot, "label-24")
            else:
                raise ValueError("labelName must be GID15 or GID24")

    # The output path of paramters
    paraSaveDir = join(self.outputDir, self.tag)
    if not exists(paraSaveDir):
        os.makedirs(paraSaveDir, exist_ok=True)
    paraSavePath = join(paraSaveDir, "hyperParas.txt")
    if self.train:
        with open(paraSavePath, "w") as f:
            f.write("")

    print('user config:')
    tplt = "{0:>20}\t{1:<10}"
    with open(paraSavePath, "a") as f:
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != "parse":
                value = str(getattr(self, k))
                print(tplt.format(k, value))
                if self.train:
                    f.write(tplt.format(k, value, chr(12288)) + "\n")

DefaultConfig.parse = parse
opt = DefaultConfig()