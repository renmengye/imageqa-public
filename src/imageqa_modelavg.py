import sys
import os

import numpy as np

import imageqa_test as it
import nn

def runAvgAll(models, data):
    print 'Running model %s' % modelId
    modelOutput = nn.test(model, data['testData'][0])
    modelOutputs.append(modelOutput)
    finalOutput = np.zeros(modelOutputs[0].shape)
    for output in modelOutputs:
        shape0 = min(finalOutput.shape[0], output.shape[0])
        shape1 = min(finalOutput.shape[1], output.shape[1])
        finalOutput[:shape0, :shape1] += output[:shape0, :shape1] / float(len(modelOutputs))
    return finalOutput

def testAvgAll(modelOutputs, mixRatio, data, outputFolder):
    # finalOutput = mixRatio * modelOutputs[0] + \
    #     (1 - mixRatio) * modelOutputs[1]
    finalOutput = np.zeros(modelOutputs[0].shape)
    for output in modelOutputs:
        shape0 = min(finalOutput.shape[0], output.shape[0])
        shape1 = min(finalOutput.shape[1], output.shape[1])
        finalOutput[:shape0, :shape1] += output[:shape0, :shape1] / float(len(modelOutputs))
    testAnswerFile = it.getAnswerFilename(outputFolder, resultsFolder)
    testTruthFile = it.getTruthFilename(outputFolder, resultsFolder)
    resultsRank, \
    resultsCategory, \
    resultsWups = it.runAllMetrics(
        data['testData'][0],
        finalOutput,
        data['testData'][1],
        data['ansIdict'],
        data['questionTypeArray'],
        testAnswerFile,
        testTruthFile)
    it.writeMetricsToFile(
        outputFolder,
        resultsRank,
        resultsCategory,
        resultsWups,
        resultsFolder)

def testAvg(modelOutputs, mixRatio, target):
    finalOutput = mixRatio * modelOutputs[0] + \
        (1 - mixRatio) * modelOutputs[1]
    rate, _, __ = it.calcPrecision(finalOutput, target)
    return rate

def validAvg(modelOutputs, mixRatios, target):
    bestRate = 0.0
    bestMixRatio = 0.0
    for mixRatio in mixRatios:
        rate = testAvg(modelOutputs, mixRatio, target)
        print 'Mix ratio %.4f Rate %.4f' % (mixRatio, rate)
        if rate > bestRate:
            bestMixRatio = mixRatio
            bestRate = rate
    return bestMixRatio

if __name__ == '__main__':
    """
    Usage: python imageqa_modelavg.py
            -m[odel] {modelId1}
            -m[odel] {modelId2}
            -vm[odel] {validModelId1}
            -vm[odel] {validModelId2}
            -d[ata] {dataFolder}
            -o[utput] {outputFolder}
            [-r[esults] {resultsFolder}]
    """
    resultsFolder = '../results'
    modelIds = []
    validModelIds = []
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            modelIds.append(sys.argv[i + 1])
        elif flag == '-vm' or flag == '-vmodel':
            validModelIds.append(sys.argv[i + 1])
        elif flag == '-r' or flag == '-results':
            resultsFolder = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            outputFolder = sys.argv[i + 1]
    data = it.loadDataset(dataFolder)
    
    models = []
    validModels = []
    for modelId in modelIds:
        print 'Loading model %s' % modelId
        models.append(it.loadModel(modelId, resultsFolder))
    for modelId in validModelIds:
        print 'Loading model %s' % modelId
        validModels.append(it.loadModel(modelId, resultsFolder))

    modelOutputs = []
    validModelOutputs = []
    # for modelId, model in zip(validModelIds, validModels):
    #     print 'Running model %s' % modelId
    #     modelOutput = nn.test(model, data['validData'][0])
    #     validModelOutputs.append(modelOutput)
    # 
    # mixRatios = np.arange(0, 11) * 0.1
    # bestMixRatio = validAvg(validModelOutputs, mixRatios, data['validData'][1])
    # print 'Best ratio found: %.4f' % bestMixRatio
    bestMixRatio = 0.5
    shape = None
    for modelId, model in zip(modelIds, models):
        print 'Running model %s' % modelId
        modelOutput = nn.test(model, data['testData'][0])
        if shape is None:
            shape = modelOutput.shape
        else:
            modelOutput = modelOutput[:shape[0],:shape[1]]
        modelOutputs.append(modelOutput)

    testAvgAll(modelOutputs, bestMixRatio, data, outputFolder)
