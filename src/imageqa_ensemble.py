import sys
import os
import nn
import numpy as np
import imageqa_test as it
import imageqa_visprior as ip
import imageqa_modelavg as ia

def loadEnsemble(
                    taskIds, 
                    resultsFolder):
    """
    Load class specific models.
    """
    models = []
    for taskId in taskIds:
        taskFolder = os.path.join(resultsFolder, taskId)
        modelSpec = os.path.join(taskFolder, '%s.model.yml' % taskId)
        modelWeights = os.path.join(taskFolder, '%s.w.npy' % taskId)
        model = nn.load(modelSpec)
        model.loadWeights(np.load(modelWeights))
        models.append(model)
    return models

def __runEnsemble(
                inputTest,
                models,
                ansDict,
                classAnsIdict,
                questionTypeArray):
    allOutput = []
    for i, model in enumerate(models):
        print 'Running test data on model #%d...' % i
        outputTest = nn.test(model, inputTest)
        allOutput.append(outputTest)
    ensembleOutputTest = np.zeros((inputTest.shape[0], len(ansDict)))
    for n in range(allOutput[0].shape[0]):
        qtype = questionTypeArray[n]
        output = allOutput[qtype]
        for i in range(output.shape[1]):
            ansId = ansDict[classAnsIdict[qtype][i]]
            ensembleOutputTest[n, ansId] = output[n, i]
    return ensembleOutputTest

def getClassDataFolders(dataset, dataFolder):
    """
    Get different original data folder name for class specific models.
    """
    if dataset == 'daquar':
        classDataFolders = [
            dataFolder + '-object',
            dataFolder + '-number',
            dataFolder + '-color'
        ]
    elif dataset == 'cocoqa':
        classDataFolders = [
            dataFolder + '-object',
            dataFolder + '-number',
            dataFolder + '-color',
            dataFolder + '-location'
        ]
    return classDataFolders

def runEnsemble(
                inputTest,
                models, 
                dataFolder, 
                classDataFolders,
                questionTypeArray):
    """
    Run a class specific model on any dataset.
    """
    data = it.loadDataset(dataFolder)
    classAnsIdict = []
    for df in classDataFolders:
        data_c = it.loadDataset(df)
        classAnsIdict.append(data_c['ansIdict'])

    ensembleOutputTest = __runEnsemble(
                                        inputTest, 
                                        models,
                                        data['ansDict'],
                                        classAnsIdict,
                                        questionTypeArray)
    return ensembleOutputTest

def testEnsemble(
                    ensembleId,
                    models,
                    dataFolder,
                    classDataFolders,
                    resultsFolder):
    """
    Test a class specific model in its original dataset.
    """
    data = it.loadDataset(dataFolder)
    inputTest = data['testData'][0]
    targetTest = data['testData'][1]

    ensembleOutputTest = runEnsemble(
                                    inputTest,
                                    models, 
                                    dataFolder, 
                                    classDataFolders,
                                    data['questionTypeArray'])
    ensembleAnswerFile = getAnswerFilename(ensembleId, resultsFolder)
    ensembleTruthFile = getTruthFilename(ensembleId, resultsFolder)

    rate, correct, total = nn.calcRate(
        model, ensembleOutputTest, data['testData'][1])
    print 'rate: %.4f' % rate
    resultsRank, \
    resultsCategory, \
    resultsWups = it.runAllMetrics(
                                inputTest,
                                ensembleOutputTest,
                                targetTest,
                                data['ansIdict'],
                                data['questionTypeArray'],
                                ensembleAnswerFile,
                                ensembleTruthFile)
    it.writeMetricsToFile(
                        ensembleId,
                        rate,
                        resultsRank,
                        resultsCategory,
                        resultsWups,
                        resultsFolder)

    return ensembleOutputTest

def runEnsemblePrior(
                        inputTest,
                        models, 
                        dataFolder, 
                        classDataFolders,
                        questionTypeArray):
    """
    Similar to "testEnsemble" in imageqa_test.
    Run visprior on number and color questions.
    """
    data = it.loadDataset(dataFolder)
    numAns = len(data['ansIdict'])
    outputTest = np.zeros((inputTest.shape[0], numAns))
    count = 0

    allOutput = []
    ensembleOutputTest = np.zeros((inputTest.shape[0], numAns))
    classAnsIdict = []

    for i, model in enumerate(models):
        data_m = it.loadDataset(classDataFolders[i])
        classAnsIdict.append(data_m['ansIdict'])
        tvData_m = ip.combineTrainValid(data_m['trainData'], data_m['validData'])
        print 'Running test data on model #%d...' % i
        if i == 0:
            # Object questions
            print 'No prior'
            outputTest = nn.test(model, data_m['testData'][0])
            print 'Accuracy:',
            print ip.calcRate(outputTest, data_m['testData'][1])
        elif i == 1 or i == 2 or i == 3:
            # Number and color and location questions
            print 'Prior'
            # Delta is pre-determined
            if i == 1:
                delta = 1e-6
                questionType = "number"
            elif i == 2:
                delta = 5e-4
                questionType = "color"
            elif i == 3:
                delta = 1.0
                questionType = "location"
            outputTest = ip.runVisPrior(
                                tvData_m,
                                data_m['testData'],
                                questionType,
                                model,
                                data_m['questionDict'],
                                data_m['questionIdict'],
                                len(data_m['ansIdict']),
                                delta)
        allOutput.append(outputTest)
    counter = [0, 0, 0, 0]
    for n in range(inputTest.shape[0]):
        qtype = questionTypeArray[n]
        output = allOutput[qtype]
        for i in range(output.shape[1]):
            ansId = data['ansDict'][classAnsIdict[qtype][i]]
            ensembleOutputTest[n, ansId] = output[counter[qtype], i]
        counter[qtype] += 1
    return ensembleOutputTest

def testEnsemblePrior(
                    ensembleId,
                    models,
                    dataFolder,
                    classDataFolders,
                    resultsFolder):
    data = it.loadDataset(dataFolder)
    inputTest = data['testData'][0]
    targetTest = data['testData'][1]
    ensembleOutputTest = runEnsemblePrior(
                                inputTest,
                                models, 
                                dataFolder, 
                                classDataFolders,
                                data['questionTypeArray'])
    ensembleAnswerFile = it.getAnswerFilename(ensembleId, resultsFolder)
    ensembleTruthFile = it.getTruthFilename(ensembleId, resultsFolder)

    rate, correct, total = nn.calcRate(
        model, ensembleOutputTest, data['testData'][1])
    print 'rate: %.4f' % rate
    resultsRank, \
    resultsCategory, \
    resultsWups = it.runAllMetrics(
                        inputTest,
                        ensembleOutputTest,
                        targetTest,
                        data['ansIdict'],
                        data['questionTypeArray'],
                        ensembleAnswerFile,
                        ensembleTruthFile)
    it.writeMetricsToFile(
                        ensembleId,
                        rate,
                        resultsRank,
                        resultsCategory,
                        resultsWups,
                        resultsFolder)
    return ensembleOutputTest

def runAllModels(
                inputTest, 
                questionTypeArray, 
                modelSpecs,
                resultsFolder,
                dataset,
                dataFolder):
    allOutputs = []
    for modelSpec in modelSpecs:
        if modelSpec['isClassEnsemble']:
            print 'Running test data on ensemble model %s...' \
                    % modelSpec['name']
            models = loadEnsemble(modelSpec['id'].split(','), resultsFolder)
            classDataFolders = getClassDataFolders(dataset, dataFolder)
            if modelSpec['runPrior']:
                outputTest = runEnsemblePrior(
                                    inputTest, 
                                    models,
                                    dataFolder,
                                    classDataFolders,
                                    questionTypeArray)
            else:
                outputTest = runEnsemble(
                                    inputTest, 
                                    models,
                                    dataFolder,
                                    classDataFolders,
                                    questionTypeArray)
        elif modelSpec['isAverageEnsemble']:
            modelOutputs = []
            for modelId in modelSpec['id'].split(','):
                model = it.loadModel(modelId, resultsFolder)
                modelOutputs.append(nn.test(model, inputTest))
            outputTest = np.zeros(modelOutputs[0].shape)
            for output in modelOutputs:
                shape0 = min(outputTest.shape[0], output.shape[0])
                shape1 = min(outputTest.shape[1], output.shape[1])
                outputTest[:shape0, :shape1] += output[:shape0, :shape1] / \
                    float(len(modelOutputs))
        else:
            print 'Running test data on model %s...' \
                    % modelSpec['name']
            model = it.loadModel(modelSpec['id'], resultsFolder)
            outputTest = nn.test(model, inputTest)
        allOutputs.append(outputTest)
    return allOutputs

if __name__ == '__main__':
    """
    Test a type-specific ensemble model
    Usage:
    python imageqa_ensemble.py -e[nsemble] {ensembleId}
                               -m[odel] {modelId1}
                               -m[odel] {modelId2},...
                               -d[ata] {dataFolder}
                               -dataset {daquar/cocoqa}
                               [-r[esults] {resultsFolder}]
                               [-prior]
    Results folder by default is '../results'
    """
    resultsFolder = '../results'
    taskIds = []
    dataset = 'cocoqa'
    runPrior = False
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            taskIds.append(sys.argv[i + 1])
        elif flag == '-e' or flag == '-ensemble':
            ensembleId = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-r' or flag == '-results':
            resultsFolder = sys.argv[i + 1]
        elif flag == '-dataset':
            dataset = sys.argv[i + 1]
        elif flag == '-prior':
            runPrior = True
    models = loadEnsemble(taskIds, resultsFolder)
    classDataFolders = getClassDataFolders(dataset, dataFolder)
    if runPrior:
        testEnsemblePrior(
                ensembleId=ensembleId,
                models=models, 
                dataFolder=dataFolder, 
                classDataFolders=classDataFolders,
                resultsFolder=resultsFolder)
    else:
        testEnsemble(
                ensembleId=ensembleId,
                models=models, 
                dataFolder=dataFolder, 
                classDataFolders=classDataFolders,
                resultsFolder=resultsFolder)
