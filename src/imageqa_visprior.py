import sys
import os

import numpy as np
import nn
import imageqa_test as it
from nltk.corpus import wordnet

lexnameDict = {}

def lookupLexname(word):
    if lexnameDict.has_key(word):
        return lexnameDict[word]
    else:
        synsets = wordnet.synsets(word)
        # Just pick the first definition
        if len(synsets) > 0:
            lexname = synsets[0].lexname()
            lexnameDict[word] = lexname
            return lexname
        else:
            return None

def locateObjLocation(data, questionDict, questionIdict):
    """
    Locate the object of where questions.
    Very naive heuristic: take the noun immediately after "where".
    """
    where = questionDict['where']
    for t in range(data.shape[0] - 1):
        if data[t, 0] == where:
            for u in range(t + 1, data.shape[0]):
                word = questionIdict[data[u, 0] - 1]
                lexname = lookupLexname(word)
                if (lexname is not None and \
                    lexname.startswith('noun')) or \
                    (lexname is None):
                    return data[u, 0]
    print 'not found'
    return data[-1, 0]

def locateObjNumberNoun(data, questionDict, questionIdict):
    """
    Locate the object of how many questions.
    Very naive heuristic: take the noun immediately after "how many".
    """
    how = questionDict['how']
    many = questionDict['many']
    for t in range(data.shape[0] - 2):
        if data[t, 0] == how and \
            data[t + 1, 0] == many:
            for u in range(t + 2, data.shape[0]):
                word = questionIdict[data[u, 0] - 1]
                lexname = lookupLexname(word)
                if (lexname is not None and \
                    lexname.startswith('noun')) or \
                    (lexname is None):
                    return data[u, 0]
    print 'not found'
    return data[-1, 0]

def locateObjNumber(data, questionDict):
    """
    Locate the object of how many questions.
    Very naive heuristic: take the word immediately after "how many".
    """
    how = questionDict['how']
    many = questionDict['many']
    for t in range(data.shape[0] - 2):
        if data[t, 0] == how and \
            data[t + 1, 0] == many:
            return data[t + 2, 0]
    print 'not found'

def locateObjColor(data):
    tmp = 0
    for i in range(data.shape[0]):
        if data[i, 0] != 0:
            tmp = data[i, 0]
        else:
            return tmp

def extractObjId(
                    data, 
                    questionType, 
                    questionDict, 
                    questionIdict):
    objIds = []
    for n in range(data.shape[0]):
        if questionType == 'color':
            objId = locateObjColor(data[n])
        elif questionType == 'number':
            objId = locateObjNumberNoun(data[n], questionDict, questionIdict)
        elif questionType == 'location':
            objId = locateObjLocation(data[n], questionDict, questionIdict)
        objIds.append(objId)
    return np.array(objIds, dtype='int')

def reindexObjId(
                inputData, 
                objDict, 
                questionDict, 
                questionIdict, 
                questionType):
    questionIdictArray = np.array(questionIdict, dtype='object')
    objIds = extractObjId(
                            inputData, 
                            questionType, 
                            questionDict, 
                            questionIdict)
    objIds = objIds - 1
    obj = questionIdictArray[objIds]
    objIds2 = np.zeros(objIds.shape, dtype='int')
    for i in range(obj.shape[0]):
        if objDict.has_key(obj[i]):
            objIds2[i] = objDict[obj[i]]
        else:
            objIds2[i] = objDict['UNK']
    return objIds2

def buildObjDict(
                    trainData,
                    questionType,
                    questionDict,
                    questionIdict):
    objDict = {}
    objIdict = []
    objIds = extractObjId(
                        trainData[0], 
                        questionType, 
                        questionDict, 
                        questionIdict)
    objIds = objIds - 1
    questionIdictArray = np.array(questionIdict, dtype='object')
    objList = questionIdictArray[objIds]
    for obj in objList:
        if not objDict.has_key(obj):
            objDict[obj] = len(objIdict)
            objIdict.append(obj)
    objDict['UNK'] = len(objIdict)
    objIdict.append('UNK')
    return objDict, objIdict

def trainCount(
                trainData, 
                questionType,
                questionDict,
                questionIdict, 
                objDict, 
                objIdict,
                numAns):
    """
    Calculates count(w, a), count(a)
    """
    count_wa = np.zeros((len(objIdict), numAns))
    count_a = np.zeros((numAns))
    objIds = extractObjId(
                            trainData[0], 
                            questionType, 
                            questionDict, 
                            questionIdict)
    for i in range(objIds.shape[0]):
        objId = objIds[i]
        obj = questionIdict[objId - 1]
        ansId = trainData[1][i, 0]
        objId2 = objDict[obj]
        count_wa[objId2, ansId] += 1
        count_a[ansId] += 1
    # Add UNK count
    count_a[-1] += 1
    return count_wa, count_a

def runVisPriorOnce(
                    objId, 
                    count_wa, 
                    count_a, 
                    modelOutput, 
                    delta):
    P_w_a = count_wa[objId, :]
    P_w_a /= count_a[:] 
    P_w_a += delta
    P_w_a /= (modelOutput.shape[1] * delta + 1)

    # (n, c)
    P_a_i = modelOutput

    # (n, c)
    P_wai = P_w_a * P_a_i
    P_a_wi = P_wai / np.sum(P_wai, axis=1).reshape(P_wai.shape[0], 1)
    return P_a_wi

def calcRate(output, target):
    outputMax = np.argmax(output, axis=-1)
    outputMax = outputMax.reshape(outputMax.size)
    targetReshape = target.reshape(target.size)
    equals = (outputMax == targetReshape).astype('int')
    rate = np.sum(equals) / \
            float(target.size)
    return rate, outputMax, equals

def validDelta(
                trainData,
                validData,
                preVisModelOutput,
                questionDict,
                questionIdict,
                numAns,
                deltas,
                questionType):
    objDict, objIdict = buildObjDict(
                                trainData,
                                questionType,
                                questionDict,
                                questionIdict)
    count_wa, count_a = trainCount(
                                trainData, 
                                questionType,
                                questionDict,
                                questionIdict,
                                objDict,
                                objIdict,
                                numAns)
    print count_wa

    # Reindex valid set
    validInput = validData[0]
    validTarget = validData[1]
    validTargetReshape = validTarget.reshape(validTarget.size)
    validObjId = reindexObjId(
                                validInput, 
                                objDict, 
                                questionDict, 
                                questionIdict, 
                                questionType)

    # Run vis model on valid set
    validOutput = nn.test(preVisModel, validInput)
    print 'Before Prior Valid Accuracy:',
    rate, _, __ = calcRate(validOutput, validTarget)
    print rate

    # Determine best delta
    bestRate = 0.0
    bestDelta = 0.0
    for delta in deltas:
        visPriorOutput = runVisPriorOnce(
                                validObjId, 
                                count_wa, 
                                count_a, 
                                validOutput, 
                                delta)        
        print 'delta=%f Valid Accuracy:' % delta,
        rate, _, __ = calcRate(visPriorOutput, validTarget)
        print rate
        if rate > bestRate:
            bestRate = rate
            bestDelta = delta
    print 'Best Delta:', bestDelta
    return bestDelta

def runVisPrior(
                trainData,
                testData,
                questionType,
                visModel,
                questionDict,
                questionIdict,
                numAns,
                delta):
    objDict, objIdict = buildObjDict(
                                trainData, 
                                questionType,
                                questionDict,
                                questionIdict)

    count_wa, count_a = trainCount(
                                trainData, 
                                questionType,
                                questionDict,
                                questionIdict,
                                objDict,
                                objIdict,
                                numAns)
    print count_wa

    # Reindex test set
    testInput = testData[0]
    testTarget = testData[1]
    testTargetReshape = testTarget.reshape(testTarget.size)
    testObjId = reindexObjId(
                                testInput, 
                                objDict, 
                                questionDict, 
                                questionIdict, 
                                questionType)

    # Run vis model on test set
    testOutput = nn.test(visModel, testInput)

    print 'Before Prior Test Accuracy:',
    rate, _, __ = calcRate(testOutput, testTarget)
    print rate
    
    # Run on test set
    visPriorOutput = runVisPriorOnce(
                                testObjId, 
                                count_wa, 
                                count_a, 
                                testOutput, 
                                delta)
    print 'delta=%f Test Accuracy:' % delta,
    rate, _, __ = calcRate(visPriorOutput, testTarget)
    print rate
    return visPriorOutput
 
def combineTrainValid(trainData, validData):
    trainDataAll = (np.concatenate((trainData[0], validData[0]), axis=0),
                    np.concatenate((trainData[1], validData[1]), axis=0))
    return trainDataAll

def calcAdaBoostAlpha(testOutput, testTarget):
    print 'Calculating alpha for boosting...'
    rate, _, correct = calcRate(testOutput, testTarget)
    alpha = np.log(rate / (1 - rate)) + np.log(float(testOutput.shape[1] - 1))
    print 'alpha:', alpha
    return alpha

def calcAdaBoostWeights(trainOutput, trainTarget, alpha):
    print 'Calculating weights for boosting...'
    rate, _, correct = calcRate(trainOutput, trainTarget)
    print correct
    print 'Train set rate:', rate
    correct2 = -(correct.astype('float32') - 0.5) * 2
    weights = np.exp(correct2 * alpha)
    weights /= np.sum(weights)
    weights *= weights.shape[0]
    print 'weights:', weights
    return weights

if __name__ == '__main__':
    """
    Usage:
    python imageqa_visprior.py
                                -pvid {preVisModelId}
                                -vid {visModelId}
                                -mid {mainModelId}
                                -bid {boostModelId}
                                -vd[ata] {visDataFolder}
                                -md[ata] {mainDataFolder}
                                -r[esults] {resultsFolder}
                                -qtype {color/number/location}
                                -o[utweights] {outputFolder}
    """
    questionType = 'color'
    visModelId = None
    mainModelId = None
    boostModelId = None
    outputWeightsFolder = None
    for i, flag in enumerate(sys.argv):
        if flag == '-pvid':
            preVisModelId = sys.argv[i + 1]
        elif flag == '-vid':
            visModelId = sys.argv[i + 1]
        elif flag == '-mid':
            mainModelId = sys.argv[i + 1]
        elif flag == '-bid':
            boostModelId = sys.argv[i + 1]
        elif flag == '-vd' or flag == '-vdata':
            visDataFolder = sys.argv[i + 1]
        elif flag == '-md' or flag == '-mdata':
            mainDataFolder = sys.argv[i + 1]
        elif flag == '-r' or flag == '-results':
            resultsFolder = sys.argv[i + 1]
        elif flag == '-qtype':
            questionType = sys.argv[i + 1]
        elif flag == '-o' or flag == '-outweights':
            outputWeightsFolder = sys.argv[i + 1]

    data = it.loadDataset(visDataFolder)
    testInput = data['testData'][0]
    testTarget = data['testData'][1]
    deltas = \
        [0.000001, 
        0.000005, 
        0.00001, 
        0.00005, 
        0.0001, 
        0.0005, 
        0.001, 
        0.005, 
        0.01, 
        0.05, 
        0.1, 
        0.5, 
        1.0]

    preVisModel = it.loadModel(preVisModelId, resultsFolder)

    print 'Num answer', len(data['ansIdict'])
    bestDelta = validDelta(
                            data['trainData'],
                            data['validData'],
                            preVisModel,
                            data['questionDict'],
                            data['questionIdict'],
                            len(data['ansIdict']),
                            deltas,
                            questionType)

    trainDataAll = combineTrainValid(data['trainData'], data['validData'])
    visModel = it.loadModel(visModelId, resultsFolder)
    visTestOutput = runVisPrior(trainDataAll,
                                data['testData'],
                                questionType,
                                visModel,
                                data['questionDict'],
                                data['questionIdict'],
                                len(data['ansIdict']),
                                bestDelta)

    visModelFolder = os.path.join(resultsFolder, visModelId)
    answerFilename = os.path.join(visModelFolder, 
                                visModelId + '_prior.test.o.txt')
    truthFilename = os.path.join(visModelFolder, 
                                visModelId + '_prior.test.t.txt')
    it.outputTxt(
                    visTestOutput, 
                    testTarget, 
                    data['ansIdict'], 
                    answerFilename, 
                    truthFilename, 
                    topK=1, 
                    outputProb=False)
    it.runWups(answerFilename, truthFilename)

    if mainModelId is not None:
        data_m = it.loadDataset(mainDataFolder)
        ansDict_m = data_m['ansDict']
        ansIdict = data['ansIdict']
        questionDict_m = data_m['questionDict']
        questionIdict = data['questionIdict']

        newTestInput = np.zeros(testInput.shape, dtype='int')
        for n in range(testInput.shape[0]):
            newTestInput[n, 0, 0] = testInput[n, 0, 0]
            for t in range(1, testInput.shape[1]):
                if testInput[n, t, 0] != 0:
                    word = questionIdict[testInput[n, t, 0] - 1]
                    newTestInput[n, t, 0] = questionDict_m[word]
                else:
                    break
        mainModel = it.loadModel(mainModelId, resultsFolder)
        mainTestOutput = nn.test(mainModel, newTestInput)

        # Need to extract the class output from mainTestOutput
        classNewId = []
        for ans in ansIdict:
            classNewId.append(ansDict_m[ans])
        classNewId = np.array(classNewId, dtype='int')
        mainTestOutput = mainTestOutput[:, classNewId]

        for i in range(len(ansIdict)):
            mixRatio = i / 10.0
            ensTestOutput = mixRatio * visTestOutput + \
                (1 - mixRatio) * mainTestOutput
            print '%.2f VIS+PRIOR & %.2f VIS+BLSTM Accuracy:' % \
                (mixRatio, 1 - mixRatio),
            rate, _, __ = calcRate(ensTestOutput, testTarget)
            print rate

    if boostModelId is not None:
        boostModel = it.loadModel(boostModelId, resultsFolder)
        boostTestOutput = nn.test(boostModel, testInput)
        alpha = calcAdaBoostAlpha(visTestOutput, testTarget)
        alphaBoost = calcAdaBoostAlpha(boostTestOutput, testTarget)
        finalTestOutput = (alpha * visTestOutput + \
                        alphaBoost * boostTestOutput) / \
                        (alpha + alphaBoost)
        rate, _, __ = calcRate(finalTestOutput, testTarget)
        answerFilename = os.path.join(visModelFolder, 
                                    visModelId + '_boost.test.o.txt')
        truthFilename = os.path.join(visModelFolder, 
                                    visModelId + '_boost.test.t.txt')
        it.outputTxt(
                        finalTestOutput, 
                        testTarget, 
                        data['ansIdict'], 
                        answerFilename, 
                        truthFilename, 
                        topK=1, 
                        outputProb=False)
        it.runWups(answerFilename, truthFilename)

    if outputWeightsFolder is not None:
        if not os.path.exists(outputWeightsFolder):
            os.makedirs(outputWeightsFolder)
        alpha = calcAdaBoostAlpha(visTestOutput, testTarget)
        visTrainOutput = runVisPrior(trainDataAll,
                                trainDataAll,
                                questionType,
                                visModel,
                                data['questionDict'],
                                data['questionIdict'],
                                len(data['ansIdict']),
                                bestDelta)
        weights = calcAdaBoostWeights(visTrainOutput, trainDataAll[1], alpha)
        trainWeights = weights[:data['trainData'][1].shape[0]]
        validWeights = weights[trainWeights.shape[0]:]
        np.save(os.path.join(outputWeightsFolder, 'adb-weights-train.npy'), trainWeights)
        np.save(os.path.join(outputWeightsFolder, 'adb-weights-valid.npy'), validWeights)