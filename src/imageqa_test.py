import sys
import os

import numpy as np

import calculate_wups
import nn

def decodeQuestion(
                    modelInput, 
                    questionIdict):
    sentence = ''
    for t in range(1, modelInput.shape[0]):
        if modelInput[t, 0] == 0:
            break
        sentence += questionIdict[modelInput[t, 0]- 1] + ' '
    sentence += '?'
    return sentence

def estimateQuestionType(question):
    if 'how many' in question:
        typ = 1
    elif question.startswith('what is the color') or \
        question.startswith('what') and 'colour' in question:
        typ = 2
    elif question.startswith('where'):
        typ = 3
    else:
        typ = 0
    return typ

def calcRate(
                modelInput, 
                modelOutput, 
                target, 
                questionIdict=None, 
                questionTypeArray=None):
    correct = np.zeros(4, dtype=int)
    total = np.zeros(4, dtype=int)
    for n in range(0, modelOutput.shape[0]):        
        sortIdx = np.argsort(modelOutput[n], axis=0)
        sortIdx = sortIdx[::-1]
        answer = sortIdx[0]
        if questionTypeArray is None:
            question = decodeQuestion(modelInput[n], questionIdict)
            typ = estimateQuestionType(question)
        else:
            typ = questionTypeArray[n]
        total[typ] += 1
        if answer == target[n, 0]:
            correct[typ] += 1
    rate = correct / total.astype('float')
    print 'object: %.4f' % rate[0]
    print 'number: %.4f' % rate[1]
    print 'color: %.4f' % rate[2]
    print 'location: %.4f' % rate[3]
    return correct, total

def calcPrecision(
                    modelOutput, 
                    target):
    # Calculate precision
    correctAt1 = 0
    correctAt5 = 0
    correctAt10 = 0
    for n in range(0, modelOutput.shape[0]):
        sortIdx = np.argsort(modelOutput[n], axis=0)
        sortIdx = sortIdx[::-1]
        for i in range(0, 10):
            if modelOutput.shape[1] < 10 and i > 0:
                break
            if len(target.shape) == 2:
                ans = target[n, 0]
            else:
                ans = target[n]
            if sortIdx[i] == ans:
                if i == 0:
                    correctAt1 += 1
                if i <= 4:
                    correctAt5 += 1
                correctAt10 += 1
    r1 = correctAt1 / float(modelOutput.shape[0])
    r5 = correctAt5 / float(modelOutput.shape[0])
    r10 = correctAt10 / float(modelOutput.shape[0])
    print 'rate @ 1: %.4f' % r1
    print 'rate @ 5: %.4f' % r5
    print 'rate @ 10: %.4f' % r10
    return (r1, r5, r10)

def outputTxt(
                modelOutput, 
                target, 
                answerArray, 
                answerFilename, 
                truthFilename, 
                topK=1, 
                outputProb=False):
    """
    Output the results of all examples into a text file.
    topK: top k answers, separated by comma.
    outputProb: whether to output the probability of the answer as well.

    Format will look like this:
    q1ans1,0.99,a1ans2,0.01...
    q2ans1,0.90,q2ans2,0.02...
    """
    with open(truthFilename, 'w+') as f:
        for n in range(0, target.shape[0]):
            f.write(answerArray[target[n, 0]] + '\n')
    with open(answerFilename, 'w+') as f:
        for n in range(0, modelOutput.shape[0]):
            if topK == 1:
                f.write(answerArray[np.argmax(modelOutput[n, :])])
                if outputProb:
                    f.write(',%.4f' % modelOutput[n, np.argmax(modelOutput[n, :])])
                f.write('\n')
            else:
                sortIdx = np.argsort(modelOutput[n], axis=0)
                sortIdx = sortIdx[::-1]
                for i in range(0, topK):
                    f.write(answerArray[sortIdx[i]])
                    if outputProb:
                        f.write(',%.4f' % modelOutput[n, sortIdx[i]])
                    f.write('\n')

def runWups(
            answerFilename, 
            truthFilename):
    w10 = calculate_wups.runAll(truthFilename, answerFilename, -1)
    w09 = calculate_wups.runAll(truthFilename, answerFilename, 0.9)
    w00 = calculate_wups.runAll(truthFilename, answerFilename, 0.0)
    print 'WUPS @ 1.0: %.4f' % w10
    print 'WUPS @ 0.9: %.4f' % w09
    print 'WUPS @ 0.0: %.4f' % w00
    return (w10, w09, w00)

def getAnswerFilename(
                        taskId, 
                        resultsFolder):
    folder = os.path.join(resultsFolder, taskId)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(
                    folder, 
                    '%s.test.o.txt' % taskId)

def getTruthFilename(
                        taskId, 
                        resultsFolder):
    folder = os.path.join(resultsFolder, taskId)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(
                    folder,
                    '%s.test.t.txt' % taskId)

def loadDataset(dataFolder):
    print 'Loading dataset...'
    trainDataFile = os.path.join(dataFolder, 'train.npy')
    validDataFile = os.path.join(dataFolder, 'valid.npy')
    testDataFile = os.path.join(dataFolder, 'test.npy')
    vocabDictFile = os.path.join(dataFolder, 'vocab-dict.npy')
    qtypeFile = os.path.join(dataFolder, 'test-qtype.npy')
    trainData = np.load(trainDataFile)
    validData = np.load(validDataFile)
    testData = np.load(testDataFile)
    vocabDict = np.load(vocabDictFile)
    qTypeArray = np.load(qtypeFile)
    inputTest = testData[0]
    targetTest = testData[1]
    qDict = vocabDict[0]
    qIdict = vocabDict[1]
    aDict = vocabDict[2]
    aIdict = vocabDict[3]
    return {
            'trainData': trainData,
            'validData': validData,
            'testData': testData,
            'questionDict': qDict,
            'questionIdict': qIdict,
            'ansDict': aDict,
            'ansIdict': aIdict,
            'questionTypeArray': qTypeArray}

def loadModel(
                taskId,
                resultsFolder):
    print 'Loading model...'
    modelSpecFile = '%s/%s/%s.model.yml' % (resultsFolder, taskId, taskId)
    modelWeightsFile = '%s/%s/%s.w.npy' % (resultsFolder, taskId, taskId)
    model = nn.load(modelSpecFile)
    model.loadWeights(np.load(modelWeightsFile))
    return model

def testAll(
            modelId, 
            model, 
            dataFolder, 
            resultsFolder):
    testAnswerFile = getAnswerFilename(modelId, resultsFolder)
    testTruthFile = getTruthFilename(modelId, resultsFolder)
    data = loadDataset(dataFolder)
    outputTest = nn.test(model, data['testData'][0])
    rate, correct, total = nn.calcRate(model, outputTest, data['testData'][1])
    print 'rate: %.4f' % rate
    resultsRank, \
    resultsCategory, \
    resultsWups = runAllMetrics(
                        data['testData'][0],
                        outputTest,
                        data['testData'][1],
                        data['ansIdict'],
                        data['questionTypeArray'],
                        testAnswerFile,
                        testTruthFile)
    writeMetricsToFile(
                modelId,
                rate,
                resultsRank,
                resultsCategory,
                resultsWups,
                resultsFolder)
    return outputTest

def runAllMetrics(
                    inputTest,
                    outputTest, 
                    targetTest, 
                    answerArray, 
                    questionTypeArray, 
                    testAnswerFile, 
                    testTruthFile):
    outputTxt(outputTest, targetTest, answerArray, 
              testAnswerFile, testTruthFile)
    resultsRank = calcPrecision(outputTest, targetTest)
    correct, total = calcRate(inputTest, 
        outputTest, targetTest, questionTypeArray=questionTypeArray)
    resultsCategory = correct / total.astype(float)
    resultsWups = runWups(testAnswerFile, testTruthFile)
    return (resultsRank, resultsCategory, resultsWups)

def writeMetricsToFile(
                        taskId, 
                        rate,
                        resultsRank, 
                        resultsCategory, 
                        resultsWups, 
                        resultsFolder):
    resultsFile = os.path.join(resultsFolder, taskId, 'result.txt')
    with open(resultsFile, 'w') as f:
        f.write('rate: %.4f\n' % rate)
        f.write('rate @ 1: %.4f\n' % resultsRank[0])
        f.write('rate @ 5: %.4f\n' % resultsRank[1])
        f.write('rate @ 10: %.4f\n' % resultsRank[2])
        f.write('object: %.4f\n' % resultsCategory[0])
        f.write('number: %.4f\n' % resultsCategory[1])
        f.write('color: %.4f\n' % resultsCategory[2])
        f.write('location: %.4f\n' % resultsCategory[3])
        f.write('WUPS 1.0: %.4f\n' % resultsWups[0])
        f.write('WUPS 0.9: %.4f\n' % resultsWups[1])
        f.write('WUPS 0.0: %.4f\n' % resultsWups[2])

if __name__ == '__main__':
    """
    Usage python imageqa_test.py -m[odel] {Model ID} 
                                 -d[ata] {dataFolder} 
                                 [-r[esults] {resultsFolder}]
    """
    dataFolder = None
    resultsFolder = '../results'
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            modelId = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]        
        elif flag == '-r' or flag == '-result':
            resultsFolder = sys.argv[i + 1]
    print modelId
    model = loadModel(modelId, resultsFolder)
    testAll(modelId, model, dataFolder, resultsFolder)
