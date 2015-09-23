import sys

import numpy as np
import imageqa_test as it
import prep
import nn

def reindexDataset(
                srcQuestions,
                srcAnswers,
                srcQuestionIdict,
                dstQuestionDict,
                srcAnsIdict,
                dstAnsDict):
    dstQuestions = np.zeros(srcQuestions.shape, dtype='int')
    dstAnswers = np.zeros(srcAnswers.shape, dtype='int')
    for n in range(srcQuestions.shape[0]):
        dstQuestions[n, 0, 0] = srcQuestions[n, 0, 0]
        for t in range(1, srcQuestions.shape[1]):
            word = srcQuestionIdict[srcQuestions[n, t, 0] - 1]
            if dstQuestionDict.has_key(word):
                dstQuestions[n, t, 0] = dstQuestionDict[word]
            else:
                dstQuestions[n, t, 0] = dstQuestionDict['UNK']
        word = srcAnsIdict[srcAnswers[n, 0]]
        if dstAnsDict.has_key(word):
            dstAnswers[n, 0] = dstAnsDict[word]
        else:
            dstAnswers[n, 0] = dstAnsDict['UNK']
    return dstQuestions, dstAnswers

if __name__ == '__main__':
    """
    Usage: python imageqa_crosstest.py 
                -m[odel] {model id}
                -d[ata] {model data folder}
                -td[ata] {test data folder}
                [-reindex {whether to reindex the test data, default false}]
                [-r[esults] {results folder}]
                [-dataset {cocoqa/daquar, default cocoqa}]
    """
    resultsFolder = '../results'
    needReindex = False
    dataset = 'cocoqa'
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            modelId = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-td' or flag == '-tdata':
            testDataFolder = sys.argv[i + 1]
        elif flag == '-reindex':
            needReindex = True
        elif flag == '-r' or flag == '-results':
            resultsFolder = sys.argv[i + 1]
        elif flag == '-dataset':
            dataset = sys.argv[i + 1]
    
    model = it.loadModel(modelId, resultsFolder)
    data = it.loadDataset(dataFolder)
    testdata = it.loadDataset(testDataFolder)
    if needReindex:
        testQuestions, testAnswers = reindexDataset(
            testdata['testData'][0],
            testdata['testData'][1],
            testdata['questionIdict'],
            data['questionDict'],
            testdata['ansIdict'],
            data['ansDict'])
    else:
        testQuestions = testdata['testData'][0]
        testAnswers = testdata['testData'][1]
    outputTest = nn.test(model, testQuestions)
    rate, correct, total = nn.calcRate(model, outputTest, testAnswers)
    print 'rate: %.4f' % rate

